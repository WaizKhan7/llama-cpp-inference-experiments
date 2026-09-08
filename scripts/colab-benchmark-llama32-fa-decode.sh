#!/usr/bin/env bash
set -euo pipefail

# Phase 6 decode-only benchmark. Run only after the Phase 1-5 correctness
# evidence has passed. Timed runs exclude model loading, prefill, warmup, and
# custom route tracing. A separate untimed traced custom probe proves dispatch.
[[ $# -ge 6 ]] || {
    echo "Usage: $0 HARNESS MODEL_GGUF PROMPT_FILE NEW_TOKENS RESULT_DIR GPU_LAYERS [CONTEXT ...]" >&2
    exit 2
}
H=$1; M=$2; P=$3; N=$4; O=$5; G=$6; shift 6
[[ $# -gt 0 ]] || set -- 128 512 2048 4096 8192
[[ -x "$H" && -f "$M" && -f "$P" ]] || {
    echo "Missing harness, model, or prompt." >&2; exit 2;
}
mkdir -p "$O"
{
    echo "Phase 6 Llama 3.2 FA-decode benchmark"
    echo "UTC: $(date -u --iso-8601=seconds)"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
} > "$O/environment.txt"
echo "context,prompt_tokens,timed_decode_steps,timed_runs,baseline_mean_ms,baseline_std_ms,baseline_tok_s,custom_mean_ms,custom_std_ms,custom_tok_s,speedup,custom_probe_routes" > "$O/summary.csv"

for C; do
    D="$O/context-$C"
    mkdir -p "$D"
    CTX=$((C + N))
    BATCH=$((C > 2048 ? C : 2048))

    env GGML_CUDA_LLAMA32_FA_DECODE_ENABLED=1 GGML_CUDA_LLAMA32_FA_DECODE_TRACE=1 \
        "$H" --model "$M" --prompt-file "$P" --context-tokens "$C" \
        --ctx-size "$CTX" --predict "$N" --gpu-layers "$G" \
        --batch-size "$BATCH" --ubatch-size 512 --runs 1 --trace-logits \
        --flash-attn on > "$D/custom-route-probe.txt" 2> "$D/custom-route-probe.stderr"
    ROUTES="$(grep -c '^llama32-fa-decode route: selected$' "$D/custom-route-probe.stderr" || true)"
    [[ "$ROUTES" -gt 0 ]] || { echo "FAIL: no custom route at context $C" >&2; exit 1; }

    env -u GGML_CUDA_LLAMA32_FA_DECODE_ENABLED -u GGML_CUDA_LLAMA32_FA_DECODE_TRACE \
        "$H" --model "$M" --prompt-file "$P" --context-tokens "$C" \
        --ctx-size "$CTX" --predict "$N" --gpu-layers "$G" \
        --batch-size "$BATCH" --ubatch-size 512 --warmup 2 --runs 5 \
        --flash-attn on > "$D/builtin.txt" 2> "$D/builtin.stderr"
    env GGML_CUDA_LLAMA32_FA_DECODE_ENABLED=1 \
        "$H" --model "$M" --prompt-file "$P" --context-tokens "$C" \
        --ctx-size "$CTX" --predict "$N" --gpu-layers "$G" \
        --batch-size "$BATCH" --ubatch-size 512 --warmup 2 --runs 5 \
        --flash-attn on > "$D/custom.txt" 2> "$D/custom.stderr"

    for IMPL in builtin custom; do
        FILE="$D/$IMPL.txt"
        for KEY in PROMPT_TOKEN_COUNT GENERATED_TOKEN_COUNT STOP_REASON TIMED_DECODE_STEPS TIMED_RUNS DECODE_MEAN_MS DECODE_STD_MS DECODE_TOKENS_PER_SECOND; do
            VALUE="$(sed -n "s/^$KEY=//p" "$FILE")"
            [[ -n "$VALUE" ]] || { echo "FAIL: $IMPL missing $KEY at context $C" >&2; exit 1; }
        done
        [[ "$(sed -n 's/^PROMPT_TOKEN_COUNT=//p' "$FILE")" == "$C" ]] || { echo "FAIL: $IMPL prompt length at context $C" >&2; exit 1; }
        [[ "$(sed -n 's/^GENERATED_TOKEN_COUNT=//p' "$FILE")" == "$N" ]] || { echo "FAIL: $IMPL did not generate $N tokens at context $C" >&2; exit 1; }
        [[ "$(sed -n 's/^STOP_REASON=//p' "$FILE")" == max_tokens ]] || { echo "FAIL: $IMPL stopped early at context $C" >&2; exit 1; }
        [[ "$(sed -n 's/^TIMED_RUNS=//p' "$FILE")" == 5 ]] || { echo "FAIL: $IMPL did not complete five timed runs at context $C" >&2; exit 1; }
    done

    BP="$(sed -n 's/^PROMPT_TOKEN_COUNT=//p' "$D/builtin.txt")"
    STEPS="$(sed -n 's/^TIMED_DECODE_STEPS=//p' "$D/builtin.txt")"
    RUNS="$(sed -n 's/^TIMED_RUNS=//p' "$D/builtin.txt")"
    BM="$(sed -n 's/^DECODE_MEAN_MS=//p' "$D/builtin.txt")"
    BS="$(sed -n 's/^DECODE_STD_MS=//p' "$D/builtin.txt")"
    BT="$(sed -n 's/^DECODE_TOKENS_PER_SECOND=//p' "$D/builtin.txt")"
    CM="$(sed -n 's/^DECODE_MEAN_MS=//p' "$D/custom.txt")"
    CS="$(sed -n 's/^DECODE_STD_MS=//p' "$D/custom.txt")"
    CT="$(sed -n 's/^DECODE_TOKENS_PER_SECOND=//p' "$D/custom.txt")"
    SPEEDUP="$(awk -v custom="$CT" -v baseline="$BT" 'BEGIN { if (baseline > 0) printf "%.6f", custom / baseline; else print "nan" }')"

    echo "$C,$BP,$STEPS,$RUNS,$BM,$BS,$BT,$CM,$CS,$CT,$SPEEDUP,$ROUTES" >> "$O/summary.csv"
    echo "context=$C baseline_tok_s=$BT custom_tok_s=$CT speedup="$SPEEDUP"x probe_routes=$ROUTES"
done

echo "PASS: Phase 6 decode benchmark completed after separate custom-route probes."
echo "Results: $O"
