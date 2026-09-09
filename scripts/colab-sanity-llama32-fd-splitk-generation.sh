#!/usr/bin/env bash
set -euo pipefail
# Phase 5 paired generation sanity. Cross-path text equality is diagnostic only.
[[ $# -eq 7 ]] || { echo "Usage: $0 HARNESS MODEL PROMPT PROMPT_TOKENS NEW_TOKENS GPU_LAYERS RESULT_DIR" >&2; exit 2; }
H=$1; M=$2; P=$3; PT=$4; N=$5; G=$6; O=$7
[[ -x "$H" && -f "$M" && -f "$P" ]] || { echo "Missing harness, model, or prompt." >&2; exit 2; }
mkdir -p "$O"
CTX=$((PT + N))
BATCH=$((PT > 2048 ? PT : 2048))
ARGS=(--model "$M" --prompt-file "$P" --context-tokens "$PT" --ctx-size "$CTX" --predict "$N" --gpu-layers "$G" --batch-size "$BATCH" --ubatch-size 512 --runs 3 --trace-logits --flash-attn on)
env -u GGML_CUDA_LLAMA32_FA_DECODE_ENABLED -u GGML_CUDA_LLAMA32_FD_SPLITK_ENABLED -u GGML_CUDA_LLAMA32_FA_DECODE_TRACE "$H" "${ARGS[@]}" > "$O/builtin.txt" 2> "$O/builtin.stderr"
env GGML_CUDA_LLAMA32_FA_DECODE_ENABLED=1 GGML_CUDA_LLAMA32_FD_SPLITK_ENABLED=1 GGML_CUDA_LLAMA32_FA_DECODE_TRACE=1 "$H" "${ARGS[@]}" > "$O/splitk.txt" 2> "$O/splitk.stderr"
for IMPL in builtin splitk; do
  FILE="$O/$IMPL.txt"
  for KEY in PROMPT_TOKEN_COUNT GENERATED_TOKEN_COUNT STOP_REASON LOGIT_TRACE_ENABLED LOGITS_FINITE INTRA_IMPLEMENTATION_DETERMINISTIC TIMED_RUNS; do
    VALUE="$(sed -n "s/^$KEY=//p" "$FILE")"
    [[ -n "$VALUE" ]] || { echo "FAIL: $IMPL missing $KEY" >&2; exit 1; }
  done
  [[ "$(sed -n "s/^PROMPT_TOKEN_COUNT=//p" "$FILE")" == "$PT" ]] || { echo "FAIL: $IMPL prompt length mismatch" >&2; exit 1; }
  [[ "$(sed -n "s/^TIMED_RUNS=//p" "$FILE")" == 3 && "$(sed -n "s/^INTRA_IMPLEMENTATION_DETERMINISTIC=//p" "$FILE")" == 1 ]] || { echo "FAIL: $IMPL determinism" >&2; exit 1; }
  [[ "$(sed -n "s/^LOGIT_TRACE_ENABLED=//p" "$FILE")" == 1 && "$(sed -n "s/^LOGITS_FINITE=//p" "$FILE")" == 1 ]] || { echo "FAIL: $IMPL finite logits" >&2; exit 1; }
  STOP="$(sed -n "s/^STOP_REASON=//p" "$FILE")"; COUNT="$(sed -n "s/^GENERATED_TOKEN_COUNT=//p" "$FILE")"
  [[ "$STOP" == eog || ( "$STOP" == max_tokens && "$COUNT" == "$N" ) ]] || { echo "FAIL: $IMPL incomplete generation" >&2; exit 1; }
  sed -n '/^GENERATED_TEXT_BEGIN$/,/^GENERATED_TEXT_END$/p' "$FILE" | sed '1d;$d' > "$O/$IMPL.generated.txt"
done
ROUTES="$(grep -c "^llama32-fd-splitk route: selected$" "$O/splitk.stderr" || true)"
[[ "$ROUTES" -gt 0 ]] || { echo "FAIL: Split-K route count is zero" >&2; exit 1; }
{
  echo "| implementation | prompt tokens | deterministic (3x) | stop reason | generated tokens | finite logits | splitk routes |"
  echo "| --- | ---: | --- | --- | ---: | --- | ---: |"
  echo "| builtin | $(sed -n 's/^PROMPT_TOKEN_COUNT=//p' "$O/builtin.txt") | PASS | $(sed -n 's/^STOP_REASON=//p' "$O/builtin.txt") | $(sed -n 's/^GENERATED_TOKEN_COUNT=//p' "$O/builtin.txt") | PASS | disabled |"
  echo "| Split-K | $(sed -n 's/^PROMPT_TOKEN_COUNT=//p' "$O/splitk.txt") | PASS | $(sed -n 's/^STOP_REASON=//p' "$O/splitk.txt") | $(sed -n 's/^GENERATED_TOKEN_COUNT=//p' "$O/splitk.txt") | PASS | $ROUTES |"
  cmp -s "$O/builtin.generated.txt" "$O/splitk.generated.txt" && echo "cross_implementation_text_equal=1" || echo "cross_implementation_text_equal=0"
} | tee "$O/summary.md"
echo "PASS: paired generation sanity completed; cross-path text equality is diagnostic only."
echo "Results: $O"
