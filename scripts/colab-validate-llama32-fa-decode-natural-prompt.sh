#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 6 ]] || { echo "Usage: $0 HARNESS MODEL_GGUF PROMPT_FILE NEW_TOKENS RESULT_DIR GPU_LAYERS [CONTEXT_CAPACITY ...]" >&2; exit 2; }

H=$1
M=$2
P=$3
N=$4
O=$5
G=$6
shift 6

[[ $# -gt 0 ]] || set -- 128 512 1024 2048 4096
[[ -x "$H" && -f "$M" && -f "$P" ]] || { echo "Missing harness, model, or prompt." >&2; exit 2; }

mkdir -p "$O"
for C; do
    D="$O/context-$C"
    mkdir -p "$D"

    env -u GGML_CUDA_LLAMA32_FA_DECODE_ENABLED -u GGML_CUDA_LLAMA32_FA_DECODE_TRACE \
        "$H" --model "$M" --prompt-file "$P" --ctx-size "$C" \
        --predict "$N" --gpu-layers "$G" --flash-attn on \
        > "$D/builtin.txt" 2> "$D/builtin.stderr"

    env GGML_CUDA_LLAMA32_FA_DECODE_ENABLED=1 GGML_CUDA_LLAMA32_FA_DECODE_TRACE=1 \
        "$H" --model "$M" --prompt-file "$P" --ctx-size "$C" \
        --predict "$N" --gpu-layers "$G" --flash-attn on \
        > "$D/custom.txt" 2> "$D/custom.stderr"

    BP=$(sed -n 's/^PROMPT_TOKEN_COUNT=//p' "$D/builtin.txt")
    CP=$(sed -n 's/^PROMPT_TOKEN_COUNT=//p' "$D/custom.txt")
    BI=$(sed -n 's/^GENERATED_TOKEN_IDS=//p' "$D/builtin.txt")
    CI=$(sed -n 's/^GENERATED_TOKEN_IDS=//p' "$D/custom.txt")
    RC=$(grep -c '^llama32-fa-decode route: selected$' "$D/custom.stderr" || true)

    [[ "$BP" == "$CP" && -n "$BI" && "$BI" == "$CI" ]] || {
        echo "FAIL: natural-prompt token IDs diverged at capacity $C." >&2
        exit 1
    }
    [[ "$RC" -gt 0 ]] || {
        echo "FAIL: custom route count is zero at capacity $C." >&2
        exit 1
    }

    echo "capacity=$C prompt_tokens=$BP route_count=$RC token_ids=PASS" | tee "$D/summary.txt"
done

echo "PASS: natural prompt matched at all requested capacities."
echo "Results: $O"
