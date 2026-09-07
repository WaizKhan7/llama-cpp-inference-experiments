#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 8 ]] || { echo "Usage: $0 HARNESS MODEL PROMPT NEW_TOKENS RESULTS TOKEN_VALIDATION_DIR FALLBACK_DIR GPU_LAYERS [CONTEXT ...]" >&2; exit 2; }
H=$1; M=$2; P=$3; N=$4; O=$5; V=$6; F=$7; G=$8; shift 8
[[ $# -gt 0 ]] || set -- 128 512 1024 2048 4096
[[ -x "$H" && -f "$M" && -f "$P" ]] || { echo "Missing harness, model, or prompt." >&2; exit 2; }
grep -q 'fallback_token_ids=PASS' "$F/summary.txt" || { echo "Refusing benchmark: no passing fallback validation." >&2; exit 1; }
for C; do grep -q 'token_ids=PASS' "$V/context-$C/summary.txt" || { echo "Refusing benchmark: no passing token validation at $C." >&2; exit 1; }; done
mkdir -p "$O"
echo 'context,baseline_ms,baseline_tok_s,custom_ms,custom_tok_s,speedup' > "$O/summary.csv"
for C; do
  D="$O/context-$C"; mkdir -p "$D"
  env -u GGML_CUDA_LLAMA32_FA_DECODE_ENABLED "$H" --model "$M" --prompt-file "$P" --context-tokens "$C" --predict "$N" --gpu-layers "$G" --flash-attn on --warmup 2 --runs 5 > "$D/builtin.txt" 2> "$D/builtin.stderr"
  env GGML_CUDA_LLAMA32_FA_DECODE_ENABLED=1 "$H" --model "$M" --prompt-file "$P" --context-tokens "$C" --predict "$N" --gpu-layers "$G" --flash-attn on --warmup 2 --runs 5 > "$D/custom.txt" 2> "$D/custom.stderr"
  BM=$(sed -n 's/^DECODE_MEAN_MS=//p' "$D/builtin.txt"); BT=$(sed -n 's/^DECODE_TOKENS_PER_SECOND=//p' "$D/builtin.txt")
  CM=$(sed -n 's/^DECODE_MEAN_MS=//p' "$D/custom.txt"); CT=$(sed -n 's/^DECODE_TOKENS_PER_SECOND=//p' "$D/custom.txt")
  S=$(awk -v c="$CT" -v b="$BT" 'BEGIN { printf "%.6f", c/b }')
  echo "$C,$BM,$BT,$CM,$CT,$S" >> "$O/summary.csv"
  echo "context=$C baseline=$BT custom=$CT speedup=$S"x
done
echo "PASS: benchmark ran only after exact-token validation."
echo "Results: $O"
