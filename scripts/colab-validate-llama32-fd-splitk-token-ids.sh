#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 7 ]] || { echo "Usage: $0 HARNESS MODEL_GGUF PROMPT_FILE NEW_TOKENS GPU_LAYERS RESULT_DIR CONTEXT..." >&2; exit 2; }
H=$1; M=$2; P=$3; N=$4; G=$5; O=$6; shift 6
[[ -x "$H" && -f "$M" && -f "$P" ]] || { echo "Missing harness, model, or prompt file." >&2; exit 2; }
mkdir -p "$O"
for C; do
  D="$O/context-$C"; mkdir -p "$D"
  U=$(( C < 512 ? C : 512 ))
  ARGS=(--model "$M" --prompt-file "$P" --context-tokens "$C" --ctx-size "$((C + N))" --predict "$N" --gpu-layers "$G" --batch-size "$C" --ubatch-size "$U" --flash-attn on)
  env -u GGML_CUDA_LLAMA32_FA_DECODE_ENABLED -u GGML_CUDA_LLAMA32_FD_SPLITK_ENABLED -u GGML_CUDA_LLAMA32_FA_DECODE_TRACE "$H" "${ARGS[@]}" > "$D/builtin.txt" 2> "$D/builtin.stderr"
  env GGML_CUDA_LLAMA32_FA_DECODE_ENABLED=1 GGML_CUDA_LLAMA32_FD_SPLITK_ENABLED=1 GGML_CUDA_LLAMA32_FA_DECODE_TRACE=1 "$H" "${ARGS[@]}" > "$D/splitk.txt" 2> "$D/splitk.stderr"
  BP=$(sed -n 's/^PROMPT_TOKEN_COUNT=//p' "$D/builtin.txt"); SP=$(sed -n 's/^PROMPT_TOKEN_COUNT=//p' "$D/splitk.txt")
  BI=$(sed -n 's/^GENERATED_TOKEN_IDS=//p' "$D/builtin.txt"); SI=$(sed -n 's/^GENERATED_TOKEN_IDS=//p' "$D/splitk.txt")
  BF=$(sed -n 's/^LOGITS_FINITE=//p' "$D/builtin.txt"); SF=$(sed -n 's/^LOGITS_FINITE=//p' "$D/splitk.txt")
  RC=$(grep -c '^llama32-fd-splitk route: selected$' "$D/splitk.stderr" || true)
  [[ "$BP" == "$C" && "$SP" == "$C" ]] || { echo "FAIL: exact prompt length failed at $C." >&2; exit 1; }
  [[ -n "$BI" && "$BI" == "$SI" ]] || { echo "FAIL: Split-K token IDs diverged at $C." >&2; exit 1; }
  [[ "$BF" == 1 && "$SF" == 1 ]] || { echo "FAIL: non-finite logits at $C." >&2; exit 1; }
  [[ "$RC" -gt 0 ]] || { echo "FAIL: Split-K route count is zero at $C." >&2; exit 1; }
  printf 'context=%s prompt_tokens=%s splitk_route_count=%s token_ids=PASS\n' "$C" "$SP" "$RC" | tee "$D/summary.txt"
done
echo "PASS: exact prompt lengths, token IDs, finite logits, and Split-K route evidence passed. Results: $O"
