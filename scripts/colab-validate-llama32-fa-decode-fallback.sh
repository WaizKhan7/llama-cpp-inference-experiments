#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 7 ]] || { echo "Usage: $0 HARNESS MODEL PROMPT CONTEXT NEW_TOKENS GPU_LAYERS RESULTS" >&2; exit 2; }
H=$1; M=$2; P=$3; C=$4; N=$5; G=$6; O=$7
[[ -x "$H" && -f "$M" && -f "$P" ]] || { echo "Missing harness, model, or prompt." >&2; exit 2; }
mkdir -p "$O"
env -u GGML_CUDA_LLAMA32_FA_DECODE_ENABLED -u GGML_CUDA_LLAMA32_FA_DECODE_TRACE "$H" --model "$M" --prompt-file "$P" --context-tokens "$C" --predict "$N" --gpu-layers "$G" --flash-attn off > "$O/builtin.txt" 2> "$O/builtin.stderr"
env GGML_CUDA_LLAMA32_FA_DECODE_ENABLED=1 GGML_CUDA_LLAMA32_FA_DECODE_TRACE=1 "$H" --model "$M" --prompt-file "$P" --context-tokens "$C" --predict "$N" --gpu-layers "$G" --flash-attn off > "$O/custom-enabled.txt" 2> "$O/custom-enabled.stderr"
B=$(sed -n 's/^GENERATED_TOKEN_IDS=//p' "$O/builtin.txt"); F=$(sed -n 's/^GENERATED_TOKEN_IDS=//p' "$O/custom-enabled.txt")
R=$(grep -c '^llama32-fa-decode route: selected$' "$O/custom-enabled.stderr" || true)
[[ -n "$B" && "$B" == "$F" ]] || { echo "FAIL: fallback token IDs differ." >&2; exit 1; }
[[ "$R" -eq 0 ]] || { echo "FAIL: custom route selected with Flash Attention off." >&2; exit 1; }
echo "fallback_token_ids=PASS" | tee "$O/summary.txt"
echo "route_count=0" | tee -a "$O/summary.txt"
echo "PASS: unsupported Flash-Attention-off workload used built-in fallback."
