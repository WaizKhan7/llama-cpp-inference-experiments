#!/usr/bin/env bash
set -euo pipefail

# Score the same held-out tokens with built-in and custom Q=1 decode paths.
[[ $# -eq 7 ]] || { echo "Usage: $0 HARNESS MODEL_GGUF HELDOUT_TEXT PREFILL_TOKENS SCORE_TOKENS GPU_LAYERS RESULT_DIR" >&2; exit 2; }
H=$1; M=$2; T=$3; PREFILL=$4; SCORE=$5; G=$6; O=$7
CTX_SIZE=$((PREFILL + SCORE))
BATCH_SIZE=$((PREFILL > 2048 ? PREFILL : 2048))
[[ -x "$H" && -f "$M" && -f "$T" ]] || { echo "Missing harness, model, or held-out text file." >&2; exit 2; }
mkdir -p "$O"

COMMON=(--model "$M" --prompt-file "$T" --score-prefill-tokens "$PREFILL" --score-tokens "$SCORE" --ctx-size "$CTX_SIZE" --gpu-layers "$G" --batch-size "$BATCH_SIZE" --ubatch-size 512 --flash-attn on)

env -u GGML_CUDA_LLAMA32_FA_DECODE_ENABLED -u GGML_CUDA_LLAMA32_FA_DECODE_TRACE "$H" "${COMMON[@]}" > "$O/builtin.txt" 2> "$O/builtin.stderr"
env GGML_CUDA_LLAMA32_FA_DECODE_ENABLED=1 GGML_CUDA_LLAMA32_FA_DECODE_TRACE=1 "$H" "${COMMON[@]}" > "$O/custom.txt" 2> "$O/custom.stderr"

ROUTE_COUNT=$(grep -c '^llama32-fa-decode route: selected$' "$O/custom.stderr" || true)
[[ "$ROUTE_COUNT" -gt 0 ]] || { echo "Custom FA-decode route was not selected." >&2; exit 1; }

python3 - "$O/builtin.txt" "$O/custom.txt" "$O/summary.txt" "$ROUTE_COUNT" "$PREFILL" "$SCORE" <<'PY'
import sys
from pathlib import Path

builtin_path, custom_path, summary_path = map(Path, sys.argv[1:4])
route_count, expected_prefill, expected_score = map(int, sys.argv[4:7])

def metrics(path):
    values = {}
    for line in path.read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            values[key] = value
    required = ('PROMPT_TOKEN_COUNT', 'CONTEXT_CAPACITY', 'SCORED_TOKEN_COUNT', 'MEAN_NEGATIVE_LOG_LIKELIHOOD', 'PERPLEXITY', 'SCORING_FINITE')
    missing = [key for key in required if key not in values]
    if missing:
        raise SystemExit(f'Missing {missing} in {path}')
    if int(values['PROMPT_TOKEN_COUNT']) != expected_prefill:
        raise SystemExit(f'Unexpected prefill length in {path}')
    if int(values['SCORED_TOKEN_COUNT']) != expected_score:
        raise SystemExit(f'Unexpected scored-token count in {path}')
    if int(values['SCORING_FINITE']) != 1:
        raise SystemExit(f'Non-finite scoring result in {path}')
    return float(values['MEAN_NEGATIVE_LOG_LIKELIHOOD']), float(values['PERPLEXITY'])

builtin_nll, builtin_ppl = metrics(builtin_path)
custom_nll, custom_ppl = metrics(custom_path)
summary = (
    f'prefill_tokens={expected_prefill} scored_tokens={expected_score} route_count={route_count} '
    f'builtin_mean_nll={builtin_nll:.9f} custom_mean_nll={custom_nll:.9f} '
    f'mean_nll_abs_diff={abs(builtin_nll-custom_nll):.9f} '
    f'builtin_ppl={builtin_ppl:.9f} custom_ppl={custom_ppl:.9f} '
    f'ppl_abs_diff={abs(builtin_ppl-custom_ppl):.9f} '
    f'ppl_relative_diff={abs(builtin_ppl-custom_ppl)/builtin_ppl:.9f}\n'
)
summary_path.write_text(summary)
print(summary, end='')
PY

echo "PASS: decode-style held-out perplexity comparison completed."
echo "Results: $O"
