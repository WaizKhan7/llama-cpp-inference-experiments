#!/usr/bin/env bash
set -euo pipefail

# Compare built-in and Split-K top-2 distributions on an identical,
# teacher-forced decode history at an exact tokenized prompt length.
[[ $# -eq 7 ]] || {
    echo "Usage: $0 HARNESS MODEL_GGUF PROMPT_FILE PROMPT_TOKENS NEW_TOKENS GPU_LAYERS RESULT_DIR" >&2
    exit 2
}

H=$1
M=$2
P=$3
PROMPT_TOKENS=$4
N=$5
G=$6
O=$7
CTX_SIZE=$((PROMPT_TOKENS + N))
BATCH_SIZE=$((PROMPT_TOKENS > 2048 ? PROMPT_TOKENS : 2048))

[[ -x "$H" && -f "$M" && -f "$P" ]] || {
    echo "Missing harness, model, or prompt file." >&2
    exit 2
}
mkdir -p "$O"

env -u GGML_CUDA_LLAMA32_FA_DECODE_ENABLED -u GGML_CUDA_LLAMA32_FD_SPLITK_ENABLED -u GGML_CUDA_LLAMA32_FA_DECODE_TRACE \
    "$H" --model "$M" --prompt-file "$P" \
    --context-tokens "$PROMPT_TOKENS" --ctx-size "$CTX_SIZE" --predict "$N" \
    --gpu-layers "$G" --batch-size "$BATCH_SIZE" --ubatch-size 512 --flash-attn on \
    --trace-logits > "$O/builtin.txt" 2> "$O/builtin.stderr"

BUILTIN_PROMPT_TOKENS=$(sed -n 's/^PROMPT_TOKEN_COUNT=//p' "$O/builtin.txt")
[[ "$BUILTIN_PROMPT_TOKENS" == "$PROMPT_TOKENS" ]] || {
    echo "Built-in prompt length does not match requested token length." >&2
    exit 1
}

TEACHER_IDS=$(sed -n 's/^GENERATED_TOKEN_IDS=//p' "$O/builtin.txt")
[[ -n "$TEACHER_IDS" ]] || {
    echo "Built-in run produced no token IDs." >&2
    exit 1
}

env GGML_CUDA_LLAMA32_FA_DECODE_ENABLED=1 GGML_CUDA_LLAMA32_FD_SPLITK_ENABLED=1 GGML_CUDA_LLAMA32_FA_DECODE_TRACE=1 \
    "$H" --model "$M" --prompt-file "$P" \
    --context-tokens "$PROMPT_TOKENS" --ctx-size "$CTX_SIZE" --predict "$N" \
    --gpu-layers "$G" --batch-size "$BATCH_SIZE" --ubatch-size 512 --flash-attn on \
    --trace-logits --teacher-token-ids "$TEACHER_IDS" \
    > "$O/splitk-teacher-forced.txt" 2> "$O/splitk-teacher-forced.stderr"

CUSTOM_PROMPT_TOKENS=$(sed -n 's/^PROMPT_TOKEN_COUNT=//p' "$O/splitk-teacher-forced.txt")
[[ "$CUSTOM_PROMPT_TOKENS" == "$PROMPT_TOKENS" ]] || {
    echo "Custom prompt length does not match requested token length." >&2
    exit 1
}

RC=$(grep -c '^llama32-fd-splitk route: selected$' "$O/splitk-teacher-forced.stderr" || true)
[[ "$RC" -gt 0 ]] || {
    echo "Split-K route was not selected." >&2
    exit 1
}

python3 - "$O/builtin.txt" "$O/splitk-teacher-forced.txt" "$O/summary.txt" "$RC" <<'PY'
import re
import sys
from pathlib import Path

builtin_path, splitk_path, summary_path = map(Path, sys.argv[1:4])
route_count = int(sys.argv[4])
rx = re.compile(
    r"^LOGIT_STEP=(\d+),TOP1_ID=(\d+),TOP1_LOGIT=([^,]+),"
    r"TOP2_ID=(\d+),TOP2_LOGIT=([^,]+),MARGIN=([^,]+),FINITE=(\d+)$"
)

def load(path):
    out = {}
    for line in path.read_text().splitlines():
        match = rx.match(line)
        if match:
            step, top1, _, top2, _, margin, finite = match.groups()
            if finite != "1":
                raise SystemExit(f"Non-finite logits at step {step}: {path}")
            out[int(step)] = (int(top1), int(top2), float(margin))
    return out

builtin = load(builtin_path)
splitk = load(splitk_path)
if not builtin or sorted(builtin) != sorted(splitk):
    raise SystemExit("Built-in and Split-K logit steps do not match.")

overlap = []
top1_matches = 0
margin_difference = []
for step in sorted(builtin):
    b = builtin[step]
    c = splitk[step]
    overlap.append(len({b[0], b[1]} & {c[0], c[1]}) / 2.0)
    top1_matches += b[0] == c[0]
    margin_difference.append(abs(b[2] - c[2]))

summary = (
    f"logit_steps={len(overlap)} route_count={route_count} top_k=2 "
    f"top2_overlap_mean={sum(overlap)/len(overlap):.6f} "
    f"top2_overlap_min={min(overlap):.6f} "
    f"top1_match_rate={top1_matches/len(overlap):.6f} "
    f"margin_abs_diff_mean={sum(margin_difference)/len(margin_difference):.6f} "
    f"margin_abs_diff_max={max(margin_difference):.6f}\n"
)
summary_path.write_text(summary)
print(summary, end="")
PY

echo "PASS: teacher-forced distributional comparison completed."
echo "Results: $O"
