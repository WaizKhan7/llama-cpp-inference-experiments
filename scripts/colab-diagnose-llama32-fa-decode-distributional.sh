#!/usr/bin/env bash
set -euo pipefail

# Compare built-in and custom top-2 distributions on an identical,
# teacher-forced decode history.
[[ $# -eq 7 ]] || {
    echo "Usage: $0 HARNESS MODEL_GGUF PROMPT_FILE CONTEXT_CAPACITY NEW_TOKENS GPU_LAYERS RESULT_DIR" >&2
    exit 2
}

H=$1
M=$2
P=$3
C=$4
N=$5
G=$6
O=$7

[[ -x "$H" && -f "$M" && -f "$P" ]] || {
    echo "Missing harness, model, or prompt file." >&2
    exit 2
}
mkdir -p "$O"

env -u GGML_CUDA_LLAMA32_FA_DECODE_ENABLED -u GGML_CUDA_LLAMA32_FA_DECODE_TRACE \
    "$H" --model "$M" --prompt-file "$P" --ctx-size "$C" --predict "$N" \
    --gpu-layers "$G" --batch-size 2048 --ubatch-size 512 --flash-attn on \
    --trace-logits > "$O/builtin.txt" 2> "$O/builtin.stderr"

TEACHER_IDS=$(sed -n 's/^GENERATED_TOKEN_IDS=//p' "$O/builtin.txt")
[[ -n "$TEACHER_IDS" ]] || {
    echo "Built-in run produced no token IDs." >&2
    exit 1
}

env GGML_CUDA_LLAMA32_FA_DECODE_ENABLED=1 GGML_CUDA_LLAMA32_FA_DECODE_TRACE=1 \
    "$H" --model "$M" --prompt-file "$P" --ctx-size "$C" --predict "$N" \
    --gpu-layers "$G" --batch-size 2048 --ubatch-size 512 --flash-attn on \
    --trace-logits --teacher-token-ids "$TEACHER_IDS" \
    > "$O/custom-teacher-forced.txt" 2> "$O/custom-teacher-forced.stderr"

RC=$(grep -c '^llama32-fa-decode route: selected$' "$O/custom-teacher-forced.stderr" || true)
[[ "$RC" -gt 0 ]] || {
    echo "Custom FA-decode route was not selected." >&2
    exit 1
}

python3 - "$O/builtin.txt" "$O/custom-teacher-forced.txt" "$O/summary.txt" "$RC" <<'PY'
import re
import sys
from pathlib import Path

builtin_path, custom_path, summary_path = map(Path, sys.argv[1:4])
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
custom = load(custom_path)
if not builtin or sorted(builtin) != sorted(custom):
    raise SystemExit("Built-in and custom logit steps do not match.")

overlap = []
top1_matches = 0
margin_difference = []
for step in sorted(builtin):
    b = builtin[step]
    c = custom[step]
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
