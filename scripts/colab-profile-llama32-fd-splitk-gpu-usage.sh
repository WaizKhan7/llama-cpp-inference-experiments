#!/usr/bin/env bash
# Coarse GPU residency/utilization observation for baseline and Split-K runs.
# Sampling covers model load, prompt prefill, warmup, and timed decode. Do not
# use this script's timings as Phase 6 performance measurements.
set -euo pipefail

[[ $# -ge 6 ]] || {
    echo "Usage: $0 HARNESS MODEL_GGUF PROMPT_FILE NEW_TOKENS RESULT_DIR GPU_LAYERS [CONTEXT ...]" >&2
    exit 2
}

H=$1; M=$2; P=$3; N=$4; O=$5; G=$6; shift 6
[[ $# -gt 0 ]] || set -- 128 512 2048 4096 8192
[[ -x "$H" && -f "$M" && -f "$P" ]] || {
    echo "Missing harness, model, or prompt." >&2
    exit 2
}
command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi is required." >&2
    exit 2
}

mkdir -p "$O"
echo "context,implementation,peak_memory_mib,mean_gpu_util_pct,peak_gpu_util_pct,samples" > "$O/summary.csv"

summarize() {
    local input=$1
    python3 - "$input" <<'PY'
import sys
from pathlib import Path

memory = []
utilization = []
for line in Path(sys.argv[1]).read_text().splitlines():
    fields = [item.strip() for item in line.split(",")]
    if len(fields) != 2:
        continue
    try:
        memory.append(float(fields[0]))
        utilization.append(float(fields[1]))
    except ValueError:
        pass

if not memory:
    raise SystemExit("No parseable nvidia-smi samples.")
print(f"{max(memory):.0f},{sum(utilization)/len(utilization):.2f},{max(utilization):.0f},{len(memory)}")
PY
}

for C; do
    D="$O/context-$C"
    mkdir -p "$D"
    CTX=$((C + N))
    BATCH=$((C > 2048 ? C : 2048))

    nvidia-smi --query-gpu=memory.used,utilization.gpu         --format=csv,noheader,nounits --loop-ms=100 > "$D/builtin.gpu.csv" &
    MONITOR=$!
    set +e
    env -u GGML_CUDA_LLAMA32_FA_DECODE_ENABLED         -u GGML_CUDA_LLAMA32_FD_SPLITK_ENABLED         -u GGML_CUDA_LLAMA32_FA_DECODE_TRACE         "$H" --model "$M" --prompt-file "$P" --context-tokens "$C"         --ctx-size "$CTX" --predict "$N" --gpu-layers "$G"         --batch-size "$BATCH" --ubatch-size 512 --warmup 2 --runs 5         --flash-attn on > "$D/builtin.txt" 2> "$D/builtin.stderr"
    STATUS=$?
    kill "$MONITOR" 2>/dev/null || true
    wait "$MONITOR" 2>/dev/null || true
    set -e
    [[ "$STATUS" -eq 0 ]] || { echo "Baseline run failed at context $C." >&2; exit "$STATUS"; }

    nvidia-smi --query-gpu=memory.used,utilization.gpu         --format=csv,noheader,nounits --loop-ms=100 > "$D/splitk.gpu.csv" &
    MONITOR=$!
    set +e
    env GGML_CUDA_LLAMA32_FA_DECODE_ENABLED=1         GGML_CUDA_LLAMA32_FD_SPLITK_ENABLED=1         "$H" --model "$M" --prompt-file "$P" --context-tokens "$C"         --ctx-size "$CTX" --predict "$N" --gpu-layers "$G"         --batch-size "$BATCH" --ubatch-size 512 --warmup 2 --runs 5         --flash-attn on > "$D/splitk.txt" 2> "$D/splitk.stderr"
    STATUS=$?
    kill "$MONITOR" 2>/dev/null || true
    wait "$MONITOR" 2>/dev/null || true
    set -e
    [[ "$STATUS" -eq 0 ]] || { echo "Split-K run failed at context $C." >&2; exit "$STATUS"; }

    BUILTIN=$(summarize "$D/builtin.gpu.csv")
    SPLITK=$(summarize "$D/splitk.gpu.csv")
    echo "$C,builtin,$BUILTIN" >> "$O/summary.csv"
    echo "$C,splitk,$SPLITK" >> "$O/summary.csv"
    echo "context=$C builtin_gpu=[$BUILTIN] splitk_gpu=[$SPLITK]"
done

echo "PASS: GPU usage observations complete. Results: $O"
