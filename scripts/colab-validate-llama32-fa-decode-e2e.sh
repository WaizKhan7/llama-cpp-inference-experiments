#!/usr/bin/env bash
set -euo pipefail

# Deterministic generated-text comparison for the built-in and experimental
# Llama 3.2 FA-decode routes. This is a text-equality smoke gate, not yet an
# exact generated-token-ID harness.

if [[ $# -lt 5 || $# -gt 7 ]]; then
    echo "Usage: $0 LLAMA_CLI MODEL_GGUF PROMPT_FILE CONTEXT_SIZE NEW_TOKENS [RESULT_DIR] [GPU_LAYERS]" >&2
    exit 2
fi

CLI=$1
MODEL_GGUF=$2
PROMPT_FILE=$3
CONTEXT_SIZE=$4
NEW_TOKENS=$5
RESULT_DIR=${6:-llama32-fa-e2e-results}
GPU_LAYERS=${7:-99}

if [[ ! -x "$CLI" ]]; then
    echo "Missing llama-cli executable: $CLI" >&2
    exit 2
fi
if [[ ! -f "$MODEL_GGUF" ]]; then
    echo "Missing model GGUF: $MODEL_GGUF" >&2
    exit 2
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "Missing prompt file: $PROMPT_FILE" >&2
    exit 2
fi
if ! [[ "$CONTEXT_SIZE" =~ ^[1-9][0-9]*$ && "$NEW_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    echo "CONTEXT_SIZE and NEW_TOKENS must be positive integers." >&2
    exit 2
fi

mkdir -p "$RESULT_DIR"

{
    echo "Llama 3.2 FA-decode end-to-end text comparison"
    echo "UTC date: $(date -u --iso-8601=seconds)"
    echo "CLI: $CLI"
    echo "Model: $MODEL_GGUF"
    echo "Prompt: $PROMPT_FILE"
    echo "Context size: $CONTEXT_SIZE"
    echo "Requested new tokens: $NEW_TOKENS"
    echo "GPU layers: $GPU_LAYERS"
    echo
    sha256sum "$MODEL_GGUF"
    echo
    nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
} > "$RESULT_DIR/environment.txt"

COMMON_ARGS=(
    --model "$MODEL_GGUF"
    --file "$PROMPT_FILE"
    --ctx-size "$CONTEXT_SIZE"
    --predict "$NEW_TOKENS"
    --temp 0
    --seed 1234
    --no-conversation
    --no-display-prompt
    --simple-io
    --no-perf
    --no-warmup
    --flash-attn on
    --gpu-layers "$GPU_LAYERS"
    --no-context-shift
)

set +e
env -u GGML_CUDA_LLAMA32_FA_DECODE_ENABLED \
    "$CLI" "${COMMON_ARGS[@]}" \
    > "$RESULT_DIR/builtin.generated.txt" \
    2> "$RESULT_DIR/builtin.stderr.log"
BUILTIN_STATUS=$?

env GGML_CUDA_LLAMA32_FA_DECODE_ENABLED=1 \
    GGML_CUDA_LLAMA32_FA_DECODE_TRACE=1 \
    "$CLI" "${COMMON_ARGS[@]}" \
    > "$RESULT_DIR/custom.generated.txt" \
    2> "$RESULT_DIR/custom.stderr.log"
CUSTOM_STATUS=$?
set -e

if [[ $BUILTIN_STATUS -ne 0 || $CUSTOM_STATUS -ne 0 ]]; then
    echo "Generation process failed: built-in=$BUILTIN_STATUS, custom=$CUSTOM_STATUS" >&2
    exit 1
fi

sha256sum \
    "$RESULT_DIR/builtin.generated.txt" \
    "$RESULT_DIR/custom.generated.txt" \
    > "$RESULT_DIR/generated-text.sha256"

ROUTE_COUNT="$(grep -c '^llama32-fa-decode route: selected$' "$RESULT_DIR/custom.stderr.log" || true)"
printf '%s\n' "$ROUTE_COUNT" > "$RESULT_DIR/custom-route-count.txt"
if [[ "$ROUTE_COUNT" -eq 0 ]]; then
    echo "FAIL: custom process selected the FA-decode route zero times." >&2
    exit 1
fi

if ! cmp -s "$RESULT_DIR/builtin.generated.txt" "$RESULT_DIR/custom.generated.txt"; then
    diff -u \
        "$RESULT_DIR/builtin.generated.txt" \
        "$RESULT_DIR/custom.generated.txt" \
        > "$RESULT_DIR/generated-text.diff" || true
    echo "FAIL: generated text differs. See $RESULT_DIR/generated-text.diff" >&2
    exit 1
fi

{
    echo "PASS: built-in and custom generated text is byte-identical."
    echo "PASS: custom FA-decode route selected $ROUTE_COUNT time(s)."
    echo "Scope: deterministic greedy text comparison only."
    echo "Not yet validated here: generated token IDs, exact prompt token count, or performance."
    echo "Results: $RESULT_DIR"
} | tee "$RESULT_DIR/summary.txt"

