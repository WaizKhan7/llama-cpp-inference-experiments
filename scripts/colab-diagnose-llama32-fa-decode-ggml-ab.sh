#!/usr/bin/env bash
set -euo pipefail

# Incrementally build and run the GGML built-in/custom FA-decode A/B
# diagnostic in an existing CUDA build directory.

[[ $# -ge 2 && $# -le 3 ]] || {
    echo "Usage: $0 REPOSITORY_DIR BUILD_DIR [RESULT_DIR]" >&2
    exit 2
}

REPO=$1
BUILD_DIR=$2
RESULT_DIR=${3:-/content/llama32-fa-ggml-ab-results}
TARGET=test-cuda-llama32-fa-decode-ggml-ab
EXECUTABLE="$BUILD_DIR/bin/$TARGET"

[[ -f "$REPO/CMakeLists.txt" ]] || {
    echo "Missing llama.cpp repository: $REPO" >&2
    exit 2
}
[[ -f "$BUILD_DIR/CMakeCache.txt" ]] || {
    echo "Missing configured build directory: $BUILD_DIR" >&2
    exit 2
}

mkdir -p "$RESULT_DIR"

echo "Configuring the new diagnostic target..."
cmake -S "$REPO" -B "$BUILD_DIR" \
    -DGGML_CUDA_LLAMA32_DECODE_TESTS=ON \
    -DGGML_CUDA_LLAMA32_FA_DECODE=ON

echo "Building $TARGET..."
cmake --build "$BUILD_DIR" \
    --target "$TARGET" \
    --config Release \
    -j2

[[ -x "$EXECUTABLE" ]] || {
    echo "Missing diagnostic executable: $EXECUTABLE" >&2
    exit 1
}

echo "Running GGML built-in/custom attention comparison..."
"$EXECUTABLE" 2>&1 | tee "$RESULT_DIR/ggml-built-in-custom-ab.log"
status=${PIPESTATUS[0]}

{
    echo "Diagnostic commit: $(git -C "$REPO" rev-parse HEAD)"
    echo "Exit status: $status"
    echo "Results: $RESULT_DIR"
} | tee "$RESULT_DIR/summary.txt"

exit "$status"
