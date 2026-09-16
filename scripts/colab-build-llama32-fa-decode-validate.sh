#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 REPOSITORY_DIR BUILD_DIR" >&2; exit 2; }
REPO=$1; BUILD_DIR=$2
[[ -f "$BUILD_DIR/CMakeCache.txt" ]] || { echo "Missing configured build directory." >&2; exit 2; }
cmake -S "$REPO" -B "$BUILD_DIR" -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_LLAMA32_FA_DECODE_VALIDATE=ON
cmake --build "$BUILD_DIR" --target ggml-cuda llama-cli llama32-fa-decode-validate --config Release -j2
[[ -x "$BUILD_DIR/bin/llama32-fa-decode-validate" ]] || { echo "Harness build failed." >&2; exit 1; }
echo "Built: $BUILD_DIR/bin/llama32-fa-decode-validate"
