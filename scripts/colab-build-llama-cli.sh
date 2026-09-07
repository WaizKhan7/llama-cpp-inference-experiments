#!/usr/bin/env bash
set -euo pipefail

# Build llama-cli from an already-configured CUDA build directory.
# This intentionally preserves the cache's CUDA and experimental-adapter options.

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 REPOSITORY_DIR BUILD_DIR" >&2
    exit 2
fi

REPO_DIR=$1
BUILD_DIR=$2

if [[ ! -f "${REPO_DIR}/CMakeLists.txt" ]]; then
    echo "Not a llama.cpp repository: ${REPO_DIR}" >&2
    exit 2
fi

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    echo "Missing configured build directory: ${BUILD_DIR}" >&2
    exit 2
fi

if ! grep -q '^GGML_CUDA:BOOL=ON$' "${BUILD_DIR}/CMakeCache.txt"; then
    echo "This build directory is not configured with GGML_CUDA=ON." >&2
    exit 2
fi

if ! grep -q '^GGML_CUDA_LLAMA32_FA_DECODE:BOOL=ON$' "${BUILD_DIR}/CMakeCache.txt"; then
    echo "This build directory does not compile the experimental adapter." >&2
    exit 2
fi

cmake -S "${REPO_DIR}" -B "${BUILD_DIR}" -DLLAMA_BUILD_TOOLS=ON
cmake --build "${BUILD_DIR}" --target llama-cli --config Release -j2

CLI="${BUILD_DIR}/bin/llama-cli"
if [[ ! -x "${CLI}" ]]; then
    echo "Missing expected executable: ${CLI}" >&2
    exit 1
fi

echo "Built: ${CLI}"

