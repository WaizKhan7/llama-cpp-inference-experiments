#!/usr/bin/env bash
set -euo pipefail

# Google Colab validation for the raw Llama 3.2-1B CUDA FA-decode harness.
# Run after cloning the branch on a Colab NVIDIA T4 runtime. No GGUF model is
# needed because this gate validates only the standalone CUDA kernel.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
RESULT_DIR="${1:-/content/llama32-fa-results}"
BUILD_DIR="$(mktemp -d /tmp/llama32-fa-sm75.XXXXXX)"
HARNESS="${BUILD_DIR}/bin/test-cuda-llama32-fa-decode"
BOUNDARY_HARNESS="${BUILD_DIR}/bin/test-cuda-llama32-fa-decode-ggml-boundary"

mkdir -p "${RESULT_DIR}"

{
    echo "Llama 3.2 raw CUDA FA-decode verification"
    echo "UTC date: $(date -u --iso-8601=seconds)"
    echo "Repository: ${REPO_DIR}"
    echo "Commit: $(git -C "${REPO_DIR}" rev-parse HEAD)"
    echo "Branch: $(git -C "${REPO_DIR}" branch --show-current)"
    echo
    echo "Git status:"
    git -C "${REPO_DIR}" status --short
    echo
    echo "GPU:"
    nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total         --format=csv,noheader
    echo
    echo "CUDA:"
    nvcc --version
    echo
    echo "CMake:"
    cmake --version
    echo
    echo "Host compiler:"
    c++ --version
} | tee "${RESULT_DIR}/environment.txt"

cmake     -S "${REPO_DIR}"     -B "${BUILD_DIR}"     -DGGML_CUDA=ON     -DGGML_CUDA_FA=ON     -DGGML_CUDA_LLAMA32_DECODE_TESTS=ON     -DGGML_CUDA_LLAMA32_FA_DECODE=ON     -DCMAKE_CUDA_ARCHITECTURES=75     -DCMAKE_BUILD_TYPE=Release     -DLLAMA_BUILD_TESTS=OFF     -DLLAMA_BUILD_EXAMPLES=OFF     -DLLAMA_BUILD_TOOLS=OFF     2>&1 | tee "${RESULT_DIR}/configure.log"

cmake     --build "${BUILD_DIR}"     --target ggml-cuda test-cuda-llama32-fa-decode-ggml-boundary test-cuda-llama32-fa-decode     --config Release     -j2     2>&1 | tee "${RESULT_DIR}/build.log"

if [[ ! -x "${HARNESS}" || ! -x "${BOUNDARY_HARNESS}" ]]; then
    echo "Missing expected validation executable." >&2
    exit 1
fi

set +e
"${BOUNDARY_HARNESS}" 2>&1 | tee "${RESULT_DIR}/ggml-boundary-correctness.log"
boundary_status=${PIPESTATUS[0]}
set -e

if [[ ${boundary_status} -ne 0 ]]; then
    echo "GGML-boundary correctness failed; raw timing is skipped." >&2
    exit "${boundary_status}"
fi

set +e
"${HARNESS}" --quick --no-benchmark 2>&1     | tee "${RESULT_DIR}/quick-correctness.log"
quick_status=${PIPESTATUS[0]}
set -e

if [[ ${quick_status} -ne 0 ]]; then
    echo "Quick correctness failed; full validation is skipped." >&2
    exit "${quick_status}"
fi

# The harness starts timing only after its full correctness suite has passed.
set +e
"${HARNESS}" --warmup 20 --iterations 100 2>&1     | tee "${RESULT_DIR}/full-correctness-and-benchmark.log"
full_status=${PIPESTATUS[0]}
set -e

if [[ ${full_status} -ne 0 ]]; then
    echo "Full correctness failed; collecting per-head diagnostics."         | tee "${RESULT_DIR}/failure-summary.txt"
    set +e
    "${HARNESS}" --verbose-heads --no-benchmark 2>&1         | tee "${RESULT_DIR}/per-head-failure.log"
    verbose_status=${PIPESTATUS[0]}
    set -e
    exit "${verbose_status}"
fi

{
    echo "PASS: direct GGML-boundary correctness completed."
    echo "PASS: quick and full correctness completed."
    echo "PASS: timing ran only after correctness."
    echo "Validated commit: $(git -C "${REPO_DIR}" rev-parse HEAD)"
    echo "Results: ${RESULT_DIR}"
} | tee "${RESULT_DIR}/summary.txt"
