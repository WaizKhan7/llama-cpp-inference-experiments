#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

// Raw CUDA parity harness for the supplied Llama 3.2-1B Triton FA-decode
// kernel. This executable does not use GGML tensors or alter llama.cpp
// dispatch. The integrated kernel will have a different RoPE boundary.

namespace {

constexpr int kQueryHeads = 32;
constexpr int kKvHeads = 8;
constexpr int kGroupSize = kQueryHeads / kKvHeads;
constexpr int kHeadDim = 64;
constexpr int kHalfDim = kHeadDim / 2;
constexpr int kBlockSeq = 128;
constexpr float kScale = 1.0f / 8.0f;
constexpr unsigned kFullWarpMask = 0xffffffffu;

void cuda_check(cudaError_t status, const char * expression, const char * file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(
            stderr,
            "CUDA error at %s:%d for %s: %s\n",
            file,
            line,
            expression,
            cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) cuda_check((call), #call, __FILE__, __LINE__)

__device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return __shfl_sync(kFullWarpMask, value, 0);
}

// One block owns one (batch, query-head) pair. The block is exactly one warp.
// Each lane owns the RoPE pair (lane, lane + 32) and the matching two output
// dimensions.
__global__ void llama32_fa_decode_raw_kernel(
        const half * __restrict__ q,
        const half * __restrict__ k,
        const half * __restrict__ v,
        const float * __restrict__ cos_table,
        const float * __restrict__ sin_table,
        half * __restrict__ output,
        int seq_len,
        int q_pos) {
    const int lane = threadIdx.x;
    const int query_head = blockIdx.x;
    const int batch = blockIdx.y;

    const int kv_head = query_head / kGroupSize;
    const int d0 = lane;
    const int d1 = lane + kHalfDim;

    const int q_base = (batch * kQueryHeads + query_head) * kHeadDim;
    const int kv_base = (batch * kKvHeads + kv_head) * seq_len * kHeadDim;
    const int out_base = q_base;

    const float q_x0 = __half2float(q[q_base + d0]);
    const float q_x1 = __half2float(q[q_base + d1]);
    const float q_cos = cos_table[q_pos * kHalfDim + lane];
    const float q_sin = sin_table[q_pos * kHalfDim + lane];

    const float q_rot0 = q_x0 * q_cos - q_x1 * q_sin;
    const float q_rot1 = q_x0 * q_sin + q_x1 * q_cos;

    float running_max = -CUDART_INF_F;
    float running_sum = 0.0f;
    float accumulator0 = 0.0f;
    float accumulator1 = 0.0f;

    // Keep the same 128-token tiling boundary as the Triton kernel. This
    // readable first version processes positions serially inside each tile.
    for (int tile_start = 0; tile_start < seq_len; tile_start += kBlockSeq) {
        const int tile_end = min(tile_start + kBlockSeq, seq_len);

        for (int key_pos = tile_start; key_pos < tile_end; ++key_pos) {
            const int kv_offset = kv_base + key_pos * kHeadDim;
            const float k_x0 = __half2float(k[kv_offset + d0]);
            const float k_x1 = __half2float(k[kv_offset + d1]);
            const float k_cos = cos_table[key_pos * kHalfDim + lane];
            const float k_sin = sin_table[key_pos * kHalfDim + lane];

            const float k_rot0 = k_x0 * k_cos - k_x1 * k_sin;
            const float k_rot1 = k_x0 * k_sin + k_x1 * k_cos;

            float score = warp_sum(q_rot0 * k_rot0 + q_rot1 * k_rot1);
            score *= kScale;

            // Online softmax keeps the state stable without materializing the
            // complete score or probability vector.
            const float new_max = fmaxf(running_max, score);
            const float old_scale = expf(running_max - new_max);
            const float weight = expf(score - new_max);

            const float value0 = __half2float(v[kv_offset + d0]);
            const float value1 = __half2float(v[kv_offset + d1]);

            accumulator0 = accumulator0 * old_scale + weight * value0;
            accumulator1 = accumulator1 * old_scale + weight * value1;
            running_sum = running_sum * old_scale + weight;
            running_max = new_max;
        }
    }

    output[out_base + d0] = __float2half(accumulator0 / running_sum);
    output[out_base + d1] = __float2half(accumulator1 / running_sum);
}

void launch_fa_decode_raw(
        const half * q,
        const half * k,
        const half * v,
        const float * cos_table,
        const float * sin_table,
        half * output,
        int batch,
        int seq_len,
        int q_pos,
        cudaStream_t stream = nullptr) {
    const dim3 grid(kQueryHeads, batch, 1);
    const dim3 block(32, 1, 1);
    llama32_fa_decode_raw_kernel<<<grid, block, 0, stream>>>(
        q, k, v, cos_table, sin_table, output, seq_len, q_pos);
    CUDA_CHECK(cudaGetLastError());
}

enum class InputPattern {
    random,
    zeros,
    constant,
    large_signed,
    dominant_score,
    distinct_kv_heads,
};

const char * pattern_name(InputPattern pattern) {
    switch (pattern) {
        case InputPattern::random:            return "random";
        case InputPattern::zeros:             return "zeros";
        case InputPattern::constant:          return "constant";
        case InputPattern::large_signed:      return "large_signed";
        case InputPattern::dominant_score:    return "dominant_score";
        case InputPattern::distinct_kv_heads: return "distinct_kv_heads";
    }
    return "unknown";
}

struct HostInputs {
    int batch = 1;
    int seq_len = 0;
    int q_pos = 0;
    std::vector<half> q;
    std::vector<half> k;
    std::vector<half> v;
    std::vector<float> cos_table;
    std::vector<float> sin_table;
};


HostInputs make_inputs(
        int seq_len,
        int q_pos,
        uint32_t seed,
        InputPattern pattern) {
    HostInputs inputs;
    inputs.seq_len = seq_len;
    inputs.q_pos = q_pos;
    inputs.q.resize(kQueryHeads * kHeadDim);
    inputs.k.resize(kKvHeads * seq_len * kHeadDim);
    inputs.v.resize(kKvHeads * seq_len * kHeadDim);

    const int position_count = std::max(seq_len, q_pos + 1);
    inputs.cos_table.resize(position_count * kHalfDim);
    inputs.sin_table.resize(position_count * kHalfDim);

    for (int pos = 0; pos < position_count; ++pos) {
        for (int pair = 0; pair < kHalfDim; ++pair) {
            const float inverse_frequency =
                1.0f / std::pow(10000.0f, 2.0f * pair / kHeadDim);
            const float angle = pos * inverse_frequency;
            inputs.cos_table[pos * kHalfDim + pair] = std::cos(angle);
            inputs.sin_table[pos * kHalfDim + pair] = std::sin(angle);
        }
    }

    std::mt19937 generator(seed);
    std::normal_distribution<float> normal(0.0f, 0.5f);

    auto q_value = [&](int head, int dim) {
        switch (pattern) {
            case InputPattern::zeros:
                return 0.0f;
            case InputPattern::constant:
                return 0.25f;
            case InputPattern::large_signed:
                return ((head + dim) % 2 == 0) ? 4.0f : -4.0f;
            case InputPattern::dominant_score:
                return 1.0f;
            case InputPattern::distinct_kv_heads:
                return 0.05f * float((head % kGroupSize) + 1) +
                       0.002f * float((dim % 7) - 3);
            case InputPattern::random:
                return normal(generator);
        }
        return 0.0f;
    };

    for (int head = 0; head < kQueryHeads; ++head) {
        for (int dim = 0; dim < kHeadDim; ++dim) {
            inputs.q[head * kHeadDim + dim] =
                __float2half(q_value(head, dim));
        }
    }

    for (int kv_head = 0; kv_head < kKvHeads; ++kv_head) {
        for (int pos = 0; pos < seq_len; ++pos) {
            for (int dim = 0; dim < kHeadDim; ++dim) {
                float key_value = 0.0f;
                float value_value = 0.0f;

                switch (pattern) {
                    case InputPattern::zeros:
                        break;
                    case InputPattern::constant:
                        key_value = 0.25f;
                        value_value = 0.125f;
                        break;
                    case InputPattern::large_signed:
                        key_value = ((kv_head + pos + dim) % 2 == 0) ? 4.0f : -4.0f;
                        value_value = ((kv_head + dim) % 2 == 0) ? 2.0f : -2.0f;
                        break;
                    case InputPattern::dominant_score:
                        key_value = (pos == seq_len - 1) ? 1.0f : -1.0f;
                        value_value =
                            (pos == seq_len - 1) ? 0.5f : -0.25f;
                        value_value += 0.005f * float((dim % 9) - 4);
                        break;
                    case InputPattern::distinct_kv_heads:
                        key_value = 0.03f * float(kv_head + 1) +
                                    0.0005f * float((pos + dim) % 11);
                        value_value = 0.2f * float(kv_head + 1) +
                                      0.001f * float(dim);
                        break;
                    case InputPattern::random:
                        key_value = normal(generator);
                        value_value = normal(generator);
                        break;
                }

                const int offset =
                    (kv_head * seq_len + pos) * kHeadDim + dim;
                inputs.k[offset] = __float2half(key_value);
                inputs.v[offset] = __float2half(value_value);
            }
        }
    }

    return inputs;
}

// Independent reference: materialize all FP32 scores, then use a conventional
// max/subtract/exp softmax. It intentionally does not reproduce the CUDA
// kernel's online recurrence.
std::vector<float> reference_attention(const HostInputs & inputs) {
    std::vector<float> output(kQueryHeads * kHeadDim, 0.0f);
    std::vector<float> scores(inputs.seq_len);

    for (int query_head = 0; query_head < kQueryHeads; ++query_head) {
        const int kv_head = query_head / kGroupSize;
        const int q_base = query_head * kHeadDim;
        const int kv_base = kv_head * inputs.seq_len * kHeadDim;

        float q_rotated[kHeadDim];
        for (int pair = 0; pair < kHalfDim; ++pair) {
            const float q0 = __half2float(inputs.q[q_base + pair]);
            const float q1 = __half2float(inputs.q[q_base + pair + kHalfDim]);
            const float c =
                inputs.cos_table[inputs.q_pos * kHalfDim + pair];
            const float s =
                inputs.sin_table[inputs.q_pos * kHalfDim + pair];
            q_rotated[pair] = q0 * c - q1 * s;
            q_rotated[pair + kHalfDim] = q0 * s + q1 * c;
        }

        float maximum = -std::numeric_limits<float>::infinity();
        for (int key_pos = 0; key_pos < inputs.seq_len; ++key_pos) {
            const int key_base = kv_base + key_pos * kHeadDim;
            float dot = 0.0f;

            for (int pair = 0; pair < kHalfDim; ++pair) {
                const float k0 = __half2float(inputs.k[key_base + pair]);
                const float k1 =
                    __half2float(inputs.k[key_base + pair + kHalfDim]);
                const float c =
                    inputs.cos_table[key_pos * kHalfDim + pair];
                const float s =
                    inputs.sin_table[key_pos * kHalfDim + pair];
                const float k_rot0 = k0 * c - k1 * s;
                const float k_rot1 = k0 * s + k1 * c;
                dot += q_rotated[pair] * k_rot0;
                dot += q_rotated[pair + kHalfDim] * k_rot1;
            }

            scores[key_pos] = dot * kScale;
            maximum = std::max(maximum, scores[key_pos]);
        }

        float denominator = 0.0f;
        for (int key_pos = 0; key_pos < inputs.seq_len; ++key_pos) {
            scores[key_pos] = std::exp(scores[key_pos] - maximum);
            denominator += scores[key_pos];
        }

        for (int dim = 0; dim < kHeadDim; ++dim) {
            float accumulator = 0.0f;
            for (int key_pos = 0; key_pos < inputs.seq_len; ++key_pos) {
                const int value_offset =
                    kv_base + key_pos * kHeadDim + dim;
                accumulator +=
                    scores[key_pos] * __half2float(inputs.v[value_offset]);
            }
            output[q_base + dim] = accumulator / denominator;
        }
    }

    return output;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&pointer_), count * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer & operator=(const DeviceBuffer &) = delete;

    T * get() {
        return pointer_;
    }

    const T * get() const {
        return pointer_;
    }

    size_t bytes() const {
        return count_ * sizeof(T);
    }

private:
    T * pointer_ = nullptr;
    size_t count_ = 0;
};

struct CaseResult {
    bool passed = true;
    float maximum_error = 0.0f;
    double mean_error = 0.0;
    int nan_count = 0;
    int infinity_count = 0;
};

CaseResult run_case(
        int seq_len,
        int q_pos,
        uint32_t seed,
        InputPattern pattern,
        bool verbose_heads) {
    const HostInputs inputs = make_inputs(seq_len, q_pos, seed, pattern);
    const std::vector<float> expected = reference_attention(inputs);

    DeviceBuffer<half> d_q(inputs.q.size());
    DeviceBuffer<half> d_k(inputs.k.size());
    DeviceBuffer<half> d_v(inputs.v.size());
    DeviceBuffer<float> d_cos(inputs.cos_table.size());
    DeviceBuffer<float> d_sin(inputs.sin_table.size());
    DeviceBuffer<half> d_output(expected.size());

    CUDA_CHECK(cudaMemcpy(
        d_q.get(), inputs.q.data(), d_q.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_k.get(), inputs.k.data(), d_k.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_v.get(), inputs.v.data(), d_v.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_cos.get(),
        inputs.cos_table.data(),
        d_cos.bytes(),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_sin.get(),
        inputs.sin_table.data(),
        d_sin.bytes(),
        cudaMemcpyHostToDevice));

    launch_fa_decode_raw(
        d_q.get(),
        d_k.get(),
        d_v.get(),
        d_cos.get(),
        d_sin.get(),
        d_output.get(),
        inputs.batch,
        seq_len,
        q_pos);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> actual_half(expected.size());
    CUDA_CHECK(cudaMemcpy(
        actual_half.data(),
        d_output.get(),
        d_output.bytes(),
        cudaMemcpyDeviceToHost));

    constexpr float absolute_tolerance = 2.0e-2f;
    constexpr float relative_tolerance = 2.0e-2f;

    CaseResult result;
    double error_sum = 0.0;

    for (int head = 0; head < kQueryHeads; ++head) {
        bool head_passed = true;
        float head_maximum_error = 0.0f;
        double head_error_sum = 0.0;

        for (int dim = 0; dim < kHeadDim; ++dim) {
            const int offset = head * kHeadDim + dim;
            const float actual = __half2float(actual_half[offset]);
            const float reference = expected[offset];

            if (std::isnan(actual)) {
                ++result.nan_count;
                head_passed = false;
                continue;
            }
            if (std::isinf(actual)) {
                ++result.infinity_count;
                head_passed = false;
                continue;
            }

            const float error = std::abs(actual - reference);
            const float allowed =
                absolute_tolerance + relative_tolerance * std::abs(reference);
            head_maximum_error = std::max(head_maximum_error, error);
            head_error_sum += error;
            if (error > allowed) {
                head_passed = false;
            }
        }

        result.maximum_error =
            std::max(result.maximum_error, head_maximum_error);
        error_sum += head_error_sum;
        result.passed = result.passed && head_passed;

        if (verbose_heads || !head_passed) {
            std::printf(
                "    head %2d: %s max_abs=%.7f mean_abs=%.7f\n",
                head,
                head_passed ? "PASS" : "FAIL",
                head_maximum_error,
                head_error_sum / kHeadDim);
        }
    }

    result.mean_error = error_sum / expected.size();
    result.passed =
        result.passed && result.nan_count == 0 && result.infinity_count == 0;

    std::printf(
        "  S=%4d q_pos=%4d seed=%u pattern=%-17s %s "
        "max_abs=%.7f mean_abs=%.7f nan=%d inf=%d\n",
        seq_len,
        q_pos,
        seed,
        pattern_name(pattern),
        result.passed ? "PASS" : "FAIL",
        result.maximum_error,
        result.mean_error,
        result.nan_count,
        result.infinity_count);

    return result;
}

bool run_correctness_suite(bool quick, bool verbose_heads) {
    const std::vector<int> full_lengths = {
        1, 31, 32, 33,
        127, 128, 129,
        255, 256, 257,
        511, 512, 513,
        2048, 4096, 6144, 8192,
    };
    const std::vector<int> quick_lengths = {1, 33, 129, 4096};
    const std::vector<uint32_t> full_seeds = {1, 7, 19, 101, 2027};
    const std::vector<uint32_t> quick_seeds = {1};

    const std::vector<int> & lengths =
        quick ? quick_lengths : full_lengths;
    const std::vector<uint32_t> & seeds =
        quick ? quick_seeds : full_seeds;

    bool passed = true;
    std::printf(
        "\nCorrectness suite: %s\n",
        quick ? "quick" : "full");

    // The supplied Triton launcher and model patch append the current K/V
    // first, so q_pos=S-1 is the primary parity convention.
    for (uint32_t seed : seeds) {
        for (int seq_len : lengths) {
            passed =
                run_case(
                    seq_len,
                    seq_len - 1,
                    seed,
                    InputPattern::random,
                    verbose_heads).passed &&
                passed;
        }
    }

    const int edge_length = quick ? 129 : 4096;
    const std::vector<InputPattern> edge_patterns = {
        InputPattern::zeros,
        InputPattern::constant,
        InputPattern::large_signed,
        InputPattern::dominant_score,
        InputPattern::distinct_kv_heads,
    };
    for (InputPattern pattern : edge_patterns) {
        passed =
            run_case(
                edge_length,
                edge_length - 1,
                11,
                pattern,
                verbose_heads).passed &&
            passed;
    }

    // Also validate the other legitimate API convention: a query immediately
    // after a cache containing S earlier tokens. The q_pos argument, rather
    // than hidden global state, defines this behavior.
    passed =
        run_case(
            129,
            129,
            23,
            InputPattern::random,
            verbose_heads).passed &&
        passed;

    std::printf(
        "\nCorrectness result: %s\n",
        passed ? "PASS" : "FAIL");
    return passed;
}

void run_benchmark(int warmup, int iterations) {
    const std::vector<int> lengths = {
        128, 512, 1024, 2048, 4096, 6144, 8192,
    };

    std::printf(
        "\nRaw FA-decode CUDA timing (%d warmups, %d iterations)\n",
        warmup,
        iterations);

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int seq_len : lengths) {
        const HostInputs inputs =
            make_inputs(seq_len, seq_len - 1, 101, InputPattern::random);

        DeviceBuffer<half> d_q(inputs.q.size());
        DeviceBuffer<half> d_k(inputs.k.size());
        DeviceBuffer<half> d_v(inputs.v.size());
        DeviceBuffer<float> d_cos(inputs.cos_table.size());
        DeviceBuffer<float> d_sin(inputs.sin_table.size());
        DeviceBuffer<half> d_output(kQueryHeads * kHeadDim);

        CUDA_CHECK(cudaMemcpy(
            d_q.get(), inputs.q.data(), d_q.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_k.get(), inputs.k.data(), d_k.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_v.get(), inputs.v.data(), d_v.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_cos.get(),
            inputs.cos_table.data(),
            d_cos.bytes(),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_sin.get(),
            inputs.sin_table.data(),
            d_sin.bytes(),
            cudaMemcpyHostToDevice));

        for (int iteration = 0; iteration < warmup; ++iteration) {
            launch_fa_decode_raw(
                d_q.get(),
                d_k.get(),
                d_v.get(),
                d_cos.get(),
                d_sin.get(),
                d_output.get(),
                inputs.batch,
                seq_len,
                inputs.q_pos);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            launch_fa_decode_raw(
                d_q.get(),
                d_k.get(),
                d_v.get(),
                d_cos.get(),
                d_sin.get(),
                d_output.get(),
                inputs.batch,
                seq_len,
                inputs.q_pos);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        std::printf(
            "  S=%4d: %.6f ms/call\n",
            seq_len,
            total_ms / iterations);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void print_usage(const char * executable) {
    std::printf(
        "Usage: %s [--quick] [--verbose-heads] [--no-benchmark] "
        "[--warmup N] [--iterations N]\n",
        executable);
}

int parse_positive_integer(const char * text, const char * option) {
    char * end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 ||
        value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "Invalid value for %s: %s\n", option, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

} // namespace

int main(int argc, char ** argv) {
    bool quick = false;
    bool verbose_heads = false;
    bool benchmark = true;
    int warmup = 20;
    int iterations = 100;

    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--quick") {
            quick = true;
        } else if (option == "--verbose-heads") {
            verbose_heads = true;
        } else if (option == "--no-benchmark") {
            benchmark = false;
        } else if (option == "--warmup" && index + 1 < argc) {
            warmup = parse_positive_integer(argv[++index], "--warmup");
        } else if (option == "--iterations" && index + 1 < argc) {
            iterations =
                parse_positive_integer(argv[++index], "--iterations");
        } else if (option == "--help") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[index]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    std::printf("Llama 3.2-1B raw CUDA FA-decode parity harness\n");
    std::printf(
        "GPU: %s (compute capability %d.%d)\n",
        properties.name,
        properties.major,
        properties.minor);
    std::printf(
        "Geometry: Hq=%d Hkv=%d GQA=%d D=%d BLOCK_SEQ=%d scale=%.3f\n",
        kQueryHeads,
        kKvHeads,
        kGroupSize,
        kHeadDim,
        kBlockSeq,
        kScale);

    const bool passed = run_correctness_suite(quick, verbose_heads);
    if (!passed) {
        std::fprintf(
            stderr,
            "Correctness failed. Timing is intentionally skipped.\n");
        return EXIT_FAILURE;
    }

    if (benchmark) {
        run_benchmark(warmup, iterations);
    }

    return EXIT_SUCCESS;
}
