#include <../common.cuh>
#include <../fattn.cuh>
#include <../fattn-llama32-fa-decode.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr int HQ = 32;
constexpr int HKV = 8;
constexpr int D = 64;
constexpr int Q_STRIDE = D;
constexpr int K_STRIDE = D;
constexpr int V_STRIDE = D;
constexpr int O_STRIDE = D;
constexpr int OUTPUT_GUARD = 32;
constexpr float SCALE = 0.125f;
constexpr float SENTINEL = -777.0f;

// Calibrated from the production-flag T4 sweep in Entry 040. These limits
// are validation-only; production inference never computes the FP64 reference.
constexpr double CUSTOM_REFERENCE_MAX_ABS_LIMIT = 2.0e-4;
constexpr double CUSTOM_REFERENCE_MEAN_ABS_LIMIT = 5.0e-5;

void check_cuda(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

template<typename T>
struct device_buffer {
    T * data = nullptr;

    explicit device_buffer(size_t count) {
        check_cuda(cudaMalloc(reinterpret_cast<void **>(&data), count * sizeof(T)), "malloc");
    }

    ~device_buffer() {
        if (data) {
            cudaFree(data);
        }
    }

    device_buffer(const device_buffer &) = delete;
    device_buffer & operator=(const device_buffer &) = delete;
};

void set_tensor(
        ggml_tensor & tensor,
        enum ggml_type type,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        size_t nb0, size_t nb1, size_t nb2, size_t nb3) {
    std::memset(&tensor, 0, sizeof(tensor));
    tensor.type = type;
    tensor.ne[0] = ne0;
    tensor.ne[1] = ne1;
    tensor.ne[2] = ne2;
    tensor.ne[3] = ne3;
    tensor.nb[0] = nb0;
    tensor.nb[1] = nb1;
    tensor.nb[2] = nb2;
    tensor.nb[3] = nb3;
}

float random_signed(uint32_t & state) {
    state = state * 1664525u + 1013904223u;
    const float unit = static_cast<float>((state >> 8) & 0x00ffffffu) /
        static_cast<float>(0x00ffffffu);
    return 2.0f * unit - 1.0f;
}

enum class input_pattern {
    random,
    zeros,
    dominant_score,
    gqa_signature,
};

const char * pattern_name(input_pattern pattern) {
    switch (pattern) {
        case input_pattern::random:         return "random";
        case input_pattern::zeros:          return "zeros";
        case input_pattern::dominant_score: return "dominant";
        case input_pattern::gqa_signature:  return "gqa_signature";
    }
    return "unknown";
}

int padded_kv_length(int visible) {
    return ((visible + 255) / 256) * 256;
}

void fill_inputs(
        input_pattern pattern,
        uint32_t seed,
        int visible,
        int padded,
        std::vector<float> & q,
        std::vector<half> & k,
        std::vector<half> & v) {
    uint32_t state = seed;

    // Deliberately nonzero padding must be ignored by the -infinity mask.
    for (half & value : k) value = __float2half(8.0f * random_signed(state));
    for (half & value : v) value = __float2half(8.0f * random_signed(state));

    if (pattern == input_pattern::zeros) {
        std::fill(q.begin(), q.end(), 0.0f);
        for (int h = 0; h < HKV; ++h) for (int p = 0; p < visible; ++p)
            for (int d = 0; d < D; ++d) {
                k[(h * padded + p) * K_STRIDE + d] = __float2half(0.0f);
                v[(h * padded + p) * V_STRIDE + d] = __float2half(0.0f);
            }
        return;
    }

    if (pattern == input_pattern::dominant_score) {
        std::fill(q.begin(), q.end(), 1.0f);
        for (int h = 0; h < HKV; ++h) for (int p = 0; p < visible; ++p)
            for (int d = 0; d < D; ++d) {
                const bool dominant = p == visible - 1;
                k[(h * padded + p) * K_STRIDE + d] =
                    __float2half(dominant ? 2.0f : -0.10f);
                v[(h * padded + p) * V_STRIDE + d] =
                    __float2half(dominant
                        ? -0.70f + 0.18f * h + 0.003f * d
                        : 0.05f * random_signed(state));
            }
        return;
    }

    if (pattern == input_pattern::gqa_signature) {
        for (int qh = 0; qh < HQ; ++qh) for (int d = 0; d < D; ++d) {
            q[qh * Q_STRIDE + d] = 0.03f * static_cast<float>(
                ((qh + 3) * (d + 5)) % 29 - 14);
        }
        for (int h = 0; h < HKV; ++h) for (int p = 0; p < visible; ++p)
            for (int d = 0; d < D; ++d) {
                k[(h * padded + p) * K_STRIDE + d] =
                    __float2half(0.02f * static_cast<float>(
                        ((h + 2) * (p + 1) * (d + 3)) % 31 - 15));
                v[(h * padded + p) * V_STRIDE + d] =
                    __float2half(-0.70f + 0.18f * h + 0.002f * d + 0.001f * p);
            }
        return;
    }

    for (float & value : q) value = 1.25f * random_signed(state);
    for (int h = 0; h < HKV; ++h) for (int p = 0; p < visible; ++p)
        for (int d = 0; d < D; ++d) {
            k[(h * padded + p) * K_STRIDE + d] =
                __float2half(0.50f * random_signed(state));
            v[(h * padded + p) * V_STRIDE + d] =
                __float2half(0.75f * random_signed(state));
        }
}

struct comparison {
    bool finite = true;
    bool padding_untouched = true;
    double max_abs = 0.0;
    double sum_abs = 0.0;
    int max_head = -1;
    int max_dim = -1;
};

std::vector<double> cpu_reference(
        const std::vector<float> & q,
        const std::vector<half> & k,
        const std::vector<half> & v,
        const std::vector<half> & mask,
        int visible,
        int padded) {
    std::vector<double> reference(HQ * D);

    for (int query_head = 0; query_head < HQ; ++query_head) {
        const int kv_head = query_head / (HQ / HKV);
        std::vector<double> scores(visible);
        double maximum = -std::numeric_limits<double>::infinity();

        for (int pos = 0; pos < visible; ++pos) {
            double dot = 0.0;
            for (int dim = 0; dim < D; ++dim) {
                dot += static_cast<double>(q[query_head * Q_STRIDE + dim]) *
                    static_cast<double>(__half2float(
                        k[(kv_head * padded + pos) * K_STRIDE + dim]));
            }
            scores[pos] = dot * static_cast<double>(SCALE) +
                static_cast<double>(__half2float(mask[pos]));
            maximum = std::fmax(maximum, scores[pos]);
        }

        double denominator = 0.0;
        for (int pos = 0; pos < visible; ++pos) {
            denominator += std::exp(scores[pos] - maximum);
        }

        for (int dim = 0; dim < D; ++dim) {
            double numerator = 0.0;
            for (int pos = 0; pos < visible; ++pos) {
                const double weight = std::exp(scores[pos] - maximum);
                numerator += weight * static_cast<double>(__half2float(
                    v[(kv_head * padded + pos) * V_STRIDE + dim]));
            }
            reference[query_head * D + dim] = numerator / denominator;
        }
    }

    return reference;
}

comparison compare_reference(
        const std::vector<float> & actual,
        const std::vector<double> & reference) {
    comparison result;

    for (int head = 0; head < HQ; ++head) {
        for (int dim = 0; dim < D; ++dim) {
            const double value = static_cast<double>(actual[head * O_STRIDE + dim]);
            const double error = std::fabs(value - reference[head * D + dim]);
            result.finite = result.finite && std::isfinite(value);

            result.sum_abs += error;
            if (error > result.max_abs) {
                result.max_abs = error;
                result.max_head = head;
                result.max_dim = dim;
            }
        }
    }

    return result;
}

comparison compare_outputs(
        const std::vector<float> & builtin,
        const std::vector<float> & custom) {
    comparison result;

    for (int head = 0; head < HQ; ++head) {
        for (int dim = 0; dim < D; ++dim) {
            const float a = builtin[head * O_STRIDE + dim];
            const float b = custom[head * O_STRIDE + dim];
            const double error = std::fabs(
                static_cast<double>(a) - static_cast<double>(b));

            result.finite = result.finite && std::isfinite(a) && std::isfinite(b);
            result.sum_abs += error;
            if (error > result.max_abs) {
                result.max_abs = error;
                result.max_head = head;
                result.max_dim = dim;
            }
        }
    }

    for (size_t index = HQ * O_STRIDE; index < builtin.size(); ++index) {
        result.padding_untouched =
            result.padding_untouched &&
            builtin[index] == SENTINEL &&
            custom[index] == SENTINEL;
    }

    return result;
}

bool run_case(
        ggml_backend_cuda_context & context,
        int visible,
        input_pattern pattern,
        uint32_t seed) {
    const int padded = padded_kv_length(visible);
    std::vector<float> q(HQ * Q_STRIDE);
    std::vector<half> k(HKV * padded * K_STRIDE);
    std::vector<half> v(HKV * padded * V_STRIDE);
    std::vector<half> mask(16 * padded);
    std::vector<float> builtin(HQ * O_STRIDE + OUTPUT_GUARD, SENTINEL);
    std::vector<float> custom(HQ * O_STRIDE + OUTPUT_GUARD, SENTINEL);

    fill_inputs(pattern, seed, visible, padded, q, k, v);

    const half negative_infinity =
        __float2half(-std::numeric_limits<float>::infinity());
    for (half & value : mask) {
        value = negative_infinity;
    }
    for (int pos = 0; pos < visible; ++pos) {
        mask[pos] = __float2half(pos == visible / 2 ? -0.125f : 0.0f);
    }

    const std::vector<double> reference = cpu_reference(q, k, v, mask, visible, padded);

    device_buffer<float> d_q(q.size());
    device_buffer<half> d_k(k.size());
    device_buffer<half> d_v(v.size());
    device_buffer<half> d_mask(mask.size());
    device_buffer<float> d_builtin(builtin.size());
    device_buffer<float> d_custom(custom.size());

    check_cuda(cudaMemcpy(d_q.data, q.data(), q.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "copy Q");
    check_cuda(cudaMemcpy(d_k.data, k.data(), k.size() * sizeof(half),
                          cudaMemcpyHostToDevice), "copy K");
    check_cuda(cudaMemcpy(d_v.data, v.data(), v.size() * sizeof(half),
                          cudaMemcpyHostToDevice), "copy V");
    check_cuda(cudaMemcpy(d_mask.data, mask.data(), mask.size() * sizeof(half),
                          cudaMemcpyHostToDevice), "copy mask");
    check_cuda(cudaMemcpy(d_builtin.data, builtin.data(), builtin.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "initialize built-in output");
    check_cuda(cudaMemcpy(d_custom.data, custom.data(), custom.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "initialize custom output");

    ggml_tensor tq;
    ggml_tensor tk;
    ggml_tensor tv;
    ggml_tensor tm;
    ggml_tensor td_builtin;
    ggml_tensor td_custom;

    set_tensor(tq, GGML_TYPE_F32, D, 1, HQ, 1,
               sizeof(float), D * sizeof(float),
               Q_STRIDE * sizeof(float), HQ * Q_STRIDE * sizeof(float));
    set_tensor(tk, GGML_TYPE_F16, D, padded, HKV, 1,
               sizeof(half), K_STRIDE * sizeof(half),
               padded * K_STRIDE * sizeof(half),
               HKV * padded * K_STRIDE * sizeof(half));
    set_tensor(tv, GGML_TYPE_F16, D, padded, HKV, 1,
               sizeof(half), V_STRIDE * sizeof(half),
               padded * V_STRIDE * sizeof(half),
               HKV * padded * V_STRIDE * sizeof(half));
    set_tensor(tm, GGML_TYPE_F16, padded, 16, 1, 1,
               sizeof(half), padded * sizeof(half),
               16 * padded * sizeof(half),
               16 * padded * sizeof(half));
    set_tensor(td_builtin, GGML_TYPE_F32, D, HQ, 1, 1,
               sizeof(float), O_STRIDE * sizeof(float),
               HQ * O_STRIDE * sizeof(float),
               HQ * O_STRIDE * sizeof(float));
    set_tensor(td_custom, GGML_TYPE_F32, D, HQ, 1, 1,
               sizeof(float), O_STRIDE * sizeof(float),
               HQ * O_STRIDE * sizeof(float),
               HQ * O_STRIDE * sizeof(float));

    tq.data = d_q.data;
    tk.data = d_k.data;
    tv.data = d_v.data;
    tm.data = d_mask.data;
    td_builtin.data = d_builtin.data;
    td_custom.data = d_custom.data;

    for (ggml_tensor * dst : {&td_builtin, &td_custom}) {
        dst->op = GGML_OP_FLASH_ATTN_EXT;
        dst->src[0] = &tq;
        dst->src[1] = &tk;
        dst->src[2] = &tv;
        dst->src[3] = &tm;
        std::memcpy(dst->op_params, &SCALE, sizeof(SCALE));
    }

    // The process starts with the runtime opt-in unset, so this dispatch uses
    // llama.cpp's normal built-in kernel. The custom launcher is then invoked
    // directly with the same tensors to avoid changing process-global routing.
    ggml_cuda_flash_attn_ext(context, &td_builtin);
    ggml_cuda_llama32_fa_decode(context, &td_custom);
    check_cuda(cudaGetLastError(), "launch");
    check_cuda(cudaDeviceSynchronize(), "synchronize");

    check_cuda(cudaMemcpy(builtin.data(), d_builtin.data,
                          builtin.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy built-in output");
    check_cuda(cudaMemcpy(custom.data(), d_custom.data,
                          custom.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy custom output");

    const comparison pair = compare_outputs(builtin, custom);
    const comparison builtin_ref = compare_reference(builtin, reference);
    const comparison custom_ref = compare_reference(custom, reference);
    const double count = static_cast<double>(HQ * D);
    const double custom_ref_mean = custom_ref.sum_abs / count;
    const bool custom_reference_within_limits =
        custom_ref.max_abs <= CUSTOM_REFERENCE_MAX_ABS_LIMIT &&
        custom_ref_mean <= CUSTOM_REFERENCE_MEAN_ABS_LIMIT;
    const bool ok =
        pair.finite && pair.padding_untouched &&
        builtin_ref.finite && custom_ref.finite &&
        custom_reference_within_limits;

    std::printf(
        "visible=%d padded=%d pattern=%s seed=%u "
        "builtin_custom_max=%.9g builtin_custom_mean=%.9g "
        "builtin_ref_max=%.9g builtin_ref_mean=%.9g "
        "custom_ref_max=%.9g custom_ref_mean=%.9g "
        "custom_ref_limits=%d max_head=%d max_dim=%d finite=%d output_guard=%d: %s\n",
        visible, padded, pattern_name(pattern), seed,
        pair.max_abs, pair.sum_abs / count,
        builtin_ref.max_abs, builtin_ref.sum_abs / count,
        custom_ref.max_abs, custom_ref_mean,
        custom_reference_within_limits ? 1 : 0,
        pair.max_head, pair.max_dim,
        (pair.finite && builtin_ref.finite && custom_ref.finite) ? 1 : 0,
        pair.padding_untouched ? 1 : 0, ok ? "PASS" : "FAIL");

    return ok;
}

} // namespace

int main() {
#if defined(_WIN32)
    if (_putenv_s("GGML_CUDA_LLAMA32_FA_DECODE_ENABLED", "") != 0) {
        std::perror("_putenv_s");
        return 1;
    }
#else
    if (unsetenv("GGML_CUDA_LLAMA32_FA_DECODE_ENABLED") != 0) {
        std::perror("unsetenv");
        return 1;
    }
#endif

    cudaDeviceProp properties;
    check_cuda(cudaGetDeviceProperties(&properties, 0), "get properties");
    std::printf("GGML built-in/custom FA-decode A/B on %s\n", properties.name);

    ggml_backend_cuda_context context(0);
    bool ok = true;

    // Length 1 is a first-decode smoke test. The remaining contexts are the
    // compact article sweep shared by correctness and benchmark reporting.
    // Seeds 1-3 calibrated the validation limits; held-out seeds 4-6 validate
    // them without changing the kernel or the test patterns.
    for (const int visible : {1, 128, 512, 2048, 4096, 8192}) {
        for (uint32_t seed = 4; seed <= 6; ++seed) {
            ok = run_case(context, visible, input_pattern::random, seed) && ok;
        }
        ok = run_case(context, visible, input_pattern::zeros, 0) && ok;
        ok = run_case(context, visible, input_pattern::dominant_score, 0) && ok;
        ok = run_case(context, visible, input_pattern::gqa_signature, 0) && ok;
    }

    std::printf("GGML built-in/custom article diagnostics: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
