#include <../common.cuh>
#include <../fattn.cuh>
#include <../fattn-llama32-fa-decode.cuh>

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
constexpr int Q_STRIDE = 68;
constexpr int K_STRIDE = 80;
constexpr int V_STRIDE = 96;
constexpr int O_STRIDE = 72;
constexpr int PADDED_KV = 256;
constexpr float SCALE = 0.125f;
constexpr float SENTINEL = -777.0f;

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

struct comparison {
    bool finite = true;
    bool padding_untouched = true;
    float max_abs = 0.0f;
    double sum_abs = 0.0;
    int max_head = -1;
    int max_dim = -1;
};

comparison compare_outputs(
        const std::vector<float> & builtin,
        const std::vector<float> & custom) {
    comparison result;

    for (int head = 0; head < HQ; ++head) {
        for (int dim = 0; dim < D; ++dim) {
            const float a = builtin[head * O_STRIDE + dim];
            const float b = custom[head * O_STRIDE + dim];
            const float error = std::fabs(a - b);

            result.finite = result.finite && std::isfinite(a) && std::isfinite(b);
            result.sum_abs += error;
            if (error > result.max_abs) {
                result.max_abs = error;
                result.max_head = head;
                result.max_dim = dim;
            }
        }

        for (int dim = D; dim < O_STRIDE; ++dim) {
            result.padding_untouched =
                result.padding_untouched &&
                builtin[head * O_STRIDE + dim] == SENTINEL &&
                custom[head * O_STRIDE + dim] == SENTINEL;
        }
    }

    return result;
}

bool run_case(ggml_backend_cuda_context & context, int visible, uint32_t seed) {
    std::vector<float> q(HQ * Q_STRIDE);
    std::vector<half> k(HKV * PADDED_KV * K_STRIDE);
    std::vector<half> v(HKV * PADDED_KV * V_STRIDE);
    std::vector<half> mask(16 * PADDED_KV);
    std::vector<float> builtin(HQ * O_STRIDE, SENTINEL);
    std::vector<float> custom(HQ * O_STRIDE, SENTINEL);

    uint32_t state = seed;
    for (float & value : q) {
        value = 1.25f * random_signed(state);
    }

    // Padded cache positions contain large nonzero values. Correct masking must
    // prevent them from affecting either attention result.
    for (half & value : k) {
        value = __float2half(8.0f * random_signed(state));
    }
    for (half & value : v) {
        value = __float2half(8.0f * random_signed(state));
    }

    for (int head = 0; head < HKV; ++head) {
        for (int pos = 0; pos < visible; ++pos) {
            for (int dim = 0; dim < D; ++dim) {
                k[(head * PADDED_KV + pos) * K_STRIDE + dim] =
                    __float2half(0.50f * random_signed(state));
                v[(head * PADDED_KV + pos) * V_STRIDE + dim] =
                    __float2half(0.75f * random_signed(state));
            }
        }
    }

    const half negative_infinity =
        __float2half(-std::numeric_limits<float>::infinity());
    for (half & value : mask) {
        value = negative_infinity;
    }
    for (int pos = 0; pos < visible; ++pos) {
        mask[pos] = __float2half(0.0f);
    }

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
    set_tensor(tk, GGML_TYPE_F16, D, PADDED_KV, HKV, 1,
               sizeof(half), K_STRIDE * sizeof(half),
               PADDED_KV * K_STRIDE * sizeof(half),
               HKV * PADDED_KV * K_STRIDE * sizeof(half));
    set_tensor(tv, GGML_TYPE_F16, D, PADDED_KV, HKV, 1,
               sizeof(half), V_STRIDE * sizeof(half),
               PADDED_KV * V_STRIDE * sizeof(half),
               HKV * PADDED_KV * V_STRIDE * sizeof(half));
    set_tensor(tm, GGML_TYPE_F16, PADDED_KV, 16, 1, 1,
               sizeof(half), PADDED_KV * sizeof(half),
               16 * PADDED_KV * sizeof(half),
               16 * PADDED_KV * sizeof(half));
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

    const comparison result = compare_outputs(builtin, custom);
    const double mean_abs = result.sum_abs / static_cast<double>(HQ * D);
    const bool ok = result.finite && result.padding_untouched;

    std::printf(
        "visible=%d padded=%d seed=%u max_abs=%.9g mean_abs=%.9g "
        "max_head=%d max_dim=%d finite=%d padding=%d: %s\n",
        visible, PADDED_KV, seed, result.max_abs, mean_abs,
        result.max_head, result.max_dim, result.finite ? 1 : 0,
        result.padding_untouched ? 1 : 0, ok ? "PASS" : "FAIL");

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
    for (int visible = 12; visible <= 21; ++visible) {
        for (uint32_t seed = 1; seed <= 5; ++seed) {
            ok = run_case(context, visible, seed) && ok;
        }
    }

    std::printf("GGML built-in/custom A/B diagnostics: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
