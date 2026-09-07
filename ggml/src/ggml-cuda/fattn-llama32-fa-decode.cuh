#pragma once

#include <math_constants.h>

// This predicate intentionally has no caller yet. It freezes the first
// llama.cpp integration contract before a new CUDA kernel is routed.
// The raw parity kernel has different inputs: FP16 Q/K/V and fused RoPE.
// GGML has FP32 Q/output and pre-rotated Q/K at this operation boundary.
#ifdef GGML_CUDA_LLAMA32_FA_DECODE
static inline bool ggml_cuda_llama32_fa_decode_supported(const ggml_tensor * dst) {
    if (!dst || dst->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (!Q || !K || !V || !mask || sinks) {
        return false;
    }

    // Version 0 is batch-one, one-token Llama 3.2-1B decode only.
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 ||
        V->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32 ||
        mask->type != GGML_TYPE_F16) {
        return false;
    }

    if (Q->ne[0] != 64 || Q->ne[1] != 1 || Q->ne[2] != 32 || Q->ne[3] != 1 ||
        K->ne[0] != 64 || K->ne[2] != 8  || K->ne[3] != 1 ||
        V->ne[0] != 64 || V->ne[2] != 8  || V->ne[3] != 1 ||
        K->ne[1] != V->ne[1] || K->ne[1] == 0 || K->ne[1] % 256 != 0) {
        return false;
    }

    // The existing CUDA FlashAttention operator requires this padded mask.
    if (mask->ne[0] != K->ne[1] || mask->ne[1] < 16 ||
        mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }

    // The specialized kernel dereferences scalar elements directly. It may
    // accept padding between rows and heads, but never inside a vector.
    if (Q->nb[0] != sizeof(float) || K->nb[0] != sizeof(half) ||
        V->nb[0] != sizeof(half) || mask->nb[0] != sizeof(half) ||
        dst->nb[0] != sizeof(float)) {
        return false;
    }

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    // ALiBi, logit softcap, and attention sinks are intentionally deferred.
    return max_bias == 0.0f && logit_softcap == 0.0f;
}

// One CUDA warp owns one query head. Each lane owns dimensions lane and
// lane + 32, matching the readable raw CUDA FA-decode implementation.
static __device__ __forceinline__ float llama32_fa_decode_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

// GGML-boundary core: Q and K are already RoPE-transformed. K/V and mask use
// true GGML byte strides; no contiguous copy or KV-head expansion is allowed.
static __global__ void llama32_fa_decode_ggml_kernel(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
              char * __restrict__ output,
        float scale,
        int kv_len,
        size_t q_nb0, size_t q_nb2,
        size_t k_nb0, size_t k_nb1, size_t k_nb2,
        size_t v_nb0, size_t v_nb1, size_t v_nb2,
        size_t mask_nb0,
        size_t dst_nb0, size_t dst_nb1) {
    const int lane = threadIdx.x;
    const int query_head = blockIdx.x;
    const int kv_head = query_head / 4;
    constexpr int half_dim = 32;
    const int d0 = lane;
    const int d1 = lane + half_dim;

    const char * q_head = Q + query_head * q_nb2;
    const char * k_head = K + kv_head * k_nb2;
    const char * v_head = V + kv_head * v_nb2;
    char * out_head = output + query_head * dst_nb1;

    const float q0 = *reinterpret_cast<const float *>(q_head + d0 * q_nb0);
    const float q1 = *reinterpret_cast<const float *>(q_head + d1 * q_nb0);

    float running_max = -CUDART_INF_F;
    float running_sum = 0.0f;
    float accumulator0 = 0.0f;
    float accumulator1 = 0.0f;

    for (int tile_start = 0; tile_start < kv_len; tile_start += 128) {
        const int tile_end = tile_start + 128 < kv_len ? tile_start + 128 : kv_len;
        for (int key_pos = tile_start; key_pos < tile_end; ++key_pos) {
            const float mask_value = __half2float(
                *reinterpret_cast<const half *>(mask + key_pos * mask_nb0));
            if (mask_value == -CUDART_INF_F) {
                continue;
            }

            const char * k_row = k_head + key_pos * k_nb1;
            const float k0 = __half2float(*reinterpret_cast<const half *>(k_row + d0 * k_nb0));
            const float k1 = __half2float(*reinterpret_cast<const half *>(k_row + d1 * k_nb0));
            float score = llama32_fa_decode_warp_sum(q0 * k0 + q1 * k1);
            score = score * scale + mask_value;

            const float new_max = fmaxf(running_max, score);
            const float old_scale = expf(running_max - new_max);
            const float weight = expf(score - new_max);

            const char * v_row = v_head + key_pos * v_nb1;
            const float value0 = __half2float(*reinterpret_cast<const half *>(v_row + d0 * v_nb0));
            const float value1 = __half2float(*reinterpret_cast<const half *>(v_row + d1 * v_nb0));
            accumulator0 = accumulator0 * old_scale + weight * value0;
            accumulator1 = accumulator1 * old_scale + weight * value1;
            running_sum = running_sum * old_scale + weight;
            running_max = new_max;
        }
    }

    // A valid causal decode always has at least the just-written current key.
    const float inverse_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    *reinterpret_cast<float *>(out_head + d0 * dst_nb0) = accumulator0 * inverse_sum;
    *reinterpret_cast<float *>(out_head + d1 * dst_nb0) = accumulator1 * inverse_sum;
}

// This launcher is intentionally not called by ggml_cuda_flash_attn_ext yet.
static inline void ggml_cuda_llama32_fa_decode(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_llama32_fa_decode_supported(dst));

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params, sizeof(float));

    const dim3 grid(Q->ne[2], 1, 1);
    const dim3 block(32, 1, 1);
    llama32_fa_decode_ggml_kernel<<<grid, block, 0, ctx.stream()>>>(
        (const char *) Q->data, (const char *) K->data, (const char *) V->data,
        (const char *) mask->data, (char *) dst->data, scale, K->ne[1],
        Q->nb[0], Q->nb[2],
        K->nb[0], K->nb[1], K->nb[2],
        V->nb[0], V->nb[1], V->nb[2],
        mask->nb[0], dst->nb[0], dst->nb[1]);
    CUDA_CHECK(cudaGetLastError());
}
#endif // GGML_CUDA_LLAMA32_FA_DECODE
