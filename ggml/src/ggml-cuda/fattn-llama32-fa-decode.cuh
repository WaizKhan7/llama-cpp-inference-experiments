#pragma once

#include <cstdlib>
#include <cstring>
#include <math_constants.h>

// This predicate freezes the first llama.cpp integration contract. The
// dispatcher calls the custom route only after both the runtime opt-in and
// this predicate accept the operation.
// The raw parity kernel has different inputs: FP16 Q/K/V and fused RoPE.
// GGML has FP32 Q/output and pre-rotated Q/K at this operation boundary.
#ifdef GGML_CUDA_LLAMA32_FA_DECODE
#ifdef GGML_CUDA_LLAMA32_FA_DECODE_TEST_HOOK
extern int ggml_cuda_llama32_fa_decode_test_dispatch_count;
extern int ggml_cuda_llama32_fa_decode_test_route;
#endif

// Normal builds require an explicit process-level opt-in. This lets one
// compiled llama.cpp binary provide a faithful built-in baseline and the
// custom FA-decode path. The focused CUDA fixture sets this variable itself.
// Its test hook counts selections only; it must never force the custom route.
static inline bool ggml_cuda_llama32_fa_decode_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("GGML_CUDA_LLAMA32_FA_DECODE_ENABLED");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}
// Split-K is separately opt-in. Requiring the existing FA opt-in as well
// preserves baseline behaviour and makes every experimental selection explicit.
static inline bool ggml_cuda_llama32_fd_splitk_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("GGML_CUDA_LLAMA32_FD_SPLITK_ENABLED");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

static inline bool ggml_cuda_llama32_fa_decode_trace_enabled() {
    const char * value = std::getenv("GGML_CUDA_LLAMA32_FA_DECODE_TRACE");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

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

    if (dst->ne[0] != 64 || dst->ne[1] != 32 ||
        dst->ne[2] != 1  || dst->ne[3] != 1) {
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

// Four CUDA warps cooperate on one query head. Each lane owns dimensions lane
// and lane + 32; each warp processes a disjoint quarter of the KV positions.
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
    constexpr int warps_per_head = 4;
    constexpr int half_dim = 32;

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    const int query_head = blockIdx.x;
    const int kv_head = query_head / 4;
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

    // Each warp scans a disjoint interleaved KV slice. This exposes four times
    // as many active warps without changing the GQA head mapping or output ABI.
    for (int key_pos = warp_id; key_pos < kv_len; key_pos += warps_per_head) {
        const float loaded_mask = lane == 0
            ? __half2float(*reinterpret_cast<const half *>(mask + key_pos * mask_nb0))
            : 0.0f;
        const float mask_value = __shfl_sync(0xffffffffu, loaded_mask, 0);
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

    // Merge the four independently normalized online-softmax partitions.
    // The accumulators are stored in their local exp(running_max) scale.
    __shared__ float partial_max[warps_per_head];
    __shared__ float partial_sum[warps_per_head];
    __shared__ float partial_accumulator[warps_per_head][2 * half_dim];

    if (lane == 0) {
        partial_max[warp_id] = running_max;
        partial_sum[warp_id] = running_sum;
    }
    partial_accumulator[warp_id][d0] = accumulator0;
    partial_accumulator[warp_id][d1] = accumulator1;
    __syncthreads();

    float merged_max = partial_max[0];
#pragma unroll
    for (int warp = 1; warp < warps_per_head; ++warp) {
        merged_max = fmaxf(merged_max, partial_max[warp]);
    }

    float merged_sum = 0.0f;
    float merged_accumulator0 = 0.0f;
    float merged_accumulator1 = 0.0f;
#pragma unroll
    for (int warp = 0; warp < warps_per_head; ++warp) {
        const float partition_scale = partial_sum[warp] > 0.0f
            ? expf(partial_max[warp] - merged_max)
            : 0.0f;
        merged_sum += partial_sum[warp] * partition_scale;
        merged_accumulator0 += partial_accumulator[warp][d0] * partition_scale;
        merged_accumulator1 += partial_accumulator[warp][d1] * partition_scale;
    }

    const float inverse_sum = merged_sum > 0.0f ? 1.0f / merged_sum : 0.0f;
    *reinterpret_cast<float *>(out_head + d0 * dst_nb0) = merged_accumulator0 * inverse_sum;
    *reinterpret_cast<float *>(out_head + d1 * dst_nb0) = merged_accumulator1 * inverse_sum;
}

// Experimental two-pass Split-K / Flash-Decoding implementation. A partial
// block owns one 128-token chunk for one query head and writes its unnormalised
// stable-softmax state; the reduction block merges chunk states per head.
static constexpr int llama32_fd_chunk = 128;
static constexpr int llama32_fd_dim = 64;

static __global__ void llama32_fd_partial(
        const char * Q, const char * K, const char * V, const char * mask,
        float * partial_out, float * partial_max, float * partial_lse,
        float scale, int kv_len, int chunks,
        size_t q_nb0, size_t q_nb2, size_t k_nb0, size_t k_nb1, size_t k_nb2,
        size_t v_nb0, size_t v_nb1, size_t v_nb2, size_t mask_nb0) {
    const int lane = threadIdx.x;
    const int chunk = blockIdx.x;
    const int head = blockIdx.y;
    const int kv_head = head / 4;
    const int begin = chunk * llama32_fd_chunk;
    const int end = min(begin + llama32_fd_chunk, kv_len);
    const char * q_head = Q + head * q_nb2;
    const char * k_head = K + kv_head * k_nb2;
    const char * v_head = V + kv_head * v_nb2;
    const float q0 = *reinterpret_cast<const float *>(q_head + lane * q_nb0);
    const float q1 = *reinterpret_cast<const float *>(q_head + (lane + 32) * q_nb0);
    float max_score = -CUDART_INF_F, sum = 0.0f, acc0 = 0.0f, acc1 = 0.0f;

    for (int pos = begin; pos < end; ++pos) {
        const float loaded_mask = lane == 0
            ? __half2float(*reinterpret_cast<const half *>(mask + pos * mask_nb0)) : 0.0f;
        const float mask_value = __shfl_sync(0xffffffffu, loaded_mask, 0);
        if (mask_value == -CUDART_INF_F) continue;
        const char * k_row = k_head + pos * k_nb1;
        const float k0 = __half2float(*reinterpret_cast<const half *>(k_row + lane * k_nb0));
        const float k1 = __half2float(*reinterpret_cast<const half *>(k_row + (lane + 32) * k_nb0));
        const float score = llama32_fa_decode_warp_sum(q0 * k0 + q1 * k1) * scale + mask_value;
        const float new_max = fmaxf(max_score, score);
        const float old_scale = sum > 0.0f ? expf(max_score - new_max) : 0.0f;
        const float weight = expf(score - new_max);
        const char * v_row = v_head + pos * v_nb1;
        const float v0 = __half2float(*reinterpret_cast<const half *>(v_row + lane * v_nb0));
        const float v1 = __half2float(*reinterpret_cast<const half *>(v_row + (lane + 32) * v_nb0));
        acc0 = acc0 * old_scale + weight * v0;
        acc1 = acc1 * old_scale + weight * v1;
        sum = sum * old_scale + weight;
        max_score = new_max;
    }
    const int index = head * chunks + chunk;
    partial_out[index * llama32_fd_dim + lane] = acc0;
    partial_out[index * llama32_fd_dim + lane + 32] = acc1;
    if (lane == 0) {
        partial_max[index] = max_score;
        partial_lse[index] = sum > 0.0f ? logf(sum) : -CUDART_INF_F;
    }
}

static __global__ void llama32_fd_reduce(
        const float * partial_out, const float * partial_max, const float * partial_lse,
        char * output, int chunks, size_t dst_nb0, size_t dst_nb1) {
    const int lane = threadIdx.x;
    const int head = blockIdx.x;
    float max_score = -CUDART_INF_F, sum = 0.0f, acc0 = 0.0f, acc1 = 0.0f;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int index = head * chunks + chunk;
        const float chunk_max = partial_max[index];
        const float chunk_lse = partial_lse[index];
        if (!isfinite(chunk_max) || !isfinite(chunk_lse)) continue;
        const float new_max = fmaxf(max_score, chunk_max);
        const float old_scale = sum > 0.0f ? expf(max_score - new_max) : 0.0f;
        const float chunk_scale = expf(chunk_max - new_max);
        acc0 = acc0 * old_scale + partial_out[index * llama32_fd_dim + lane] * chunk_scale;
        acc1 = acc1 * old_scale + partial_out[index * llama32_fd_dim + lane + 32] * chunk_scale;
        sum = sum * old_scale + expf(chunk_lse) * chunk_scale;
        max_score = new_max;
    }
    const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    char * out = output + head * dst_nb1;
    *reinterpret_cast<float *>(out + lane * dst_nb0) = acc0 * inv_sum;
    *reinterpret_cast<float *>(out + (lane + 32) * dst_nb0) = acc1 * inv_sum;
}

// The dispatcher calls this launcher only after the strict decode predicate passes.
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
    const dim3 block(128, 1, 1);
    llama32_fa_decode_ggml_kernel<<<grid, block, 0, ctx.stream()>>>(
        (const char *) Q->data, (const char *) K->data, (const char *) V->data,
        (const char *) mask->data, (char *) dst->data, scale, K->ne[1],
        Q->nb[0], Q->nb[2],
        K->nb[0], K->nb[1], K->nb[2],
        V->nb[0], V->nb[1], V->nb[2],
        mask->nb[0], dst->nb[0], dst->nb[1]);
    CUDA_CHECK(cudaGetLastError());
}


// This launcher is intentionally not routed by fattn.cu yet. It is exercised
// only by the direct FP64 A/B fixture until its correctness gates pass.
static inline void ggml_cuda_llama32_fd_splitk(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_llama32_fa_decode_supported(dst));
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params, sizeof(float));
    const int chunks = (K->ne[1] + llama32_fd_chunk - 1) / llama32_fd_chunk;
    const size_t count = static_cast<size_t>(Q->ne[2]) * chunks;
    ggml_cuda_pool_alloc<float> partial_out(ctx.pool(), count * llama32_fd_dim);
    ggml_cuda_pool_alloc<float> partial_max(ctx.pool(), count);
    ggml_cuda_pool_alloc<float> partial_lse(ctx.pool(), count);
    const dim3 partial_grid(chunks, Q->ne[2], 1);
    const dim3 block(32, 1, 1);
    llama32_fd_partial<<<partial_grid, block, 0, ctx.stream()>>>(
        (const char *) Q->data, (const char *) K->data, (const char *) V->data,
        (const char *) mask->data, partial_out.get(), partial_max.get(), partial_lse.get(),
        scale, K->ne[1], chunks, Q->nb[0], Q->nb[2], K->nb[0], K->nb[1], K->nb[2],
        V->nb[0], V->nb[1], V->nb[2], mask->nb[0]);
    CUDA_CHECK(cudaGetLastError());
    llama32_fd_reduce<<<dim3(Q->ne[2], 1, 1), block, 0, ctx.stream()>>>(
        partial_out.get(), partial_max.get(), partial_lse.get(), (char *) dst->data,
        chunks, dst->nb[0], dst->nb[1]);
    CUDA_CHECK(cudaGetLastError());
}

#endif // GGML_CUDA_LLAMA32_FA_DECODE
