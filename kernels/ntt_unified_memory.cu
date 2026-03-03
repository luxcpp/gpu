// =============================================================================
// Unified/Managed Memory NTT CUDA Kernels
// =============================================================================
// CUDA port of ntt_unified_memory.metal -- byte-identical arithmetic output.
//
// On CUDA, "unified memory" maps to cudaMallocManaged. The kernel structure
// is identical; the host-side allocation strategy differs.
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#ifdef __CUDA_ARCH__
#define NTT_DEVICE __device__ __forceinline__
#else
#define NTT_DEVICE inline
#define __global__
#define __shared__
static inline uint64_t __umul64hi(uint64_t a, uint64_t b) {
    __uint128_t r = (__uint128_t)a * b; return (uint64_t)(r >> 64);
}
static inline void __syncthreads() {}
#endif

static const uint32_t MAX_SHARED_TWIDDLES = 4096;

struct NTTUnifiedParams {
    uint64_t Q;
    uint64_t mu;
    uint64_t N_inv;
    uint64_t N_inv_precon;
    uint32_t N;
    uint32_t log_N;
    uint32_t stage;
    uint32_t batch;
};

NTT_DEVICE uint64_t barrett_mul_unified(uint64_t a, uint64_t b, uint64_t Q, uint64_t mu) {
    uint64_t lo = a * b;
    uint64_t q = __umul64hi(lo, mu);
    uint64_t result = lo - q * Q;
    // Branch-free conditional subtraction
    uint64_t mask = (result >= Q) ? ~0ULL : 0ULL;
    result -= (Q & mask);
    return result;
}

NTT_DEVICE uint64_t mod_add_unified(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    uint64_t mask = (sum >= Q) ? ~0ULL : 0ULL;
    return sum - (Q & mask);
}

NTT_DEVICE uint64_t mod_sub_unified(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t diff = a - b;
    uint64_t mask = (a < b) ? ~0ULL : 0ULL;
    return diff + (Q & mask);
}

NTT_DEVICE void ct_butterfly(uint64_t& lo, uint64_t& hi,
                              uint64_t tw, uint64_t Q, uint64_t mu) {
    uint64_t hi_tw = barrett_mul_unified(hi, tw, Q, mu);
    uint64_t new_lo = mod_add_unified(lo, hi_tw, Q);
    uint64_t new_hi = mod_sub_unified(lo, hi_tw, Q);
    lo = new_lo;
    hi = new_hi;
}

NTT_DEVICE void gs_butterfly(uint64_t& lo, uint64_t& hi,
                              uint64_t tw, uint64_t Q, uint64_t mu) {
    uint64_t sum = mod_add_unified(lo, hi, Q);
    uint64_t diff = mod_sub_unified(lo, hi, Q);
    lo = sum;
    hi = barrett_mul_unified(diff, tw, Q, mu);
}

// =============================================================================
// Forward NTT Stage
// =============================================================================

extern "C" __global__ void unified_ntt_forward_stage(
    uint64_t* data, const uint64_t* twiddles, const NTTUnifiedParams params)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared_tw[];
    uint32_t tid = threadIdx.x;
    uint32_t tg_size = blockDim.x;
    uint32_t batch_idx = blockIdx.x;
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint32_t stage = params.stage;
    uint32_t m = 1u << stage;
    uint32_t t = N >> (stage + 1);

    uint32_t tw_to_load = min(m, MAX_SHARED_TWIDDLES);
    uint32_t loads_per_thread = (tw_to_load + tg_size - 1) / tg_size;
    for (uint32_t i = 0; i < loads_per_thread; ++i) {
        uint32_t tw_idx = tid + i * tg_size;
        if (tw_idx < tw_to_load)
            shared_tw[tw_idx] = twiddles[m + tw_idx];
    }
    __syncthreads();

    uint64_t* batch_data = data + batch_idx * N;
    uint32_t butterflies_total = N / 2;
    uint32_t butterflies_per_thread = (butterflies_total + tg_size - 1) / tg_size;
    for (uint32_t b = 0; b < butterflies_per_thread; ++b) {
        uint32_t butterfly_idx = tid + b * tg_size;
        if (butterfly_idx >= butterflies_total) break;
        uint32_t group = butterfly_idx / t;
        uint32_t elem = butterfly_idx % t;
        uint32_t idx_lo = (group << (params.log_N - stage)) + elem;
        uint32_t idx_hi = idx_lo + t;
        uint64_t lo = batch_data[idx_lo];
        uint64_t hi = batch_data[idx_hi];
        uint64_t tw = (group < MAX_SHARED_TWIDDLES) ? shared_tw[group] : twiddles[m + group];
        ct_butterfly(lo, hi, tw, Q, mu);
        batch_data[idx_lo] = lo;
        batch_data[idx_hi] = hi;
    }
#endif
}

// =============================================================================
// Inverse NTT Stage
// =============================================================================

extern "C" __global__ void unified_ntt_inverse_stage(
    uint64_t* data, const uint64_t* twiddles, const NTTUnifiedParams params)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared_tw[];
    uint32_t tid = threadIdx.x;
    uint32_t tg_size = blockDim.x;
    uint32_t batch_idx = blockIdx.x;
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint32_t stage = params.stage;
    uint32_t m = N >> (stage + 1);
    uint32_t t = 1u << stage;

    uint32_t tw_to_load = min(m, MAX_SHARED_TWIDDLES);
    uint32_t loads_per_thread = (tw_to_load + tg_size - 1) / tg_size;
    for (uint32_t i = 0; i < loads_per_thread; ++i) {
        uint32_t tw_idx = tid + i * tg_size;
        if (tw_idx < tw_to_load)
            shared_tw[tw_idx] = twiddles[m + tw_idx];
    }
    __syncthreads();

    uint64_t* batch_data = data + batch_idx * N;
    uint32_t butterflies_total = N / 2;
    uint32_t bpt = (butterflies_total + tg_size - 1) / tg_size;
    for (uint32_t b = 0; b < bpt; ++b) {
        uint32_t butterfly_idx = tid + b * tg_size;
        if (butterfly_idx >= butterflies_total) break;
        uint32_t group = butterfly_idx / t;
        uint32_t elem = butterfly_idx % t;
        uint32_t idx_lo = (group << (stage + 1)) + elem;
        uint32_t idx_hi = idx_lo + t;
        uint64_t lo = batch_data[idx_lo];
        uint64_t hi = batch_data[idx_hi];
        uint64_t tw = (group < MAX_SHARED_TWIDDLES) ? shared_tw[group] : twiddles[m + group];
        gs_butterfly(lo, hi, tw, Q, mu);
        batch_data[idx_lo] = lo;
        batch_data[idx_hi] = hi;
    }
#endif
}

// =============================================================================
// Fused Forward NTT (all stages in shared memory, N <= 4096)
// =============================================================================

extern "C" __global__ void unified_ntt_forward_fused(
    uint64_t* data, const uint64_t* twiddles, const NTTUnifiedParams params)
{
#ifdef __CUDA_ARCH__
    // Shared memory: first half for twiddles, second half for data
    extern __shared__ uint64_t smem[];
    uint64_t* shared_tw = smem;
    uint64_t* shared_data = smem + MAX_SHARED_TWIDDLES;

    uint32_t tid = threadIdx.x;
    uint32_t tg_size = blockDim.x;
    uint32_t batch_idx = blockIdx.x;
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint32_t log_N = params.log_N;
    uint64_t* batch_data = data + batch_idx * N;

    // Load twiddles
    uint32_t total_twiddles = N - 1;
    for (uint32_t i = tid; i < total_twiddles && i < MAX_SHARED_TWIDDLES; i += tg_size) {
        shared_tw[i] = twiddles[i + 1];
    }
    // Load polynomial
    for (uint32_t i = tid; i < N; i += tg_size) {
        shared_data[i] = batch_data[i];
    }
    __syncthreads();

    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = 1u << stage;
        uint32_t t = N >> (stage + 1);
        uint32_t tw_offset = m - 1;
        uint32_t bpt = (N / 2 + tg_size - 1) / tg_size;
        for (uint32_t b = 0; b < bpt; ++b) {
            uint32_t butterfly_idx = tid + b * tg_size;
            if (butterfly_idx >= N / 2) break;
            uint32_t group = butterfly_idx / t;
            uint32_t elem = butterfly_idx % t;
            uint32_t idx_lo = (group << (log_N - stage)) + elem;
            uint32_t idx_hi = idx_lo + t;
            uint64_t lo = shared_data[idx_lo];
            uint64_t hi = shared_data[idx_hi];
            uint64_t tw = shared_tw[tw_offset + group];
            ct_butterfly(lo, hi, tw, Q, mu);
            shared_data[idx_lo] = lo;
            shared_data[idx_hi] = hi;
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < N; i += tg_size) {
        batch_data[i] = shared_data[i];
    }
#endif
}

// =============================================================================
// Fused Inverse NTT
// =============================================================================

extern "C" __global__ void unified_ntt_inverse_fused(
    uint64_t* data, const uint64_t* twiddles, const NTTUnifiedParams params)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t smem[];
    uint64_t* shared_tw = smem;
    uint64_t* shared_data = smem + MAX_SHARED_TWIDDLES;

    uint32_t tid = threadIdx.x;
    uint32_t tg_size = blockDim.x;
    uint32_t batch_idx = blockIdx.x;
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint64_t N_inv = params.N_inv;
    uint32_t log_N = params.log_N;
    uint64_t* batch_data = data + batch_idx * N;

    uint32_t total_twiddles = N - 1;
    for (uint32_t i = tid; i < total_twiddles && i < MAX_SHARED_TWIDDLES; i += tg_size) {
        shared_tw[i] = twiddles[i + 1];
    }
    for (uint32_t i = tid; i < N; i += tg_size) {
        shared_data[i] = batch_data[i];
    }
    __syncthreads();

    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = N >> (stage + 1);
        uint32_t t = 1u << stage;
        uint32_t tw_offset = m - 1;
        uint32_t bpt = (N / 2 + tg_size - 1) / tg_size;
        for (uint32_t b = 0; b < bpt; ++b) {
            uint32_t butterfly_idx = tid + b * tg_size;
            if (butterfly_idx >= N / 2) break;
            uint32_t group = butterfly_idx / t;
            uint32_t elem = butterfly_idx % t;
            uint32_t idx_lo = (group << (stage + 1)) + elem;
            uint32_t idx_hi = idx_lo + t;
            uint64_t lo = shared_data[idx_lo];
            uint64_t hi = shared_data[idx_hi];
            uint64_t tw = shared_tw[tw_offset + group];
            gs_butterfly(lo, hi, tw, Q, mu);
            shared_data[idx_lo] = lo;
            shared_data[idx_hi] = hi;
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < N; i += tg_size) {
        batch_data[i] = barrett_mul_unified(shared_data[i], N_inv, Q, mu);
    }
#endif
}

// =============================================================================
// Scaling and Pointwise Kernels
// =============================================================================

extern "C" __global__ void unified_scale_ninv(uint64_t* data, const NTTUnifiedParams params)
{
#ifdef __CUDA_ARCH__
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    uint32_t total = params.N * params.batch;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint64_t N_inv = params.N_inv;
    for (uint32_t i = tid; i < total; i += stride) {
        data[i] = barrett_mul_unified(data[i], N_inv, Q, mu);
    }
#endif
}

extern "C" __global__ void unified_pointwise_mul(
    uint64_t* result, const uint64_t* a, const uint64_t* b, const NTTUnifiedParams params)
{
#ifdef __CUDA_ARCH__
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    uint32_t total = params.N * params.batch;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    for (uint32_t i = tid; i < total; i += stride) {
        result[i] = barrett_mul_unified(a[i], b[i], Q, mu);
    }
#endif
}

extern "C" __global__ void unified_pointwise_add(
    uint64_t* result, const uint64_t* a, const uint64_t* b, const NTTUnifiedParams params)
{
#ifdef __CUDA_ARCH__
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    uint32_t total = params.N * params.batch;
    uint64_t Q = params.Q;
    for (uint32_t i = tid; i < total; i += stride) {
        result[i] = mod_add_unified(a[i], b[i], Q);
    }
#endif
}

extern "C" __global__ void unified_pointwise_sub(
    uint64_t* result, const uint64_t* a, const uint64_t* b, const NTTUnifiedParams params)
{
#ifdef __CUDA_ARCH__
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    uint32_t total = params.N * params.batch;
    uint64_t Q = params.Q;
    for (uint32_t i = tid; i < total; i += stride) {
        result[i] = mod_sub_unified(a[i], b[i], Q);
    }
#endif
}

extern "C" __global__ void unified_memcpy(uint64_t* dst, const uint64_t* src, uint32_t count)
{
#ifdef __CUDA_ARCH__
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = tid; i < count; i += stride) {
        dst[i] = src[i];
    }
#endif
}
