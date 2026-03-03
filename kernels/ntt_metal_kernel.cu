// =============================================================================
// NTT CUDA Kernels with Shared Memory Twiddle Prefetch
// =============================================================================
// CUDA port of ntt_metal_kernel.metal -- byte-identical arithmetic output.
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
static inline void __threadfence_block() {}
#endif

struct NTTMetalParams {
    uint64_t Q;
    uint64_t mu;
    uint64_t N_inv;
    uint64_t N_inv_precon;
    uint32_t N;
    uint32_t log_N;
    uint32_t stage;
    uint32_t batch;
};

NTT_DEVICE uint64_t barrett_mul(uint64_t a, uint64_t b, uint64_t Q, uint64_t mu) {
    uint64_t lo = a * b;
    uint64_t q = __umul64hi(lo, mu);
    uint64_t result = lo - q * Q;
    if (result >= Q) result -= Q;
    return result;
}

NTT_DEVICE uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    return (sum >= Q) ? sum - Q : sum;
}

NTT_DEVICE uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    return (a >= b) ? a - b : a + Q - b;
}

static const uint32_t MAX_SHARED_TWIDDLES = 4096;

extern "C" __global__ void ntt_forward_stage_shared(
    uint64_t* data, const uint64_t* twiddles, const NTTMetalParams params)
{
#ifdef __CUDA_ARCH__
    __shared__ uint64_t twiddles_shared[MAX_SHARED_TWIDDLES];
    uint32_t thread_idx = threadIdx.x;
    uint32_t threadgroup_size = blockDim.x;
    uint32_t batch_idx = blockIdx.x;
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint32_t stage = params.stage;
    uint32_t m = 1u << stage;
    uint32_t t = N >> (stage + 1);

    uint32_t twiddles_to_load = m;
    uint32_t loads_per_thread = (twiddles_to_load + threadgroup_size - 1) / threadgroup_size;
    for (uint32_t i = 0; i < loads_per_thread; ++i) {
        uint32_t tw_idx = thread_idx + i * threadgroup_size;
        if (tw_idx < m && tw_idx < MAX_SHARED_TWIDDLES)
            twiddles_shared[tw_idx] = twiddles[m + tw_idx];
    }
    __syncthreads();

    uint64_t* batch_data = data + batch_idx * N;
    uint32_t butterflies_per_thread = (N / 2 + threadgroup_size - 1) / threadgroup_size;
    for (uint32_t b = 0; b < butterflies_per_thread; ++b) {
        uint32_t butterfly_idx = thread_idx + b * threadgroup_size;
        if (butterfly_idx >= N / 2) break;
        uint32_t group = butterfly_idx / t;
        uint32_t elem = butterfly_idx % t;
        uint32_t idx_lo = (group << (params.log_N - stage)) + elem;
        uint32_t idx_hi = idx_lo + t;
        uint64_t lo = batch_data[idx_lo];
        uint64_t hi = batch_data[idx_hi];
        uint64_t tw = twiddles_shared[group];
        uint64_t hi_tw = barrett_mul(hi, tw, Q, mu);
        batch_data[idx_lo] = mod_add(lo, hi_tw, Q);
        batch_data[idx_hi] = mod_sub(lo, hi_tw, Q);
    }
#endif
}

extern "C" __global__ void ntt_inverse_stage_shared(
    uint64_t* data, const uint64_t* twiddles, const NTTMetalParams params)
{
#ifdef __CUDA_ARCH__
    __shared__ uint64_t twiddles_shared[MAX_SHARED_TWIDDLES];
    uint32_t thread_idx = threadIdx.x;
    uint32_t threadgroup_size = blockDim.x;
    uint32_t batch_idx = blockIdx.x;
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint32_t stage = params.stage;
    uint32_t m = N >> (stage + 1);
    uint32_t t = 1u << stage;

    uint32_t twiddles_to_load = m;
    uint32_t loads_per_thread = (twiddles_to_load + threadgroup_size - 1) / threadgroup_size;
    for (uint32_t i = 0; i < loads_per_thread; ++i) {
        uint32_t tw_idx = thread_idx + i * threadgroup_size;
        if (tw_idx < m && tw_idx < MAX_SHARED_TWIDDLES)
            twiddles_shared[tw_idx] = twiddles[m + tw_idx];
    }
    __syncthreads();

    uint64_t* batch_data = data + batch_idx * N;
    uint32_t butterflies_per_thread = (N / 2 + threadgroup_size - 1) / threadgroup_size;
    for (uint32_t b = 0; b < butterflies_per_thread; ++b) {
        uint32_t butterfly_idx = thread_idx + b * threadgroup_size;
        if (butterfly_idx >= N / 2) break;
        uint32_t group = butterfly_idx / t;
        uint32_t elem = butterfly_idx % t;
        uint32_t idx_lo = (group << (stage + 1)) + elem;
        uint32_t idx_hi = idx_lo + t;
        uint64_t lo = batch_data[idx_lo];
        uint64_t hi = batch_data[idx_hi];
        uint64_t tw = twiddles_shared[group];
        batch_data[idx_lo] = mod_add(lo, hi, Q);
        uint64_t diff = mod_sub(lo, hi, Q);
        batch_data[idx_hi] = barrett_mul(diff, tw, Q, mu);
    }
#endif
}

extern "C" __global__ void ntt_forward_fused(
    uint64_t* data, const uint64_t* twiddles_flat,
    const uint32_t* stage_offsets, const NTTMetalParams params)
{
#ifdef __CUDA_ARCH__
    __shared__ uint64_t twiddles_shared[MAX_SHARED_TWIDDLES];
    uint32_t thread_idx = threadIdx.x;
    uint32_t threadgroup_size = blockDim.x;
    uint32_t batch_idx = blockIdx.x;
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint32_t log_N = params.log_N;
    uint64_t* batch_data = data + batch_idx * N;

    uint32_t total_twiddles = N - 1;
    uint32_t loads_per_thread = (total_twiddles + threadgroup_size - 1) / threadgroup_size;
    for (uint32_t i = 0; i < loads_per_thread; ++i) {
        uint32_t tw_idx = thread_idx + i * threadgroup_size;
        if (tw_idx < total_twiddles && tw_idx < MAX_SHARED_TWIDDLES)
            twiddles_shared[tw_idx] = twiddles_flat[tw_idx];
    }
    __syncthreads();

    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = 1u << stage;
        uint32_t t = N >> (stage + 1);
        uint32_t tw_base = m;
        uint32_t bpt = (N / 2 + threadgroup_size - 1) / threadgroup_size;
        for (uint32_t b = 0; b < bpt; ++b) {
            uint32_t butterfly_idx = thread_idx + b * threadgroup_size;
            if (butterfly_idx >= N / 2) break;
            uint32_t group = butterfly_idx / t;
            uint32_t elem = butterfly_idx % t;
            uint32_t idx_lo = (group << (log_N - stage)) + elem;
            uint32_t idx_hi = idx_lo + t;
            uint64_t lo = batch_data[idx_lo];
            uint64_t hi = batch_data[idx_hi];
            uint64_t tw = twiddles_shared[tw_base + group];
            uint64_t hi_tw = barrett_mul(hi, tw, Q, mu);
            batch_data[idx_lo] = mod_add(lo, hi_tw, Q);
            batch_data[idx_hi] = mod_sub(lo, hi_tw, Q);
        }
        __threadfence_block();
        __syncthreads();
    }
#endif
}

extern "C" __global__ void ntt_scale_ninv(uint64_t* data, const NTTMetalParams params)
{
#ifdef __CUDA_ARCH__
    uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = params.N * params.batch;
    if (global_idx >= total) return;
    data[global_idx] = barrett_mul(data[global_idx], params.N_inv, params.Q, params.mu);
#endif
}

extern "C" __global__ void pointwise_mul_mod(
    uint64_t* result, const uint64_t* a, const uint64_t* b, const NTTMetalParams params)
{
#ifdef __CUDA_ARCH__
    uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = params.N * params.batch;
    if (global_idx >= total) return;
    result[global_idx] = barrett_mul(a[global_idx], b[global_idx], params.Q, params.mu);
#endif
}
