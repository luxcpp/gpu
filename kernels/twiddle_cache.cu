// =============================================================================
// Twiddle Hotset Caching Kernels for CUDA
// =============================================================================
// CUDA port of twiddle_cache.metal -- byte-identical arithmetic output.
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-2-Clause

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

static const uint32_t MAX_THREADGROUP_TWIDDLES = 4096;
static const uint32_t FIRST_LEVEL_TWIDDLE_COUNT = 8;
static const uint32_t MAX_RNS_PRIMES = 16;
static const uint32_t BANK_WIDTH = 32;
static const uint32_t BANK_PADDING = 1;

struct PrimeConstants {
    uint64_t q;
    uint64_t q_inv;
    uint64_t mu_hi;
    uint64_t mu_lo;
    uint64_t r_squared;
    uint64_t root;
    uint64_t root_inv;
    uint64_t n_inv;
};

struct ConstantCache {
    uint32_t numPrimes;
    uint32_t ringDim;
    uint32_t padding[2];
    PrimeConstants primes[MAX_RNS_PRIMES];
    uint64_t firstLevelTwiddles[MAX_RNS_PRIMES][FIRST_LEVEL_TWIDDLE_COUNT];
    uint64_t firstLevelInvTwiddles[MAX_RNS_PRIMES][FIRST_LEVEL_TWIDDLE_COUNT];
};

struct NTTCacheParams {
    uint64_t Q;
    uint64_t mu;
    uint64_t N_inv;
    uint64_t N_inv_precon;
    uint32_t N;
    uint32_t log_N;
    uint32_t stage;
    uint32_t primeIdx;
    uint32_t batch;
    uint32_t prefetchStage;
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

NTT_DEVICE uint32_t padded_index(uint32_t idx) {
    return idx + (idx / BANK_WIDTH) * BANK_PADDING;
}

// =============================================================================
// Single Stage NTT with Hotset Caching
// =============================================================================

extern "C" __global__ void ntt_hotset_forward_stage(
    uint64_t* data, const uint64_t* twiddles,
    const ConstantCache cache, const NTTCacheParams params)
{
#ifdef __CUDA_ARCH__
    __shared__ uint64_t twiddles_shared[MAX_THREADGROUP_TWIDDLES + MAX_THREADGROUP_TWIDDLES / BANK_WIDTH];
    __shared__ uint64_t twiddles_prefetch[MAX_THREADGROUP_TWIDDLES + MAX_THREADGROUP_TWIDDLES / BANK_WIDTH];

    uint32_t thread_idx = threadIdx.x;
    uint32_t threadgroup_size = blockDim.x;
    uint32_t batch_idx = blockIdx.x;
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint32_t stage = params.stage;
    uint32_t primeIdx = params.primeIdx;
    uint32_t m = 1u << stage;
    uint32_t t = N >> (stage + 1);
    uint64_t* batch_data = data + batch_idx * N;

    bool use_constant_memory = (stage < 4 && m <= FIRST_LEVEL_TWIDDLE_COUNT);

    if (!use_constant_memory) {
        uint32_t twiddles_to_load = m;
        uint32_t loads_per_thread = (twiddles_to_load + threadgroup_size - 1) / threadgroup_size;
        for (uint32_t i = 0; i < loads_per_thread; ++i) {
            uint32_t tw_idx = thread_idx + i * threadgroup_size;
            if (tw_idx < m) {
                uint32_t padded = padded_index(tw_idx);
                twiddles_shared[padded] = twiddles[m + tw_idx];
            }
        }
        if (params.prefetchStage < params.log_N && params.prefetchStage > stage) {
            uint32_t next_m = 1u << params.prefetchStage;
            uint32_t prefetch_loads = (next_m + threadgroup_size - 1) / threadgroup_size;
            for (uint32_t i = 0; i < prefetch_loads; ++i) {
                uint32_t tw_idx = thread_idx + i * threadgroup_size;
                if (tw_idx < next_m && tw_idx < MAX_THREADGROUP_TWIDDLES) {
                    uint32_t padded = padded_index(tw_idx);
                    twiddles_prefetch[padded] = twiddles[next_m + tw_idx];
                }
            }
        }
        __syncthreads();
    }

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

        uint64_t tw;
        if (use_constant_memory) {
            tw = cache.firstLevelTwiddles[primeIdx][group];
        } else {
            tw = twiddles_shared[padded_index(group)];
        }

        uint64_t hi_tw = barrett_mul(hi, tw, Q, mu);
        batch_data[idx_lo] = mod_add(lo, hi_tw, Q);
        batch_data[idx_hi] = mod_sub(lo, hi_tw, Q);
    }
#endif
}

// =============================================================================
// Inverse NTT Stage with Hotset
// =============================================================================

extern "C" __global__ void ntt_hotset_inverse_stage(
    uint64_t* data, const uint64_t* twiddles,
    const ConstantCache cache, const NTTCacheParams params)
{
#ifdef __CUDA_ARCH__
    __shared__ uint64_t twiddles_shared[MAX_THREADGROUP_TWIDDLES + MAX_THREADGROUP_TWIDDLES / BANK_WIDTH];

    uint32_t thread_idx = threadIdx.x;
    uint32_t threadgroup_size = blockDim.x;
    uint32_t batch_idx = blockIdx.x;
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint32_t stage = params.stage;
    uint32_t primeIdx = params.primeIdx;
    uint32_t m = N >> (stage + 1);
    uint32_t t = 1u << stage;
    uint64_t* batch_data = data + batch_idx * N;

    bool use_constant_memory = (stage >= params.log_N - 4 && m <= FIRST_LEVEL_TWIDDLE_COUNT);

    if (!use_constant_memory) {
        uint32_t twiddles_to_load = m;
        uint32_t loads_per_thread = (twiddles_to_load + threadgroup_size - 1) / threadgroup_size;
        for (uint32_t i = 0; i < loads_per_thread; ++i) {
            uint32_t tw_idx = thread_idx + i * threadgroup_size;
            if (tw_idx < m) {
                twiddles_shared[padded_index(tw_idx)] = twiddles[m + tw_idx];
            }
        }
        __syncthreads();
    }

    uint32_t bpt = (N / 2 + threadgroup_size - 1) / threadgroup_size;
    for (uint32_t b = 0; b < bpt; ++b) {
        uint32_t butterfly_idx = thread_idx + b * threadgroup_size;
        if (butterfly_idx >= N / 2) break;
        uint32_t group = butterfly_idx / t;
        uint32_t elem = butterfly_idx % t;
        uint32_t idx_lo = (group << (stage + 1)) + elem;
        uint32_t idx_hi = idx_lo + t;
        uint64_t lo = batch_data[idx_lo];
        uint64_t hi = batch_data[idx_hi];

        uint64_t tw;
        if (use_constant_memory) {
            tw = cache.firstLevelInvTwiddles[primeIdx][group];
        } else {
            tw = twiddles_shared[padded_index(group)];
        }

        uint64_t sum = mod_add(lo, hi, Q);
        uint64_t diff = mod_sub(lo, hi, Q);
        batch_data[idx_lo] = sum;
        batch_data[idx_hi] = barrett_mul(diff, tw, Q, mu);
    }
#endif
}

// =============================================================================
// Multi-Stage Fused NTT with Full Hotset
// =============================================================================

extern "C" __global__ void ntt_hotset_fused(
    uint64_t* data, const uint64_t* twiddles_flat,
    const ConstantCache cache, const NTTCacheParams params)
{
#ifdef __CUDA_ARCH__
    __shared__ uint64_t twiddles_shared[MAX_THREADGROUP_TWIDDLES];

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
        if (tw_idx < total_twiddles)
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

// =============================================================================
// N^(-1) Scaling
// =============================================================================

extern "C" __global__ void ntt_hotset_scale_ninv(uint64_t* data, const NTTCacheParams params)
{
#ifdef __CUDA_ARCH__
    uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = params.N * params.batch;
    if (global_idx >= total) return;
    data[global_idx] = barrett_mul(data[global_idx], params.N_inv, params.Q, params.mu);
#endif
}
