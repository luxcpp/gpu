// =============================================================================
// Batched External Product CUDA Kernel - Lux FHE GPU Acceleration
// =============================================================================
// CUDA port of external_product_batch.metal -- byte-identical output.
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#ifdef __CUDA_ARCH__
#define EP_DEVICE __device__ __forceinline__
#else
#define EP_DEVICE inline
#define __global__
#define __shared__
static inline uint64_t __umul64hi(uint64_t a, uint64_t b) {
    __uint128_t r = (__uint128_t)a * b; return (uint64_t)(r >> 64);
}
static inline void __syncthreads() {}
#endif

struct BatchedExtProdParams {
    uint64_t Q;
    uint64_t barrett_mu;
    uint64_t N_inv;
    uint64_t N_inv_precon;
    uint32_t N;
    uint32_t log_N;
    uint32_t L;
    uint32_t base_log;
    uint64_t base_mask;
    uint32_t batch_size;
};

EP_DEVICE uint64_t barrett_reduce(uint64_t x, uint64_t Q, uint64_t mu) {
    uint64_t q_approx = __umul64hi(x, mu);
    uint64_t result = x - q_approx * Q;
    return (result >= Q) ? result - Q : result;
}

EP_DEVICE uint64_t barrett_mul(uint64_t a, uint64_t b, uint64_t Q, uint64_t mu) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    if (hi == 0) return barrett_reduce(lo, Q, mu);
    uint64_t two32 = (uint64_t)1 << 32;
    uint64_t two32_mod = two32 % Q;
    uint64_t two64_mod = barrett_reduce(two32_mod * two32_mod, Q, mu);
    uint64_t hi_contrib = barrett_reduce(hi * two64_mod, Q, mu);
    uint64_t lo_mod = barrett_reduce(lo, Q, mu);
    return barrett_reduce(lo_mod + hi_contrib, Q, mu);
}

EP_DEVICE uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    return (sum >= Q) ? sum - Q : sum;
}

EP_DEVICE uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    return (a >= b) ? a - b : a + Q - b;
}

EP_DEVICE void ct_butterfly(uint64_t* data, uint32_t idx_lo, uint32_t idx_hi,
                             uint64_t omega, uint64_t Q, uint64_t mu) {
    uint64_t lo = data[idx_lo], hi = data[idx_hi];
    uint64_t hi_tw = barrett_mul(hi, omega, Q, mu);
    data[idx_lo] = mod_add(lo, hi_tw, Q);
    data[idx_hi] = mod_sub(lo, hi_tw, Q);
}

EP_DEVICE void gs_butterfly(uint64_t* data, uint32_t idx_lo, uint32_t idx_hi,
                             uint64_t omega, uint64_t Q, uint64_t mu) {
    uint64_t lo = data[idx_lo], hi = data[idx_hi];
    uint64_t sum = mod_add(lo, hi, Q);
    uint64_t diff = mod_sub(lo, hi, Q);
    data[idx_lo] = sum;
    data[idx_hi] = barrett_mul(diff, omega, Q, mu);
}

// =============================================================================
// MAIN BATCHED EXTERNAL PRODUCT KERNEL
// =============================================================================

extern "C" __global__ void external_product_batched(
    uint64_t* out_c0, uint64_t* out_c1,
    const uint64_t* rlwe_c0, const uint64_t* rlwe_c1,
    const uint64_t* rgsw,
    const BatchedExtProdParams p)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared[];

    uint32_t batch_idx = blockIdx.x;
    if (batch_idx >= p.batch_size) return;

    uint32_t local_id = threadIdx.x;
    uint32_t threads = blockDim.x;
    uint32_t N = p.N, L = p.L;
    uint64_t Q = p.Q, mu = p.barrett_mu;
    uint32_t base_log = p.base_log;
    uint64_t base_mask = p.base_mask;

    uint64_t* decomp_c0 = shared;
    uint64_t* decomp_c1 = shared + L * N;
    uint64_t* acc_c0 = shared + 2 * L * N;
    uint64_t* acc_c1 = shared + 2 * L * N + N;

    const uint64_t* c0 = rlwe_c0 + batch_idx * N;
    const uint64_t* c1 = rlwe_c1 + batch_idx * N;
    uint64_t* o_c0 = out_c0 + batch_idx * N;
    uint64_t* o_c1 = out_c1 + batch_idx * N;

    // Stage 1: Gadget decomposition
    for (uint32_t i = local_id; i < N; i += threads) {
        uint64_t val_c0 = c0[i], val_c1 = c1[i];
        for (uint32_t l = 0; l < L; ++l) {
            decomp_c0[l * N + i] = (val_c0 >> (l * base_log)) & base_mask;
            decomp_c1[l * N + i] = (val_c1 >> (l * base_log)) & base_mask;
        }
    }
    __syncthreads();

    // Stage 2: Initialize accumulators
    for (uint32_t i = local_id; i < N; i += threads) {
        acc_c0[i] = 0; acc_c1[i] = 0;
    }
    __syncthreads();

    // Stage 3: External product accumulation
    for (uint32_t l = 0; l < L; ++l) {
        const uint64_t* rgsw_c0_l_0 = rgsw + (0 * L * 2 * N) + (l * 2 * N) + (0 * N);
        const uint64_t* rgsw_c0_l_1 = rgsw + (0 * L * 2 * N) + (l * 2 * N) + (1 * N);
        for (uint32_t i = local_id; i < N; i += threads) {
            uint64_t d0 = decomp_c0[l * N + i];
            acc_c0[i] = mod_add(acc_c0[i], barrett_mul(d0, rgsw_c0_l_0[i], Q, mu), Q);
            acc_c1[i] = mod_add(acc_c1[i], barrett_mul(d0, rgsw_c0_l_1[i], Q, mu), Q);
        }
        __syncthreads();

        const uint64_t* rgsw_c1_l_0 = rgsw + (1 * L * 2 * N) + (l * 2 * N) + (0 * N);
        const uint64_t* rgsw_c1_l_1 = rgsw + (1 * L * 2 * N) + (l * 2 * N) + (1 * N);
        for (uint32_t i = local_id; i < N; i += threads) {
            uint64_t d1 = decomp_c1[l * N + i];
            acc_c0[i] = mod_add(acc_c0[i], barrett_mul(d1, rgsw_c1_l_0[i], Q, mu), Q);
            acc_c1[i] = mod_add(acc_c1[i], barrett_mul(d1, rgsw_c1_l_1[i], Q, mu), Q);
        }
        __syncthreads();
    }

    // Stage 4: Write output
    for (uint32_t i = local_id; i < N; i += threads) {
        o_c0[i] = acc_c0[i];
        o_c1[i] = acc_c1[i];
    }
#endif
}

// =============================================================================
// ACCUMULATING VARIANT
// =============================================================================

extern "C" __global__ void external_product_batched_accumulate(
    uint64_t* acc_c0, uint64_t* acc_c1,
    const uint64_t* rlwe_c0, const uint64_t* rlwe_c1,
    const uint64_t* rgsw,
    const BatchedExtProdParams p)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared[];

    uint32_t batch_idx = blockIdx.x;
    if (batch_idx >= p.batch_size) return;

    uint32_t local_id = threadIdx.x, threads = blockDim.x;
    uint32_t N = p.N, L = p.L;
    uint64_t Q = p.Q, mu = p.barrett_mu;
    uint32_t base_log = p.base_log;
    uint64_t base_mask = p.base_mask;

    uint64_t* decomp_c0 = shared;
    uint64_t* decomp_c1 = shared + L * N;

    const uint64_t* c0 = rlwe_c0 + batch_idx * N;
    const uint64_t* c1 = rlwe_c1 + batch_idx * N;
    uint64_t* a_c0 = acc_c0 + batch_idx * N;
    uint64_t* a_c1 = acc_c1 + batch_idx * N;

    for (uint32_t i = local_id; i < N; i += threads) {
        uint64_t val_c0 = c0[i], val_c1 = c1[i];
        for (uint32_t l = 0; l < L; ++l) {
            decomp_c0[l * N + i] = (val_c0 >> (l * base_log)) & base_mask;
            decomp_c1[l * N + i] = (val_c1 >> (l * base_log)) & base_mask;
        }
    }
    __syncthreads();

    for (uint32_t l = 0; l < L; ++l) {
        const uint64_t* rgsw_c0_l_0 = rgsw + (0 * L * 2 * N) + (l * 2 * N);
        const uint64_t* rgsw_c0_l_1 = rgsw + (0 * L * 2 * N) + (l * 2 * N) + N;
        const uint64_t* rgsw_c1_l_0 = rgsw + (1 * L * 2 * N) + (l * 2 * N);
        const uint64_t* rgsw_c1_l_1 = rgsw + (1 * L * 2 * N) + (l * 2 * N) + N;

        for (uint32_t i = local_id; i < N; i += threads) {
            uint64_t d0 = decomp_c0[l * N + i];
            uint64_t d1 = decomp_c1[l * N + i];
            uint64_t prod_00 = barrett_mul(d0, rgsw_c0_l_0[i], Q, mu);
            uint64_t prod_01 = barrett_mul(d0, rgsw_c0_l_1[i], Q, mu);
            uint64_t prod_10 = barrett_mul(d1, rgsw_c1_l_0[i], Q, mu);
            uint64_t prod_11 = barrett_mul(d1, rgsw_c1_l_1[i], Q, mu);
            a_c0[i] = mod_add(a_c0[i], mod_add(prod_00, prod_10, Q), Q);
            a_c1[i] = mod_add(a_c1[i], mod_add(prod_01, prod_11, Q), Q);
        }
        __syncthreads();
    }
#endif
}

// =============================================================================
// NTT-DOMAIN BATCHED EXTERNAL PRODUCT
// =============================================================================

extern "C" __global__ void external_product_batched_ntt(
    uint64_t* out_c0, uint64_t* out_c1,
    const uint64_t* rlwe_c0, const uint64_t* rlwe_c1,
    const uint64_t* rgsw_ntt,
    const uint64_t* fwd_twiddles, const uint64_t* fwd_precon,
    const BatchedExtProdParams p)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared[];

    uint32_t batch_idx = blockIdx.x;
    if (batch_idx >= p.batch_size) return;

    uint32_t local_id = threadIdx.x, threads = blockDim.x;
    uint32_t N = p.N, log_N = p.log_N, L = p.L;
    uint64_t Q = p.Q, mu = p.barrett_mu;
    uint32_t base_log = p.base_log;
    uint64_t base_mask = p.base_mask;

    uint64_t* work = shared;
    uint64_t* tw_fwd = shared + N;
    uint64_t* pre_fwd = shared + 2 * N;
    uint64_t* s_acc_c0 = shared + 3 * N;
    uint64_t* s_acc_c1 = shared + 4 * N;

    const uint64_t* c0 = rlwe_c0 + batch_idx * N;
    const uint64_t* c1 = rlwe_c1 + batch_idx * N;
    uint64_t* o_c0 = out_c0 + batch_idx * N;
    uint64_t* o_c1 = out_c1 + batch_idx * N;

    for (uint32_t i = local_id; i < N; i += threads) {
        tw_fwd[i] = fwd_twiddles[i];
        pre_fwd[i] = fwd_precon[i];
        s_acc_c0[i] = 0;
        s_acc_c1[i] = 0;
    }
    __syncthreads();

    for (uint32_t in_c = 0; in_c < 2; ++in_c) {
        const uint64_t* rlwe_comp = (in_c == 0) ? c0 : c1;
        for (uint32_t l = 0; l < L; ++l) {
            for (uint32_t i = local_id; i < N; i += threads) {
                work[i] = (rlwe_comp[i] >> (l * base_log)) & base_mask;
            }
            __syncthreads();

            // Forward NTT
            for (uint32_t stage = 0; stage < log_N; ++stage) {
                uint32_t m = 1u << stage, t = N >> (stage + 1);
                for (uint32_t bf = local_id; bf < N / 2; bf += threads) {
                    uint32_t ii = bf / t, jj = bf % t;
                    uint32_t idx_lo = (ii << (log_N - stage)) + jj;
                    uint32_t idx_hi = idx_lo + t;
                    uint32_t tw_idx = m + ii;
                    uint64_t lo = work[idx_lo], hi = work[idx_hi];
                    uint64_t omega = tw_fwd[tw_idx], precon = pre_fwd[tw_idx];
                    uint64_t q_approx = __umul64hi(hi, precon);
                    uint64_t hi_tw = hi * omega - q_approx * Q;
                    if (hi_tw >= Q) hi_tw -= Q;
                    work[idx_lo] = mod_add(lo, hi_tw, Q);
                    work[idx_hi] = mod_sub(lo, hi_tw, Q);
                }
                __syncthreads();
            }

            const uint64_t* rgsw_l_0 = rgsw_ntt + (in_c * L * 2 * N) + (l * 2 * N);
            const uint64_t* rgsw_l_1 = rgsw_ntt + (in_c * L * 2 * N) + (l * 2 * N) + N;
            for (uint32_t i = local_id; i < N; i += threads) {
                uint64_t digit_ntt = work[i];
                s_acc_c0[i] = mod_add(s_acc_c0[i], barrett_mul(digit_ntt, rgsw_l_0[i], Q, mu), Q);
                s_acc_c1[i] = mod_add(s_acc_c1[i], barrett_mul(digit_ntt, rgsw_l_1[i], Q, mu), Q);
            }
            __syncthreads();
        }
    }

    for (uint32_t i = local_id; i < N; i += threads) {
        o_c0[i] = s_acc_c0[i];
        o_c1[i] = s_acc_c1[i];
    }
#endif
}

// =============================================================================
// CMUX GATE
// =============================================================================

extern "C" __global__ void cmux_batched(
    uint64_t* out_c0, uint64_t* out_c1,
    const uint64_t* d0_c0, const uint64_t* d0_c1,
    const uint64_t* d1_c0, const uint64_t* d1_c1,
    const uint64_t* rgsw_bit,
    const BatchedExtProdParams p)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared[];

    uint32_t batch_idx = blockIdx.x;
    if (batch_idx >= p.batch_size) return;

    uint32_t local_id = threadIdx.x, threads = blockDim.x;
    uint32_t N = p.N, L = p.L;
    uint64_t Q = p.Q, mu = p.barrett_mu;
    uint32_t base_log = p.base_log;
    uint64_t base_mask = p.base_mask;

    uint64_t* decomp_c0 = shared;
    uint64_t* decomp_c1 = shared + L * N;
    uint64_t* acc_c0 = shared + 2 * L * N;
    uint64_t* acc_c1 = shared + 2 * L * N + N;

    const uint64_t* d0_0 = d0_c0 + batch_idx * N;
    const uint64_t* d0_1 = d0_c1 + batch_idx * N;
    const uint64_t* d1_0 = d1_c0 + batch_idx * N;
    const uint64_t* d1_1 = d1_c1 + batch_idx * N;
    uint64_t* o_c0 = out_c0 + batch_idx * N;
    uint64_t* o_c1 = out_c1 + batch_idx * N;

    for (uint32_t i = local_id; i < N; i += threads) {
        uint64_t diff_c0 = mod_sub(d1_0[i], d0_0[i], Q);
        uint64_t diff_c1 = mod_sub(d1_1[i], d0_1[i], Q);
        for (uint32_t l = 0; l < L; ++l) {
            decomp_c0[l * N + i] = (diff_c0 >> (l * base_log)) & base_mask;
            decomp_c1[l * N + i] = (diff_c1 >> (l * base_log)) & base_mask;
        }
        acc_c0[i] = d0_0[i];
        acc_c1[i] = d0_1[i];
    }
    __syncthreads();

    for (uint32_t l = 0; l < L; ++l) {
        const uint64_t* rgsw_c0_l_0 = rgsw_bit + (0 * L * 2 * N) + (l * 2 * N);
        const uint64_t* rgsw_c0_l_1 = rgsw_bit + (0 * L * 2 * N) + (l * 2 * N) + N;
        const uint64_t* rgsw_c1_l_0 = rgsw_bit + (1 * L * 2 * N) + (l * 2 * N);
        const uint64_t* rgsw_c1_l_1 = rgsw_bit + (1 * L * 2 * N) + (l * 2 * N) + N;

        for (uint32_t i = local_id; i < N; i += threads) {
            uint64_t d0 = decomp_c0[l * N + i];
            uint64_t d1 = decomp_c1[l * N + i];
            uint64_t prod_00 = barrett_mul(d0, rgsw_c0_l_0[i], Q, mu);
            uint64_t prod_01 = barrett_mul(d0, rgsw_c0_l_1[i], Q, mu);
            uint64_t prod_10 = barrett_mul(d1, rgsw_c1_l_0[i], Q, mu);
            uint64_t prod_11 = barrett_mul(d1, rgsw_c1_l_1[i], Q, mu);
            acc_c0[i] = mod_add(acc_c0[i], mod_add(prod_00, prod_10, Q), Q);
            acc_c1[i] = mod_add(acc_c1[i], mod_add(prod_01, prod_11, Q), Q);
        }
        __syncthreads();
    }

    for (uint32_t i = local_id; i < N; i += threads) {
        o_c0[i] = acc_c0[i];
        o_c1[i] = acc_c1[i];
    }
#endif
}

// =============================================================================
// MULTI-RGSW BATCHED EXTERNAL PRODUCT
// =============================================================================

extern "C" __global__ void external_product_multi_batched(
    uint64_t* out_c0, uint64_t* out_c1,
    const uint64_t* rlwe_c0, const uint64_t* rlwe_c1,
    const uint64_t* rgsw_batch,
    const BatchedExtProdParams p)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared[];

    uint32_t batch_idx = blockIdx.x;
    if (batch_idx >= p.batch_size) return;

    uint32_t local_id = threadIdx.x, threads = blockDim.x;
    uint32_t N = p.N, L = p.L;
    uint64_t Q = p.Q, mu = p.barrett_mu;
    uint32_t base_log = p.base_log;
    uint64_t base_mask = p.base_mask;

    uint64_t* decomp_c0 = shared;
    uint64_t* decomp_c1 = shared + L * N;
    uint64_t* acc_c0 = shared + 2 * L * N;
    uint64_t* acc_c1 = shared + 2 * L * N + N;

    uint32_t rgsw_stride = 2 * L * 2 * N;
    const uint64_t* rgsw = rgsw_batch + batch_idx * rgsw_stride;
    const uint64_t* c0 = rlwe_c0 + batch_idx * N;
    const uint64_t* c1 = rlwe_c1 + batch_idx * N;
    uint64_t* o_c0 = out_c0 + batch_idx * N;
    uint64_t* o_c1 = out_c1 + batch_idx * N;

    for (uint32_t i = local_id; i < N; i += threads) {
        uint64_t val_c0 = c0[i], val_c1 = c1[i];
        for (uint32_t l = 0; l < L; ++l) {
            decomp_c0[l * N + i] = (val_c0 >> (l * base_log)) & base_mask;
            decomp_c1[l * N + i] = (val_c1 >> (l * base_log)) & base_mask;
        }
        acc_c0[i] = 0; acc_c1[i] = 0;
    }
    __syncthreads();

    for (uint32_t l = 0; l < L; ++l) {
        const uint64_t* rgsw_c0_l_0 = rgsw + (0 * L * 2 * N) + (l * 2 * N);
        const uint64_t* rgsw_c0_l_1 = rgsw + (0 * L * 2 * N) + (l * 2 * N) + N;
        const uint64_t* rgsw_c1_l_0 = rgsw + (1 * L * 2 * N) + (l * 2 * N);
        const uint64_t* rgsw_c1_l_1 = rgsw + (1 * L * 2 * N) + (l * 2 * N) + N;
        for (uint32_t i = local_id; i < N; i += threads) {
            uint64_t d0 = decomp_c0[l * N + i], d1 = decomp_c1[l * N + i];
            acc_c0[i] = mod_add(acc_c0[i], mod_add(barrett_mul(d0, rgsw_c0_l_0[i], Q, mu),
                                                      barrett_mul(d1, rgsw_c1_l_0[i], Q, mu), Q), Q);
            acc_c1[i] = mod_add(acc_c1[i], mod_add(barrett_mul(d0, rgsw_c0_l_1[i], Q, mu),
                                                      barrett_mul(d1, rgsw_c1_l_1[i], Q, mu), Q), Q);
        }
        __syncthreads();
    }

    for (uint32_t i = local_id; i < N; i += threads) {
        o_c0[i] = acc_c0[i];
        o_c1[i] = acc_c1[i];
    }
#endif
}
