// =============================================================================
// Fused External Product CUDA Kernel - Lux FHE GPU Acceleration
// =============================================================================
// CUDA port of fused_external_product.metal -- byte-identical output.
//
// Fusion Pipeline: Decompose -> NTT -> Multiply -> INTT -> Accumulate
// All in shared memory, no intermediate global writes.
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#ifdef __CUDA_ARCH__
#define FEP_DEVICE __device__ __forceinline__
#else
#define FEP_DEVICE inline
#define __global__
#define __shared__
static inline uint64_t __umul64hi(uint64_t a, uint64_t b) {
    __uint128_t r = (__uint128_t)a * b; return (uint64_t)(r >> 64);
}
static inline void __syncthreads() {}
#endif

struct FusedParams {
    uint64_t Q;
    uint64_t mu;
    uint64_t N_inv;
    uint64_t N_inv_precon;
    uint32_t N;
    uint32_t log_N;
    uint32_t L;
    uint32_t base_log;
    uint64_t base_mask;
    uint32_t batch_size;
};

FEP_DEVICE uint64_t barrett_mul(uint64_t a, uint64_t omega, uint64_t Q, uint64_t precon) {
    uint64_t q_approx = __umul64hi(a, precon);
    uint64_t product = a * omega;
    uint64_t result = product - q_approx * Q;
    return (result >= Q) ? result - Q : result;
}

FEP_DEVICE uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t Q) {
#ifdef __CUDA_ARCH__
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    if (hi == 0) return lo % Q;
    uint64_t two32_mod_q = ((uint64_t)1 << 32) % Q;
    uint64_t two64_mod_q = (two32_mod_q * two32_mod_q) % Q;
    return (lo % Q + (hi % Q) * two64_mod_q % Q) % Q;
#else
    __uint128_t prod = (__uint128_t)a * b;
    return (uint64_t)(prod % Q);
#endif
}

FEP_DEVICE uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    return (sum >= Q) ? sum - Q : sum;
}

FEP_DEVICE uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    return (a >= b) ? a - b : a + Q - b;
}

FEP_DEVICE void ct_butterfly(uint64_t* data, uint32_t idx_lo, uint32_t idx_hi,
                              uint64_t omega, uint64_t precon_omega, uint64_t Q) {
    uint64_t lo_val = data[idx_lo], hi_val = data[idx_hi];
    uint64_t omega_factor = barrett_mul(hi_val, omega, Q, precon_omega);
    data[idx_lo] = mod_add(lo_val, omega_factor, Q);
    data[idx_hi] = mod_sub(lo_val, omega_factor, Q);
}

FEP_DEVICE void gs_butterfly(uint64_t* data, uint32_t idx_lo, uint32_t idx_hi,
                              uint64_t omega, uint64_t precon_omega, uint64_t Q) {
    uint64_t lo_val = data[idx_lo], hi_val = data[idx_hi];
    uint64_t sum = mod_add(lo_val, hi_val, Q);
    uint64_t diff = mod_sub(lo_val, hi_val, Q);
    data[idx_lo] = sum;
    data[idx_hi] = barrett_mul(diff, omega, Q, precon_omega);
}

// =============================================================================
// FUSED EXTERNAL PRODUCT KERNEL
// =============================================================================

extern "C" __global__ void fused_external_product(
    uint64_t* result, const uint64_t* rlwe, const uint64_t* rgsw,
    const uint64_t* twiddles, const uint64_t* precon_twiddles,
    const uint64_t* inv_twiddles, const uint64_t* inv_precon,
    const FusedParams params)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared[];

    uint32_t batch_idx = blockIdx.y;
    uint32_t local_idx = threadIdx.x;
    uint32_t N = params.N, log_N = params.log_N, L = params.L;
    uint64_t Q = params.Q;
    uint64_t base_mask = params.base_mask;
    uint32_t base_log = params.base_log;

    if (batch_idx >= params.batch_size) return;

    uint64_t* work_buf = shared;
    uint64_t* fwd_tw = shared + N;
    uint64_t* fwd_precon = shared + 2 * N;
    uint64_t* inv_tw = shared + 3 * N;
    uint64_t* inv_pre = shared + 4 * N;

    // Cooperative twiddle prefetch
    for (uint32_t i = local_idx; i < N; i += blockDim.x) {
        fwd_tw[i] = twiddles[i];
        fwd_precon[i] = precon_twiddles[i];
        inv_tw[i] = inv_twiddles[i];
        inv_pre[i] = inv_precon[i];
    }
    __syncthreads();

    const uint64_t* rlwe_batch = rlwe + batch_idx * 2 * N;
    const uint64_t* rgsw_batch = rgsw + batch_idx * 2 * L * 2 * N;
    uint64_t* result_batch = result + batch_idx * 2 * N;

    for (uint32_t in_c = 0; in_c < 2; ++in_c) {
        for (uint32_t l = 0; l < L; ++l) {
            // Load and decompose
            if (l == 0 && in_c == 0) {
                for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                    uint64_t val = rlwe_batch[in_c * N + i];
                    work_buf[i] = (val >> (l * base_log)) & base_mask;
                }
            } else {
                for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                    uint64_t val = rlwe_batch[in_c * N + i];
                    work_buf[i] = (val >> (l * base_log)) & base_mask;
                }
            }
            __syncthreads();

            // Forward NTT
            for (uint32_t stage = 0; stage < log_N; ++stage) {
                uint32_t m = 1u << stage, t = N >> (stage + 1);
                for (uint32_t bf = local_idx; bf < N / 2; bf += blockDim.x) {
                    uint32_t i = bf / t, j = bf % t;
                    uint32_t idx_lo = (i << (log_N - stage)) + j;
                    uint32_t idx_hi = idx_lo + t;
                    uint32_t tw_idx = m + i;
                    ct_butterfly(work_buf, idx_lo, idx_hi, fwd_tw[tw_idx], fwd_precon[tw_idx], Q);
                }
                __syncthreads();
            }

            // Multiply, INTT, accumulate for each output component
            for (uint32_t out_c = 0; out_c < 2; ++out_c) {
                // Save NTT'd digit and pointwise multiply with RGSW
                for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                    uint64_t digit_ntt = work_buf[i];
                    uint32_t rgsw_idx = in_c * L * 2 * N + l * 2 * N + out_c * N + i;
                    work_buf[i] = mod_mul(digit_ntt, rgsw_batch[rgsw_idx], Q);
                }
                __syncthreads();

                // Inverse NTT
                for (uint32_t stage = 0; stage < log_N; ++stage) {
                    uint32_t m = N >> (stage + 1), t = 1u << stage;
                    for (uint32_t bf = local_idx; bf < N / 2; bf += blockDim.x) {
                        uint32_t i = bf / t, j = bf % t;
                        uint32_t idx_lo = (i << (stage + 1)) + j;
                        uint32_t idx_hi = idx_lo + t;
                        uint32_t tw_idx = m + i;
                        gs_butterfly(work_buf, idx_lo, idx_hi, inv_tw[tw_idx], inv_pre[tw_idx], Q);
                    }
                    __syncthreads();
                }

                // Scale by N^{-1}
                for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                    work_buf[i] = barrett_mul(work_buf[i], params.N_inv, Q, params.N_inv_precon);
                }
                __syncthreads();

                // Accumulate
                if (in_c == 0 && l == 0) {
                    for (uint32_t i = local_idx; i < N; i += blockDim.x)
                        result_batch[out_c * N + i] = work_buf[i];
                } else {
                    for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                        uint64_t acc = result_batch[out_c * N + i];
                        result_batch[out_c * N + i] = mod_add(acc, work_buf[i], Q);
                    }
                }
                __syncthreads();

                // Reload digit for next out_c (need fresh NTT)
                if (out_c == 0) {
                    for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                        uint64_t val = rlwe_batch[in_c * N + i];
                        work_buf[i] = (val >> (l * base_log)) & base_mask;
                    }
                    __syncthreads();
                    for (uint32_t stage = 0; stage < log_N; ++stage) {
                        uint32_t m = 1u << stage, t = N >> (stage + 1);
                        for (uint32_t bf = local_idx; bf < N / 2; bf += blockDim.x) {
                            uint32_t ii = bf / t, jj = bf % t;
                            uint32_t idx_lo = (ii << (log_N - stage)) + jj;
                            uint32_t idx_hi = idx_lo + t;
                            uint32_t tw_idx = m + ii;
                            ct_butterfly(work_buf, idx_lo, idx_hi, fwd_tw[tw_idx], fwd_precon[tw_idx], Q);
                        }
                        __syncthreads();
                    }
                }
            }
        }
    }
#endif
}

// =============================================================================
// ACCUMULATING VARIANT
// =============================================================================

extern "C" __global__ void fused_external_product_accumulate(
    uint64_t* accumulator, const uint64_t* rlwe, const uint64_t* rgsw,
    const uint64_t* twiddles, const uint64_t* precon_twiddles,
    const uint64_t* inv_twiddles, const uint64_t* inv_precon,
    const FusedParams params)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared[];

    uint32_t batch_idx = blockIdx.y;
    uint32_t local_idx = threadIdx.x;
    uint32_t N = params.N, log_N = params.log_N, L = params.L;
    uint64_t Q = params.Q;
    uint64_t base_mask = params.base_mask;
    uint32_t base_log = params.base_log;

    if (batch_idx >= params.batch_size) return;

    uint64_t* work_buf = shared;
    uint64_t* fwd_tw = shared + N;
    uint64_t* fwd_precon = shared + 2 * N;
    uint64_t* inv_tw = shared + 3 * N;
    uint64_t* inv_pre = shared + 4 * N;

    for (uint32_t i = local_idx; i < N; i += blockDim.x) {
        fwd_tw[i] = twiddles[i];
        fwd_precon[i] = precon_twiddles[i];
        inv_tw[i] = inv_twiddles[i];
        inv_pre[i] = inv_precon[i];
    }
    __syncthreads();

    const uint64_t* rlwe_batch = rlwe + batch_idx * 2 * N;
    const uint64_t* rgsw_batch = rgsw + batch_idx * 2 * L * 2 * N;
    uint64_t* acc_batch = accumulator + batch_idx * 2 * N;

    for (uint32_t in_c = 0; in_c < 2; ++in_c) {
        for (uint32_t l = 0; l < L; ++l) {
            for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                uint64_t val = rlwe_batch[in_c * N + i];
                work_buf[i] = (val >> (l * base_log)) & base_mask;
            }
            __syncthreads();

            for (uint32_t stage = 0; stage < log_N; ++stage) {
                uint32_t m = 1u << stage, t = N >> (stage + 1);
                for (uint32_t bf = local_idx; bf < N / 2; bf += blockDim.x) {
                    uint32_t ii = bf / t, jj = bf % t;
                    uint32_t idx_lo = (ii << (log_N - stage)) + jj;
                    uint32_t idx_hi = idx_lo + t;
                    uint32_t tw_idx = m + ii;
                    ct_butterfly(work_buf, idx_lo, idx_hi, fwd_tw[tw_idx], fwd_precon[tw_idx], Q);
                }
                __syncthreads();
            }

            for (uint32_t out_c = 0; out_c < 2; ++out_c) {
                for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                    uint32_t rgsw_idx = in_c * L * 2 * N + l * 2 * N + out_c * N + i;
                    work_buf[i] = mod_mul(work_buf[i], rgsw_batch[rgsw_idx], Q);
                }
                __syncthreads();

                for (uint32_t stage = 0; stage < log_N; ++stage) {
                    uint32_t m = N >> (stage + 1), t = 1u << stage;
                    for (uint32_t bf = local_idx; bf < N / 2; bf += blockDim.x) {
                        uint32_t ii = bf / t, jj = bf % t;
                        uint32_t idx_lo = (ii << (stage + 1)) + jj;
                        uint32_t idx_hi = idx_lo + t;
                        uint32_t tw_idx = m + ii;
                        gs_butterfly(work_buf, idx_lo, idx_hi, inv_tw[tw_idx], inv_pre[tw_idx], Q);
                    }
                    __syncthreads();
                }

                for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                    uint64_t val = barrett_mul(work_buf[i], params.N_inv, Q, params.N_inv_precon);
                    acc_batch[out_c * N + i] = mod_add(acc_batch[out_c * N + i], val, Q);
                }
                __syncthreads();

                // Reload for next out_c
                if (out_c == 0) {
                    for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                        uint64_t val = rlwe_batch[in_c * N + i];
                        work_buf[i] = (val >> (l * base_log)) & base_mask;
                    }
                    __syncthreads();
                    for (uint32_t stage = 0; stage < log_N; ++stage) {
                        uint32_t m = 1u << stage, t = N >> (stage + 1);
                        for (uint32_t bf = local_idx; bf < N / 2; bf += blockDim.x) {
                            uint32_t ii = bf / t, jj = bf % t;
                            uint32_t idx_lo = (ii << (log_N - stage)) + jj;
                            uint32_t idx_hi = idx_lo + t;
                            uint32_t tw_idx = m + ii;
                            ct_butterfly(work_buf, idx_lo, idx_hi, fwd_tw[tw_idx], fwd_precon[tw_idx], Q);
                        }
                        __syncthreads();
                    }
                }
            }
        }
    }
#endif
}
