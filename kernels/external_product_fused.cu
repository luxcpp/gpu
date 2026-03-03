// =============================================================================
// Fused External Product CUDA Kernel - Lux FHE GPU Acceleration
// =============================================================================
//
// CUDA port of external_product_fused.metal. Single GPU kernel that fuses the
// entire external product operation for FHE blind rotation.
//
// Key Innovation: Zero intermediate global memory writes
// Traditional approach (5 kernels) vs Fused approach (1 kernel).
//
// Copyright (C) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#ifdef __CUDA_ARCH__

// =============================================================================
// Kernel Parameters
// =============================================================================

struct FusedExternalProductParams {
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

// =============================================================================
// Barrett Modular Arithmetic
// =============================================================================

__device__ __forceinline__
uint64_t mulhi64(uint64_t a, uint64_t b) {
    return __umul64hi(a, b);
}

__device__ __forceinline__
uint64_t barrett_mul(uint64_t a, uint64_t omega,
                     uint64_t Q, uint64_t precon) {
    uint64_t q_approx = mulhi64(a, precon);
    uint64_t product = a * omega;
    uint64_t result = product - q_approx * Q;
    return (result >= Q) ? result - Q : result;
}

__device__ __forceinline__
uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t lo = a * b;
    uint64_t hi = mulhi64(a, b);

    if (hi == 0) {
        return lo % Q;
    }

    uint64_t two32 = uint64_t(1) << 32;
    uint64_t two32_mod = two32 % Q;
    uint64_t two64_mod = (two32_mod * two32_mod) % Q;

    uint64_t hi_contrib = (hi % Q) * two64_mod % Q;
    return (lo % Q + hi_contrib) % Q;
}

__device__ __forceinline__
uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    return (sum >= Q) ? sum - Q : sum;
}

__device__ __forceinline__
uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    return (a >= b) ? a - b : a + Q - b;
}

// =============================================================================
// NTT Butterfly Operations
// =============================================================================

__device__ __forceinline__
void ct_butterfly(uint64_t* data,
                  uint32_t idx_lo, uint32_t idx_hi,
                  uint64_t omega, uint64_t precon,
                  uint64_t Q) {
    uint64_t lo = data[idx_lo];
    uint64_t hi = data[idx_hi];

    uint64_t hi_omega = barrett_mul(hi, omega, Q, precon);

    data[idx_lo] = mod_add(lo, hi_omega, Q);
    data[idx_hi] = mod_sub(lo, hi_omega, Q);
}

__device__ __forceinline__
void gs_butterfly(uint64_t* data,
                  uint32_t idx_lo, uint32_t idx_hi,
                  uint64_t omega, uint64_t precon,
                  uint64_t Q) {
    uint64_t lo = data[idx_lo];
    uint64_t hi = data[idx_hi];

    uint64_t sum = mod_add(lo, hi, Q);
    uint64_t diff = mod_sub(lo, hi, Q);
    uint64_t diff_omega = barrett_mul(diff, omega, Q, precon);

    data[idx_lo] = sum;
    data[idx_hi] = diff_omega;
}

// =============================================================================
// MAIN FUSED EXTERNAL PRODUCT KERNEL
// =============================================================================
//
// Thread Organization:
//   Grid:        dim3(1, batch_size)
//   Block:       dim3(threads_x, 1) where threads_x = min(N/2, 256)
//   Shared mem:  5 * N * sizeof(uint64_t)

extern "C" __global__
void fused_external_product_v2(
    uint64_t* result,
    const uint64_t* rlwe,
    const uint64_t* rgsw,
    const uint64_t* fwd_twiddles,
    const uint64_t* fwd_precon,
    const uint64_t* inv_twiddles,
    const uint64_t* inv_precon,
    const FusedExternalProductParams p
) {
    extern __shared__ uint64_t shared[];

    uint32_t batch_idx = blockIdx.y;
    uint32_t local_idx = threadIdx.x;
    uint32_t N = p.N;
    uint32_t log_N = p.log_N;
    uint32_t L = p.L;
    uint64_t Q = p.Q;
    uint64_t base_mask = p.base_mask;
    uint32_t base_log = p.base_log;

    if (batch_idx >= p.batch_size) return;

    // Shared memory partitioning
    uint64_t* work    = shared;
    uint64_t* tw_fwd  = shared + N;
    uint64_t* pre_fwd = shared + 2 * N;
    uint64_t* tw_inv  = shared + 3 * N;
    uint64_t* pre_inv = shared + 4 * N;

    // Twiddle prefetch
    for (uint32_t i = local_idx; i < N; i += blockDim.x) {
        tw_fwd[i]  = fwd_twiddles[i];
        pre_fwd[i] = fwd_precon[i];
        tw_inv[i]  = inv_twiddles[i];
        pre_inv[i] = inv_precon[i];
    }
    __syncthreads();

    uint64_t digits[4];

    const uint64_t* rlwe_b   = rlwe + batch_idx * 2 * N;
    const uint64_t* rgsw_b   = rgsw + batch_idx * 2 * L * 2 * N;
    uint64_t*       result_b = result + batch_idx * 2 * N;

    // Main computation loop
    for (uint32_t in_c = 0; in_c < 2; ++in_c) {
        for (uint32_t i = local_idx; i < N; i += blockDim.x) {
            uint64_t val = rlwe_b[in_c * N + i];
            for (uint32_t l = 0; l < L && l < 4; ++l) {
                digits[l] = (val >> (l * base_log)) & base_mask;
            }
            work[i] = digits[0];
        }
        __syncthreads();

        for (uint32_t l = 0; l < L; ++l) {
            // Reload digit if not first level
            if (l > 0) {
                for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                    uint64_t val = rlwe_b[in_c * N + i];
                    work[i] = (val >> (l * base_log)) & base_mask;
                }
                __syncthreads();
            }

            // Forward NTT (Cooley-Tukey)
            for (uint32_t stage = 0; stage < log_N; ++stage) {
                uint32_t m = 1u << stage;
                uint32_t t = N >> (stage + 1);

                for (uint32_t bf = local_idx; bf < N / 2; bf += blockDim.x) {
                    uint32_t i = bf / t;
                    uint32_t j = bf % t;
                    uint32_t idx_lo = (i << (log_N - stage)) + j;
                    uint32_t idx_hi = idx_lo + t;
                    uint32_t tw_idx = m + i;

                    ct_butterfly(work, idx_lo, idx_hi,
                                tw_fwd[tw_idx], pre_fwd[tw_idx], Q);
                }
                __syncthreads();
            }

            // Pointwise multiply & INTT for each output component
            for (uint32_t out_c = 0; out_c < 2; ++out_c) {
                for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                    uint32_t rgsw_idx = in_c * L * 2 * N + l * 2 * N + out_c * N + i;
                    uint64_t rgsw_val = rgsw_b[rgsw_idx];
                    work[i] = mod_mul(work[i], rgsw_val, Q);
                }
                __syncthreads();

                // Inverse NTT (Gentleman-Sande)
                for (uint32_t stage = 0; stage < log_N; ++stage) {
                    uint32_t m = N >> (stage + 1);
                    uint32_t t = 1u << stage;

                    for (uint32_t bf = local_idx; bf < N / 2; bf += blockDim.x) {
                        uint32_t i = bf / t;
                        uint32_t j = bf % t;
                        uint32_t idx_lo = (i << (stage + 1)) + j;
                        uint32_t idx_hi = idx_lo + t;
                        uint32_t tw_idx = m + i;

                        gs_butterfly(work, idx_lo, idx_hi,
                                    tw_inv[tw_idx], pre_inv[tw_idx], Q);
                    }
                    __syncthreads();
                }

                // Scale by N^{-1}
                for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                    work[i] = barrett_mul(work[i], p.N_inv, Q, p.N_inv_precon);
                }
                __syncthreads();

                // Accumulate to result
                if (in_c == 0 && l == 0) {
                    for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                        result_b[out_c * N + i] = work[i];
                    }
                } else {
                    for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                        uint64_t acc = result_b[out_c * N + i];
                        result_b[out_c * N + i] = mod_add(acc, work[i], Q);
                    }
                }
                __syncthreads();

                // Reload NTT result for next output component
                if (out_c == 0 && l < L) {
                    for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                        uint64_t val = rlwe_b[in_c * N + i];
                        work[i] = (val >> (l * base_log)) & base_mask;
                    }
                    __syncthreads();

                    for (uint32_t stage = 0; stage < log_N; ++stage) {
                        uint32_t m = 1u << stage;
                        uint32_t t = N >> (stage + 1);

                        for (uint32_t bf = local_idx; bf < N / 2; bf += blockDim.x) {
                            uint32_t ii = bf / t;
                            uint32_t jj = bf % t;
                            uint32_t idx_lo = (ii << (log_N - stage)) + jj;
                            uint32_t idx_hi = idx_lo + t;
                            uint32_t tw_idx = m + ii;

                            ct_butterfly(work, idx_lo, idx_hi,
                                        tw_fwd[tw_idx], pre_fwd[tw_idx], Q);
                        }
                        __syncthreads();
                    }
                }
            }
        }
    }
}

// =============================================================================
// ACCUMULATING VARIANT: result += ExternalProduct(rlwe, rgsw)
// =============================================================================

extern "C" __global__
void fused_external_product_accumulate_v2(
    uint64_t* accumulator,
    const uint64_t* rlwe,
    const uint64_t* rgsw,
    const uint64_t* fwd_twiddles,
    const uint64_t* fwd_precon,
    const uint64_t* inv_twiddles,
    const uint64_t* inv_precon,
    const FusedExternalProductParams p
) {
    extern __shared__ uint64_t shared[];

    uint32_t batch_idx = blockIdx.y;
    uint32_t local_idx = threadIdx.x;
    uint32_t N = p.N;
    uint32_t log_N = p.log_N;
    uint32_t L = p.L;
    uint64_t Q = p.Q;
    uint64_t base_mask = p.base_mask;
    uint32_t base_log = p.base_log;

    if (batch_idx >= p.batch_size) return;

    uint64_t* work    = shared;
    uint64_t* tw_fwd  = shared + N;
    uint64_t* pre_fwd = shared + 2 * N;
    uint64_t* tw_inv  = shared + 3 * N;
    uint64_t* pre_inv = shared + 4 * N;

    // Prefetch twiddles
    for (uint32_t i = local_idx; i < N; i += blockDim.x) {
        tw_fwd[i]  = fwd_twiddles[i];
        pre_fwd[i] = fwd_precon[i];
        tw_inv[i]  = inv_twiddles[i];
        pre_inv[i] = inv_precon[i];
    }
    __syncthreads();

    const uint64_t* rlwe_b = rlwe + batch_idx * 2 * N;
    const uint64_t* rgsw_b = rgsw + batch_idx * 2 * L * 2 * N;
    uint64_t*       acc_b  = accumulator + batch_idx * 2 * N;

    for (uint32_t in_c = 0; in_c < 2; ++in_c) {
        for (uint32_t l = 0; l < L; ++l) {
            // Load and decompose
            for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                uint64_t val = rlwe_b[in_c * N + i];
                work[i] = (val >> (l * base_log)) & base_mask;
            }
            __syncthreads();

            // Forward NTT
            for (uint32_t stage = 0; stage < log_N; ++stage) {
                uint32_t m = 1u << stage;
                uint32_t t = N >> (stage + 1);

                for (uint32_t bf = local_idx; bf < N / 2; bf += blockDim.x) {
                    uint32_t ii = bf / t;
                    uint32_t jj = bf % t;
                    uint32_t idx_lo = (ii << (log_N - stage)) + jj;
                    uint32_t idx_hi = idx_lo + t;
                    uint32_t tw_idx = m + ii;

                    ct_butterfly(work, idx_lo, idx_hi,
                                tw_fwd[tw_idx], pre_fwd[tw_idx], Q);
                }
                __syncthreads();
            }

            // Process each output component
            for (uint32_t out_c = 0; out_c < 2; ++out_c) {
                // Multiply with RGSW
                for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                    uint32_t rgsw_idx = in_c * L * 2 * N + l * 2 * N + out_c * N + i;
                    work[i] = mod_mul(work[i], rgsw_b[rgsw_idx], Q);
                }
                __syncthreads();

                // Inverse NTT
                for (uint32_t stage = 0; stage < log_N; ++stage) {
                    uint32_t m = N >> (stage + 1);
                    uint32_t t = 1u << stage;

                    for (uint32_t bf = local_idx; bf < N / 2; bf += blockDim.x) {
                        uint32_t ii = bf / t;
                        uint32_t jj = bf % t;
                        uint32_t idx_lo = (ii << (stage + 1)) + jj;
                        uint32_t idx_hi = idx_lo + t;
                        uint32_t tw_idx = m + ii;

                        gs_butterfly(work, idx_lo, idx_hi,
                                    tw_inv[tw_idx], pre_inv[tw_idx], Q);
                    }
                    __syncthreads();
                }

                // Scale and accumulate
                for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                    uint64_t val = barrett_mul(work[i], p.N_inv, Q, p.N_inv_precon);
                    acc_b[out_c * N + i] = mod_add(acc_b[out_c * N + i], val, Q);
                }
                __syncthreads();

                // Reload NTT digit for next output component
                if (out_c == 0) {
                    for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                        uint64_t val = rlwe_b[in_c * N + i];
                        work[i] = (val >> (l * base_log)) & base_mask;
                    }
                    __syncthreads();

                    for (uint32_t stage = 0; stage < log_N; ++stage) {
                        uint32_t m = 1u << stage;
                        uint32_t t = N >> (stage + 1);

                        for (uint32_t bf = local_idx; bf < N / 2; bf += blockDim.x) {
                            uint32_t ii = bf / t;
                            uint32_t jj = bf % t;
                            uint32_t idx_lo = (ii << (log_N - stage)) + jj;
                            uint32_t idx_hi = idx_lo + t;
                            uint32_t tw_idx = m + ii;

                            ct_butterfly(work, idx_lo, idx_hi,
                                        tw_fwd[tw_idx], pre_fwd[tw_idx], Q);
                        }
                        __syncthreads();
                    }
                }
            }
        }
    }
}

// =============================================================================
// HIGH-THROUGHPUT BATCH KERNEL
// =============================================================================
// Processes multiple batches per block to amortize twiddle prefetch.

extern "C" __global__
void fused_external_product_batch_v2(
    uint64_t* result,
    const uint64_t* rlwe,
    const uint64_t* rgsw,
    const uint64_t* fwd_twiddles,
    const uint64_t* fwd_precon,
    const uint64_t* inv_twiddles,
    const uint64_t* inv_precon,
    const FusedExternalProductParams p
) {
    extern __shared__ uint64_t shared[];

    const uint32_t BATCHES_PER_BLK = 4;

    uint32_t local_batch = threadIdx.y;
    uint32_t global_batch = blockIdx.y * BATCHES_PER_BLK + local_batch;
    uint32_t local_idx = threadIdx.x;
    uint32_t N = p.N;
    uint32_t log_N = p.log_N;
    uint32_t L = p.L;
    uint64_t Q = p.Q;
    uint64_t base_mask = p.base_mask;
    uint32_t base_log = p.base_log;

    if (global_batch >= p.batch_size) return;

    // Shared: twiddles (shared) + work buffers (per batch)
    uint64_t* tw_fwd    = shared;
    uint64_t* pre_fwd   = shared + N;
    uint64_t* tw_inv    = shared + 2 * N;
    uint64_t* pre_inv   = shared + 3 * N;
    uint64_t* work_bufs = shared + 4 * N;

    // Only first batch row prefetches twiddles
    if (local_batch == 0) {
        for (uint32_t i = local_idx; i < N; i += blockDim.x) {
            tw_fwd[i]  = fwd_twiddles[i];
            pre_fwd[i] = fwd_precon[i];
            tw_inv[i]  = inv_twiddles[i];
            pre_inv[i] = inv_precon[i];
        }
    }
    __syncthreads();

    uint64_t* work = work_bufs + local_batch * N;

    const uint64_t* rlwe_b   = rlwe + global_batch * 2 * N;
    const uint64_t* rgsw_b   = rgsw + global_batch * 2 * L * 2 * N;
    uint64_t*       result_b = result + global_batch * 2 * N;

    for (uint32_t in_c = 0; in_c < 2; ++in_c) {
        for (uint32_t l = 0; l < L; ++l) {
            // Decompose
            for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                uint64_t val = rlwe_b[in_c * N + i];
                work[i] = (val >> (l * base_log)) & base_mask;
            }
            __syncthreads();

            // Forward NTT
            for (uint32_t stage = 0; stage < log_N; ++stage) {
                uint32_t m = 1u << stage;
                uint32_t t = N >> (stage + 1);

                for (uint32_t bf = local_idx; bf < N / 2; bf += blockDim.x) {
                    uint32_t ii = bf / t;
                    uint32_t jj = bf % t;
                    uint32_t idx_lo = (ii << (log_N - stage)) + jj;
                    uint32_t idx_hi = idx_lo + t;
                    uint32_t tw_idx = m + ii;

                    ct_butterfly(work, idx_lo, idx_hi,
                                tw_fwd[tw_idx], pre_fwd[tw_idx], Q);
                }
                __syncthreads();
            }

            // Multiply and INTT for each output
            for (uint32_t out_c = 0; out_c < 2; ++out_c) {
                for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                    uint32_t rgsw_idx = in_c * L * 2 * N + l * 2 * N + out_c * N + i;
                    work[i] = mod_mul(work[i], rgsw_b[rgsw_idx], Q);
                }
                __syncthreads();

                // Inverse NTT
                for (uint32_t stage = 0; stage < log_N; ++stage) {
                    uint32_t m = N >> (stage + 1);
                    uint32_t t = 1u << stage;

                    for (uint32_t bf = local_idx; bf < N / 2; bf += blockDim.x) {
                        uint32_t ii = bf / t;
                        uint32_t jj = bf % t;
                        uint32_t idx_lo = (ii << (stage + 1)) + jj;
                        uint32_t idx_hi = idx_lo + t;
                        uint32_t tw_idx = m + ii;

                        gs_butterfly(work, idx_lo, idx_hi,
                                    tw_inv[tw_idx], pre_inv[tw_idx], Q);
                    }
                    __syncthreads();
                }

                // Scale and accumulate
                for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                    uint64_t val = barrett_mul(work[i], p.N_inv, Q, p.N_inv_precon);
                    if (in_c == 0 && l == 0) {
                        result_b[out_c * N + i] = val;
                    } else {
                        result_b[out_c * N + i] = mod_add(result_b[out_c * N + i], val, Q);
                    }
                }
                __syncthreads();

                // Reload for next output if needed
                if (out_c == 0) {
                    for (uint32_t i = local_idx; i < N; i += blockDim.x) {
                        uint64_t val = rlwe_b[in_c * N + i];
                        work[i] = (val >> (l * base_log)) & base_mask;
                    }
                    __syncthreads();

                    for (uint32_t stage = 0; stage < log_N; ++stage) {
                        uint32_t m = 1u << stage;
                        uint32_t t = N >> (stage + 1);

                        for (uint32_t bf = local_idx; bf < N / 2; bf += blockDim.x) {
                            uint32_t ii = bf / t;
                            uint32_t jj = bf % t;
                            uint32_t idx_lo = (ii << (log_N - stage)) + jj;
                            uint32_t idx_hi = idx_lo + t;
                            uint32_t tw_idx = m + ii;

                            ct_butterfly(work, idx_lo, idx_hi,
                                        tw_fwd[tw_idx], pre_fwd[tw_idx], Q);
                        }
                        __syncthreads();
                    }
                }
            }
        }
    }
}

#endif // __CUDA_ARCH__
