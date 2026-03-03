// =============================================================================
// Fused Blind Rotation CUDA Kernel - Lux FHE GPU Acceleration
// =============================================================================
// CUDA port of blind_rotate_fused.metal -- byte-identical output.
//
// Fuses the entire blind rotation loop (n iterations) into a single dispatch.
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#ifdef __CUDA_ARCH__
#define BR_DEVICE __device__ __forceinline__
#else
#define BR_DEVICE inline
#define __global__
#define __shared__
static inline uint64_t __umul64hi(uint64_t a, uint64_t b) {
    __uint128_t r = (__uint128_t)a * b; return (uint64_t)(r >> 64);
}
static inline void __syncthreads() {}
#endif

struct BlindRotateParams {
    uint64_t Q;
    uint64_t mu;
    uint64_t N_inv;
    uint64_t N_inv_precon;
    uint32_t N;
    uint32_t log_N;
    uint32_t n;
    uint32_t L;
    uint32_t base_log;
    uint64_t base_mask;
    uint32_t batch_size;
};

BR_DEVICE uint64_t barrett_mul(uint64_t a, uint64_t omega, uint64_t Q, uint64_t precon) {
    uint64_t q_approx = __umul64hi(a, precon);
    uint64_t product = a * omega;
    uint64_t result = product - q_approx * Q;
    uint64_t mask = (uint64_t)((int64_t)(result >= Q) * -1);
    return result - (mask & Q);
}

BR_DEVICE uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t Q) {
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

BR_DEVICE uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    uint64_t mask = (uint64_t)((int64_t)(sum >= Q) * -1);
    return sum - (mask & Q);
}

BR_DEVICE uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t mask = (uint64_t)((int64_t)(a < b) * -1);
    return a - b + (mask & Q);
}

// CT and GS butterflies operating on shared memory
BR_DEVICE void ct_butterfly(uint64_t* data, uint32_t idx_lo, uint32_t idx_hi,
                             uint64_t omega, uint64_t precon, uint64_t Q) {
    uint64_t lo_val = data[idx_lo];
    uint64_t hi_val = data[idx_hi];
    uint64_t hi_tw = barrett_mul(hi_val, omega, Q, precon);
    data[idx_lo] = mod_add(lo_val, hi_tw, Q);
    data[idx_hi] = mod_sub(lo_val, hi_tw, Q);
}

BR_DEVICE void gs_butterfly(uint64_t* data, uint32_t idx_lo, uint32_t idx_hi,
                             uint64_t omega, uint64_t precon, uint64_t Q) {
    uint64_t lo_val = data[idx_lo];
    uint64_t hi_val = data[idx_hi];
    uint64_t sum = mod_add(lo_val, hi_val, Q);
    uint64_t diff = mod_sub(lo_val, hi_val, Q);
    data[idx_lo] = sum;
    data[idx_hi] = barrett_mul(diff, omega, Q, precon);
}

BR_DEVICE void negacyclic_rotate_inplace(
    uint64_t* dst, uint64_t* src, int32_t rotation,
    uint32_t N, uint64_t Q, uint32_t local_id, uint32_t num_threads)
{
    int32_t rot = rotation;
    while (rot < 0) rot += 2 * (int32_t)N;
    rot = rot % (2 * (int32_t)N);

    for (uint32_t i = local_id; i < N; i += num_threads) {
        int32_t src_idx = (int32_t)i - rot;
        bool negate = false;
        while (src_idx < 0) { src_idx += N; negate = !negate; }
        while (src_idx >= (int32_t)N) { src_idx -= N; negate = !negate; }
        uint64_t val = src[src_idx];
        dst[i] = negate ? mod_sub(0, val, Q) : val;
    }
}

// =============================================================================
// FUSED BLIND ROTATION KERNEL
// =============================================================================

extern "C" __global__ void blind_rotate_fused(
    int64_t* acc_out,
    const int64_t* lwe_in,
    const int64_t* bsk,
    const int64_t* test_poly,
    const uint64_t* twiddles,
    const uint64_t* tw_precon,
    const uint64_t* inv_tw,
    const uint64_t* inv_precon,
    const BlindRotateParams params)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared[];

    uint32_t batch_idx = blockIdx.x;
    uint32_t local_id = threadIdx.x;
    uint32_t num_threads = blockDim.x;

    uint32_t N = params.N;
    uint32_t log_N = params.log_N;
    uint32_t n = params.n;
    uint32_t L = params.L;
    uint64_t Q = params.Q;
    uint64_t base_mask = params.base_mask;
    uint32_t base_log = params.base_log;

    if (batch_idx >= params.batch_size) return;

    uint64_t* acc_c0 = shared;
    uint64_t* acc_c1 = shared + N;
    uint64_t* rot_c0 = shared + 2 * N;
    uint64_t* rot_c1 = shared + 3 * N;
    uint64_t* work   = shared + 4 * N;

    const int64_t* lwe = lwe_in + batch_idx * (n + 1);
    int64_t* out = acc_out + batch_idx * 2 * N;

    // Stage 1: Initialize accumulator
    int64_t b_val = lwe[n];
    int32_t b_mod = (int32_t)((b_val % (int64_t)(2 * N) + (int64_t)(2 * N)) % (int64_t)(2 * N));
    int32_t init_rot = (b_mod == 0) ? 0 : (int32_t)(2 * N) - b_mod;

    for (uint32_t i = local_id; i < N; i += num_threads) {
        acc_c0[i] = 0;
        int32_t src_idx = (int32_t)i - init_rot;
        bool negate = false;
        while (src_idx < 0) { src_idx += N; negate = !negate; }
        while (src_idx >= (int32_t)N) { src_idx -= N; negate = !negate; }
        uint64_t val = (uint64_t)test_poly[src_idx] % Q;
        acc_c1[i] = negate ? mod_sub(0, val, Q) : val;
    }
    __syncthreads();

    // Stage 2: Main blind rotation loop
    for (uint32_t j = 0; j < n; ++j) {
        int64_t a_j = lwe[j];
        int32_t rot = (int32_t)((a_j % (int64_t)(2 * N) + (int64_t)(2 * N)) % (int64_t)(2 * N));
        if (rot == 0) continue;

        negacyclic_rotate_inplace(rot_c0, acc_c0, rot, N, Q, local_id, num_threads);
        negacyclic_rotate_inplace(rot_c1, acc_c1, rot, N, Q, local_id, num_threads);
        __syncthreads();

        for (uint32_t i = local_id; i < N; i += num_threads) {
            rot_c0[i] = mod_sub(rot_c0[i], acc_c0[i], Q);
            rot_c1[i] = mod_sub(rot_c1[i], acc_c1[i], Q);
        }
        __syncthreads();

        for (uint32_t in_c = 0; in_c < 2; ++in_c) {
            uint64_t* diff_comp = (in_c == 0) ? rot_c0 : rot_c1;
            for (uint32_t l = 0; l < L; ++l) {
                for (uint32_t i = local_id; i < N; i += num_threads) {
                    work[i] = (diff_comp[i] >> (l * base_log)) & base_mask;
                }
                __syncthreads();

                // Forward NTT
                for (uint32_t stage = 0; stage < log_N; ++stage) {
                    uint32_t m = 1u << stage;
                    uint32_t t = N >> (stage + 1);
                    for (uint32_t bf = local_id; bf < N / 2; bf += num_threads) {
                        uint32_t ii = bf / t, jj = bf % t;
                        uint32_t idx_lo = (ii << (log_N - stage)) + jj;
                        uint32_t idx_hi = idx_lo + t;
                        uint32_t tw_idx = m + ii;
                        ct_butterfly(work, idx_lo, idx_hi, twiddles[tw_idx], tw_precon[tw_idx], Q);
                    }
                    __syncthreads();
                }

                const int64_t* rgsw_base = bsk + j * 2 * L * 2 * N + in_c * L * 2 * N + l * 2 * N;
                for (uint32_t out_c = 0; out_c < 2; ++out_c) {
                    const int64_t* rgsw_poly = rgsw_base + out_c * N;
                    for (uint32_t i = local_id; i < N; i += num_threads) {
                        uint64_t digit_ntt = work[i];
                        uint64_t rgsw_val = (uint64_t)rgsw_poly[i] % Q;
                        if (in_c == 0 && l == 0) {
                            if (out_c == 0) rot_c0[i] = mod_mul(digit_ntt, rgsw_val, Q);
                            else            rot_c1[i] = mod_mul(digit_ntt, rgsw_val, Q);
                        } else {
                            uint64_t prod = mod_mul(digit_ntt, rgsw_val, Q);
                            if (out_c == 0) rot_c0[i] = mod_add(rot_c0[i], prod, Q);
                            else            rot_c1[i] = mod_add(rot_c1[i], prod, Q);
                        }
                    }
                    __syncthreads();
                }
            }
        }

        // Inverse NTT on rot_c0
        for (uint32_t stage = 0; stage < log_N; ++stage) {
            uint32_t m = N >> (stage + 1);
            uint32_t t = 1u << stage;
            for (uint32_t bf = local_id; bf < N / 2; bf += num_threads) {
                uint32_t ii = bf / t, jj = bf % t;
                uint32_t idx_lo = (ii << (stage + 1)) + jj;
                uint32_t idx_hi = idx_lo + t;
                uint32_t tw_idx = m + ii;
                gs_butterfly(rot_c0, idx_lo, idx_hi, inv_tw[tw_idx], inv_precon[tw_idx], Q);
            }
            __syncthreads();
        }
        // Inverse NTT on rot_c1
        for (uint32_t stage = 0; stage < log_N; ++stage) {
            uint32_t m = N >> (stage + 1);
            uint32_t t = 1u << stage;
            for (uint32_t bf = local_id; bf < N / 2; bf += num_threads) {
                uint32_t ii = bf / t, jj = bf % t;
                uint32_t idx_lo = (ii << (stage + 1)) + jj;
                uint32_t idx_hi = idx_lo + t;
                uint32_t tw_idx = m + ii;
                gs_butterfly(rot_c1, idx_lo, idx_hi, inv_tw[tw_idx], inv_precon[tw_idx], Q);
            }
            __syncthreads();
        }

        for (uint32_t i = local_id; i < N; i += num_threads) {
            uint64_t prod0 = barrett_mul(rot_c0[i], params.N_inv, Q, params.N_inv_precon);
            uint64_t prod1 = barrett_mul(rot_c1[i], params.N_inv, Q, params.N_inv_precon);
            acc_c0[i] = mod_add(acc_c0[i], prod0, Q);
            acc_c1[i] = mod_add(acc_c1[i], prod1, Q);
        }
        __syncthreads();
    }

    // Stage 3: Write output
    for (uint32_t i = local_id; i < N; i += num_threads) {
        out[i] = (int64_t)acc_c0[i];
        out[N + i] = (int64_t)acc_c1[i];
    }
#endif
}

// =============================================================================
// SIMPLIFIED VARIANT (testing)
// =============================================================================

extern "C" __global__ void blind_rotate_simplified(
    int64_t* acc_out, const int64_t* lwe_in, const int64_t* test_poly,
    const BlindRotateParams params)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared[];

    uint32_t batch_idx = blockIdx.x;
    uint32_t local_id = threadIdx.x;
    uint32_t num_threads = blockDim.x;
    uint32_t N = params.N;
    uint32_t n = params.n;
    uint64_t Q = params.Q;

    if (batch_idx >= params.batch_size) return;

    uint64_t* acc_c0 = shared;
    uint64_t* acc_c1 = shared + N;

    const int64_t* lwe = lwe_in + batch_idx * (n + 1);
    int64_t* out = acc_out + batch_idx * 2 * N;

    int64_t b_val = lwe[n];
    int32_t b_mod = (int32_t)((b_val % (int64_t)(2 * N) + (int64_t)(2 * N)) % (int64_t)(2 * N));
    int32_t init_rot = (b_mod == 0) ? 0 : (int32_t)(2 * N) - b_mod;

    for (uint32_t i = local_id; i < N; i += num_threads) {
        acc_c0[i] = 0;
        int32_t src_idx = (int32_t)i - init_rot;
        bool negate = false;
        while (src_idx < 0) { src_idx += N; negate = !negate; }
        while (src_idx >= (int32_t)N) { src_idx -= N; negate = !negate; }
        uint64_t val = (uint64_t)test_poly[src_idx] % Q;
        acc_c1[i] = negate ? mod_sub(0, val, Q) : val;
    }
    __syncthreads();

    for (uint32_t i = local_id; i < N; i += num_threads) {
        out[i] = (int64_t)acc_c0[i];
        out[N + i] = (int64_t)acc_c1[i];
    }
#endif
}
