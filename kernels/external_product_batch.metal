// =============================================================================
// Batched External Product Metal Kernel - Lux FHE GPU Acceleration
// =============================================================================
//
// Fully batched external product: multiple RLWE ciphertexts against a single
// RGSW ciphertext. Optimized for CMux gates in blind rotation.
//
// Key Innovation: Fuses decompose + multiply + accumulate into one kernel
// - Eliminates 2 kernel launches per external product
// - Single RGSW shared across all batch elements
// - Barrett reduction throughout for efficiency
//
// Expected speedup: 2-3x for external product operations
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Kernel Parameters
// =============================================================================

struct BatchedExtProdParams {
    uint64_t Q;              // Prime modulus
    uint64_t barrett_mu;     // Barrett constant: floor(2^64 / Q)
    uint64_t N_inv;          // N^{-1} mod Q (for INTT scaling)
    uint64_t N_inv_precon;   // Barrett precomputation for N_inv
    uint32_t N;              // Ring dimension (power of 2)
    uint32_t log_N;          // log2(N)
    uint32_t L;              // Decomposition levels
    uint32_t base_log;       // log2 of decomposition base
    uint64_t base_mask;      // (1 << base_log) - 1
    uint32_t batch_size;     // Number of RLWE ciphertexts in batch
};

// =============================================================================
// Barrett Modular Arithmetic
// =============================================================================

// High 64 bits of 64x64 multiplication
inline uint64_t mulhi64(uint64_t a, uint64_t b) {
    return metal::mulhi(a, b);
}

// Barrett reduction: x mod Q
// For x in [0, Q^2), returns x mod Q
inline uint64_t barrett_reduce(uint64_t x, uint64_t Q, uint64_t mu) {
    uint64_t q_approx = mulhi64(x, mu);
    uint64_t result = x - q_approx * Q;
    // Result in [0, 2Q), one conditional subtraction
    return (result >= Q) ? result - Q : result;
}

// Barrett multiplication: (a * b) mod Q
inline uint64_t barrett_mul(uint64_t a, uint64_t b, uint64_t Q, uint64_t mu) {
    uint64_t lo = a * b;
    uint64_t hi = mulhi64(a, b);

    // Fast path for products fitting in 64 bits
    if (hi == 0) {
        return barrett_reduce(lo, Q, mu);
    }

    // Full 128-bit reduction
    // (hi * 2^64 + lo) mod Q
    // 2^64 mod Q = (2^32 mod Q)^2 mod Q
    uint64_t two32 = uint64_t(1) << 32;
    uint64_t two32_mod = two32 % Q;
    uint64_t two64_mod = barrett_reduce(two32_mod * two32_mod, Q, mu);

    uint64_t hi_contrib = barrett_reduce(hi * two64_mod, Q, mu);
    uint64_t lo_mod = barrett_reduce(lo, Q, mu);

    return barrett_reduce(lo_mod + hi_contrib, Q, mu);
}

// Modular addition: (a + b) mod Q
inline uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    return (sum >= Q) ? sum - Q : sum;
}

// Modular subtraction: (a - b) mod Q
inline uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    return (a >= b) ? a - b : a + Q - b;
}

// =============================================================================
// Gadget Decomposition Helper
// =============================================================================
//
// Decomposes polynomial coefficient into L digits in base 2^base_log
// Each digit is in [0, 2^base_log - 1]

inline void gadget_decompose_coefficient(
    thread uint64_t* digits,     // Output: L digits
    uint64_t value,              // Input coefficient
    uint32_t L,                  // Number of levels
    uint32_t base_log,           // Bits per digit
    uint64_t base_mask           // (1 << base_log) - 1
) {
    for (uint32_t l = 0; l < L && l < 8; ++l) {
        digits[l] = (value >> (l * base_log)) & base_mask;
    }
}

// =============================================================================
// NTT Butterfly Operations (for optional NTT-domain processing)
// =============================================================================

// Cooley-Tukey butterfly for forward NTT
inline void ct_butterfly(
    threadgroup uint64_t* data,
    uint32_t idx_lo,
    uint32_t idx_hi,
    uint64_t omega,
    uint64_t Q,
    uint64_t mu
) {
    uint64_t lo = data[idx_lo];
    uint64_t hi = data[idx_hi];

    uint64_t hi_tw = barrett_mul(hi, omega, Q, mu);

    data[idx_lo] = mod_add(lo, hi_tw, Q);
    data[idx_hi] = mod_sub(lo, hi_tw, Q);
}

// Gentleman-Sande butterfly for inverse NTT
inline void gs_butterfly(
    threadgroup uint64_t* data,
    uint32_t idx_lo,
    uint32_t idx_hi,
    uint64_t omega,
    uint64_t Q,
    uint64_t mu
) {
    uint64_t lo = data[idx_lo];
    uint64_t hi = data[idx_hi];

    uint64_t sum = mod_add(lo, hi, Q);
    uint64_t diff = mod_sub(lo, hi, Q);
    uint64_t diff_tw = barrett_mul(diff, omega, Q, mu);

    data[idx_lo] = sum;
    data[idx_hi] = diff_tw;
}

// =============================================================================
// MAIN BATCHED EXTERNAL PRODUCT KERNEL
// =============================================================================
//
// Computes: out[b] = ExternalProduct(rlwe[b], rgsw) for all b in [0, batch_size)
//
// This kernel is optimized for the CMux pattern where a single RGSW ciphertext
// is multiplied with multiple RLWE ciphertexts.
//
// Thread Organization:
//   Grid: [batch_size, 1, 1]
//   Threadgroup: [threads_per_group, 1, 1] where threads_per_group = min(N, 256)
//
// Memory Layout:
//   rlwe_c0, rlwe_c1: [batch, N] - RLWE components
//   rgsw: [2, L, 2, N] - Single RGSW ciphertext
//   out_c0, out_c1: [batch, N] - Output RLWE components
//
// Shared Memory:
//   decomp_c0: [L, N] - Decomposed digits for c0
//   decomp_c1: [L, N] - Decomposed digits for c1
//   acc_c0, acc_c1: [N] - Accumulators for output

kernel void external_product_batched(
    device uint64_t* out_c0                  [[buffer(0)]],   // [batch, N]
    device uint64_t* out_c1                  [[buffer(1)]],   // [batch, N]
    device const uint64_t* rlwe_c0           [[buffer(2)]],   // [batch, N]
    device const uint64_t* rlwe_c1           [[buffer(3)]],   // [batch, N]
    device const uint64_t* rgsw              [[buffer(4)]],   // [2, L, 2, N]
    constant BatchedExtProdParams& p         [[buffer(5)]],

    uint3 tid                                [[thread_position_in_threadgroup]],
    uint3 tgid                               [[threadgroup_position_in_grid]],
    uint3 tg_size                            [[threads_per_threadgroup]],

    threadgroup uint64_t* shared             [[threadgroup(0)]]
) {
    uint32_t batch_idx = tgid.x;
    if (batch_idx >= p.batch_size) return;

    uint32_t local_id = tid.x;
    uint32_t threads = tg_size.x;
    uint32_t N = p.N;
    uint32_t L = p.L;
    uint64_t Q = p.Q;
    uint64_t mu = p.barrett_mu;
    uint32_t base_log = p.base_log;
    uint64_t base_mask = p.base_mask;

    // Shared memory layout:
    // [0, L*N)           : decomp_c0
    // [L*N, 2*L*N)       : decomp_c1
    // [2*L*N, 2*L*N + N) : acc_c0
    // [2*L*N + N, 2*L*N + 2*N) : acc_c1
    threadgroup uint64_t* decomp_c0 = shared;
    threadgroup uint64_t* decomp_c1 = shared + L * N;
    threadgroup uint64_t* acc_c0 = shared + 2 * L * N;
    threadgroup uint64_t* acc_c1 = shared + 2 * L * N + N;

    // Pointers to batch data
    device const uint64_t* c0 = rlwe_c0 + batch_idx * N;
    device const uint64_t* c1 = rlwe_c1 + batch_idx * N;
    device uint64_t* o_c0 = out_c0 + batch_idx * N;
    device uint64_t* o_c1 = out_c1 + batch_idx * N;

    // =========================================================================
    // STAGE 1: Gadget Decomposition (cooperative, all threads participate)
    // =========================================================================
    // Decompose both RLWE components into L digits

    for (uint32_t i = local_id; i < N; i += threads) {
        uint64_t val_c0 = c0[i];
        uint64_t val_c1 = c1[i];

        for (uint32_t l = 0; l < L; ++l) {
            uint64_t digit_c0 = (val_c0 >> (l * base_log)) & base_mask;
            uint64_t digit_c1 = (val_c1 >> (l * base_log)) & base_mask;

            decomp_c0[l * N + i] = digit_c0;
            decomp_c1[l * N + i] = digit_c1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // STAGE 2: Initialize Accumulators
    // =========================================================================

    for (uint32_t i = local_id; i < N; i += threads) {
        acc_c0[i] = 0;
        acc_c1[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // STAGE 3: External Product Accumulation
    // =========================================================================
    // For each level l and input component c:
    //   acc += decomp[c][l] * rgsw[c][l]
    //
    // RGSW layout: [input_component, level, output_component, coefficient]
    //   rgsw[c, l, out_c, i] = rgsw[(c * L * 2 * N) + (l * 2 * N) + (out_c * N) + i]

    for (uint32_t l = 0; l < L; ++l) {
        // Process contribution from RLWE component 0 (c0)
        // RGSW row for c0, level l: rgsw[0, l, :, :]
        device const uint64_t* rgsw_c0_l_0 = rgsw + (0 * L * 2 * N) + (l * 2 * N) + (0 * N);
        device const uint64_t* rgsw_c0_l_1 = rgsw + (0 * L * 2 * N) + (l * 2 * N) + (1 * N);

        for (uint32_t i = local_id; i < N; i += threads) {
            uint64_t d0 = decomp_c0[l * N + i];

            // Multiply digit with RGSW components
            uint64_t prod_0 = barrett_mul(d0, rgsw_c0_l_0[i], Q, mu);
            uint64_t prod_1 = barrett_mul(d0, rgsw_c0_l_1[i], Q, mu);

            // Accumulate
            acc_c0[i] = mod_add(acc_c0[i], prod_0, Q);
            acc_c1[i] = mod_add(acc_c1[i], prod_1, Q);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process contribution from RLWE component 1 (c1)
        // RGSW row for c1, level l: rgsw[1, l, :, :]
        device const uint64_t* rgsw_c1_l_0 = rgsw + (1 * L * 2 * N) + (l * 2 * N) + (0 * N);
        device const uint64_t* rgsw_c1_l_1 = rgsw + (1 * L * 2 * N) + (l * 2 * N) + (1 * N);

        for (uint32_t i = local_id; i < N; i += threads) {
            uint64_t d1 = decomp_c1[l * N + i];

            uint64_t prod_0 = barrett_mul(d1, rgsw_c1_l_0[i], Q, mu);
            uint64_t prod_1 = barrett_mul(d1, rgsw_c1_l_1[i], Q, mu);

            acc_c0[i] = mod_add(acc_c0[i], prod_0, Q);
            acc_c1[i] = mod_add(acc_c1[i], prod_1, Q);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // =========================================================================
    // STAGE 4: Write Output
    // =========================================================================

    for (uint32_t i = local_id; i < N; i += threads) {
        o_c0[i] = acc_c0[i];
        o_c1[i] = acc_c1[i];
    }
}

// =============================================================================
// ACCUMULATING VARIANT: acc += ExternalProduct(rlwe, rgsw)
// =============================================================================
// For blind rotation where results accumulate iteratively

kernel void external_product_batched_accumulate(
    device uint64_t* acc_c0                  [[buffer(0)]],   // [batch, N] in/out
    device uint64_t* acc_c1                  [[buffer(1)]],   // [batch, N] in/out
    device const uint64_t* rlwe_c0           [[buffer(2)]],   // [batch, N]
    device const uint64_t* rlwe_c1           [[buffer(3)]],   // [batch, N]
    device const uint64_t* rgsw              [[buffer(4)]],   // [2, L, 2, N]
    constant BatchedExtProdParams& p         [[buffer(5)]],

    uint3 tid                                [[thread_position_in_threadgroup]],
    uint3 tgid                               [[threadgroup_position_in_grid]],
    uint3 tg_size                            [[threads_per_threadgroup]],

    threadgroup uint64_t* shared             [[threadgroup(0)]]
) {
    uint32_t batch_idx = tgid.x;
    if (batch_idx >= p.batch_size) return;

    uint32_t local_id = tid.x;
    uint32_t threads = tg_size.x;
    uint32_t N = p.N;
    uint32_t L = p.L;
    uint64_t Q = p.Q;
    uint64_t mu = p.barrett_mu;
    uint32_t base_log = p.base_log;
    uint64_t base_mask = p.base_mask;

    // Shared memory for decomposed digits
    threadgroup uint64_t* decomp_c0 = shared;
    threadgroup uint64_t* decomp_c1 = shared + L * N;

    device const uint64_t* c0 = rlwe_c0 + batch_idx * N;
    device const uint64_t* c1 = rlwe_c1 + batch_idx * N;
    device uint64_t* a_c0 = acc_c0 + batch_idx * N;
    device uint64_t* a_c1 = acc_c1 + batch_idx * N;

    // Decompose RLWE
    for (uint32_t i = local_id; i < N; i += threads) {
        uint64_t val_c0 = c0[i];
        uint64_t val_c1 = c1[i];

        for (uint32_t l = 0; l < L; ++l) {
            decomp_c0[l * N + i] = (val_c0 >> (l * base_log)) & base_mask;
            decomp_c1[l * N + i] = (val_c1 >> (l * base_log)) & base_mask;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate external product contributions
    for (uint32_t l = 0; l < L; ++l) {
        device const uint64_t* rgsw_c0_l_0 = rgsw + (0 * L * 2 * N) + (l * 2 * N);
        device const uint64_t* rgsw_c0_l_1 = rgsw + (0 * L * 2 * N) + (l * 2 * N) + N;
        device const uint64_t* rgsw_c1_l_0 = rgsw + (1 * L * 2 * N) + (l * 2 * N);
        device const uint64_t* rgsw_c1_l_1 = rgsw + (1 * L * 2 * N) + (l * 2 * N) + N;

        for (uint32_t i = local_id; i < N; i += threads) {
            uint64_t d0 = decomp_c0[l * N + i];
            uint64_t d1 = decomp_c1[l * N + i];

            // Contributions from c0 decomposition
            uint64_t prod_00 = barrett_mul(d0, rgsw_c0_l_0[i], Q, mu);
            uint64_t prod_01 = barrett_mul(d0, rgsw_c0_l_1[i], Q, mu);

            // Contributions from c1 decomposition
            uint64_t prod_10 = barrett_mul(d1, rgsw_c1_l_0[i], Q, mu);
            uint64_t prod_11 = barrett_mul(d1, rgsw_c1_l_1[i], Q, mu);

            // Accumulate to global memory
            a_c0[i] = mod_add(a_c0[i], mod_add(prod_00, prod_10, Q), Q);
            a_c1[i] = mod_add(a_c1[i], mod_add(prod_01, prod_11, Q), Q);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// =============================================================================
// NTT-DOMAIN BATCHED EXTERNAL PRODUCT
// =============================================================================
// For when RGSW is in NTT domain and output should also be in NTT domain
// Skips coefficient-domain conversion for faster chained operations

kernel void external_product_batched_ntt(
    device uint64_t* out_c0                  [[buffer(0)]],   // [batch, N] output (NTT domain)
    device uint64_t* out_c1                  [[buffer(1)]],   // [batch, N] output (NTT domain)
    device const uint64_t* rlwe_c0           [[buffer(2)]],   // [batch, N] (coeff domain)
    device const uint64_t* rlwe_c1           [[buffer(3)]],   // [batch, N] (coeff domain)
    device const uint64_t* rgsw_ntt          [[buffer(4)]],   // [2, L, 2, N] (NTT domain)
    constant uint64_t* fwd_twiddles          [[buffer(5)]],   // [N] forward twiddles
    constant uint64_t* fwd_precon            [[buffer(6)]],   // [N] forward precon
    constant BatchedExtProdParams& p         [[buffer(7)]],

    uint3 tid                                [[thread_position_in_threadgroup]],
    uint3 tgid                               [[threadgroup_position_in_grid]],
    uint3 tg_size                            [[threads_per_threadgroup]],

    threadgroup uint64_t* shared             [[threadgroup(0)]]
) {
    uint32_t batch_idx = tgid.x;
    if (batch_idx >= p.batch_size) return;

    uint32_t local_id = tid.x;
    uint32_t threads = tg_size.x;
    uint32_t N = p.N;
    uint32_t log_N = p.log_N;
    uint32_t L = p.L;
    uint64_t Q = p.Q;
    uint64_t mu = p.barrett_mu;
    uint32_t base_log = p.base_log;
    uint64_t base_mask = p.base_mask;

    // Shared memory layout:
    // [0, N)      : work buffer for NTT
    // [N, 2N)     : forward twiddles (cached)
    // [2N, 3N)    : forward precon (cached)
    // [3N, 4N)    : acc_c0
    // [4N, 5N)    : acc_c1
    threadgroup uint64_t* work = shared;
    threadgroup uint64_t* tw_fwd = shared + N;
    threadgroup uint64_t* pre_fwd = shared + 2 * N;
    threadgroup uint64_t* acc_c0 = shared + 3 * N;
    threadgroup uint64_t* acc_c1 = shared + 4 * N;

    device const uint64_t* c0 = rlwe_c0 + batch_idx * N;
    device const uint64_t* c1 = rlwe_c1 + batch_idx * N;
    device uint64_t* o_c0 = out_c0 + batch_idx * N;
    device uint64_t* o_c1 = out_c1 + batch_idx * N;

    // Prefetch twiddles to shared memory
    for (uint32_t i = local_id; i < N; i += threads) {
        tw_fwd[i] = fwd_twiddles[i];
        pre_fwd[i] = fwd_precon[i];
        acc_c0[i] = 0;
        acc_c1[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process each decomposition level
    for (uint32_t in_c = 0; in_c < 2; ++in_c) {
        device const uint64_t* rlwe_comp = (in_c == 0) ? c0 : c1;

        for (uint32_t l = 0; l < L; ++l) {
            // Load and decompose to work buffer
            for (uint32_t i = local_id; i < N; i += threads) {
                uint64_t val = rlwe_comp[i];
                work[i] = (val >> (l * base_log)) & base_mask;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Forward NTT (Cooley-Tukey DIT)
            for (uint32_t stage = 0; stage < log_N; ++stage) {
                uint32_t m = 1u << stage;
                uint32_t t = N >> (stage + 1);

                for (uint32_t bf = local_id; bf < N / 2; bf += threads) {
                    uint32_t ii = bf / t;
                    uint32_t jj = bf % t;
                    uint32_t idx_lo = (ii << (log_N - stage)) + jj;
                    uint32_t idx_hi = idx_lo + t;
                    uint32_t tw_idx = m + ii;

                    uint64_t lo = work[idx_lo];
                    uint64_t hi = work[idx_hi];
                    uint64_t omega = tw_fwd[tw_idx];
                    uint64_t precon = pre_fwd[tw_idx];

                    // Barrett multiplication with precomputed constant
                    uint64_t q_approx = mulhi64(hi, precon);
                    uint64_t hi_tw = hi * omega - q_approx * Q;
                    if (hi_tw >= Q) hi_tw -= Q;

                    work[idx_lo] = mod_add(lo, hi_tw, Q);
                    work[idx_hi] = mod_sub(lo, hi_tw, Q);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Pointwise multiply with RGSW and accumulate
            device const uint64_t* rgsw_l_0 = rgsw_ntt + (in_c * L * 2 * N) + (l * 2 * N);
            device const uint64_t* rgsw_l_1 = rgsw_ntt + (in_c * L * 2 * N) + (l * 2 * N) + N;

            for (uint32_t i = local_id; i < N; i += threads) {
                uint64_t digit_ntt = work[i];

                uint64_t prod_0 = barrett_mul(digit_ntt, rgsw_l_0[i], Q, mu);
                uint64_t prod_1 = barrett_mul(digit_ntt, rgsw_l_1[i], Q, mu);

                acc_c0[i] = mod_add(acc_c0[i], prod_0, Q);
                acc_c1[i] = mod_add(acc_c1[i], prod_1, Q);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write accumulated result (in NTT domain)
    for (uint32_t i = local_id; i < N; i += threads) {
        o_c0[i] = acc_c0[i];
        o_c1[i] = acc_c1[i];
    }
}

// =============================================================================
// CMUX GATE: result = d0 + ExternalProduct(d1 - d0, rgsw_bit)
// =============================================================================
// Fused CMux gate for blind rotation - single kernel for entire operation

kernel void cmux_batched(
    device uint64_t* out_c0                  [[buffer(0)]],   // [batch, N] output
    device uint64_t* out_c1                  [[buffer(1)]],   // [batch, N] output
    device const uint64_t* d0_c0             [[buffer(2)]],   // [batch, N] first option
    device const uint64_t* d0_c1             [[buffer(3)]],   // [batch, N]
    device const uint64_t* d1_c0             [[buffer(4)]],   // [batch, N] second option
    device const uint64_t* d1_c1             [[buffer(5)]],   // [batch, N]
    device const uint64_t* rgsw_bit          [[buffer(6)]],   // [2, L, 2, N] RGSW(bit)
    constant BatchedExtProdParams& p         [[buffer(7)]],

    uint3 tid                                [[thread_position_in_threadgroup]],
    uint3 tgid                               [[threadgroup_position_in_grid]],
    uint3 tg_size                            [[threads_per_threadgroup]],

    threadgroup uint64_t* shared             [[threadgroup(0)]]
) {
    uint32_t batch_idx = tgid.x;
    if (batch_idx >= p.batch_size) return;

    uint32_t local_id = tid.x;
    uint32_t threads = tg_size.x;
    uint32_t N = p.N;
    uint32_t L = p.L;
    uint64_t Q = p.Q;
    uint64_t mu = p.barrett_mu;
    uint32_t base_log = p.base_log;
    uint64_t base_mask = p.base_mask;

    // Shared memory
    threadgroup uint64_t* decomp_c0 = shared;
    threadgroup uint64_t* decomp_c1 = shared + L * N;
    threadgroup uint64_t* acc_c0 = shared + 2 * L * N;
    threadgroup uint64_t* acc_c1 = shared + 2 * L * N + N;

    // Batch pointers
    device const uint64_t* d0_0 = d0_c0 + batch_idx * N;
    device const uint64_t* d0_1 = d0_c1 + batch_idx * N;
    device const uint64_t* d1_0 = d1_c0 + batch_idx * N;
    device const uint64_t* d1_1 = d1_c1 + batch_idx * N;
    device uint64_t* o_c0 = out_c0 + batch_idx * N;
    device uint64_t* o_c1 = out_c1 + batch_idx * N;

    // Compute diff = d1 - d0 and decompose
    for (uint32_t i = local_id; i < N; i += threads) {
        uint64_t diff_c0 = mod_sub(d1_0[i], d0_0[i], Q);
        uint64_t diff_c1 = mod_sub(d1_1[i], d0_1[i], Q);

        // Decompose diff
        for (uint32_t l = 0; l < L; ++l) {
            decomp_c0[l * N + i] = (diff_c0 >> (l * base_log)) & base_mask;
            decomp_c1[l * N + i] = (diff_c1 >> (l * base_log)) & base_mask;
        }

        // Initialize accumulators with d0
        acc_c0[i] = d0_0[i];
        acc_c1[i] = d0_1[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // External product: ExternalProduct(diff, rgsw_bit)
    for (uint32_t l = 0; l < L; ++l) {
        device const uint64_t* rgsw_c0_l_0 = rgsw_bit + (0 * L * 2 * N) + (l * 2 * N);
        device const uint64_t* rgsw_c0_l_1 = rgsw_bit + (0 * L * 2 * N) + (l * 2 * N) + N;
        device const uint64_t* rgsw_c1_l_0 = rgsw_bit + (1 * L * 2 * N) + (l * 2 * N);
        device const uint64_t* rgsw_c1_l_1 = rgsw_bit + (1 * L * 2 * N) + (l * 2 * N) + N;

        for (uint32_t i = local_id; i < N; i += threads) {
            uint64_t d0 = decomp_c0[l * N + i];
            uint64_t d1 = decomp_c1[l * N + i];

            // c0 contribution
            uint64_t prod_00 = barrett_mul(d0, rgsw_c0_l_0[i], Q, mu);
            uint64_t prod_01 = barrett_mul(d0, rgsw_c0_l_1[i], Q, mu);

            // c1 contribution
            uint64_t prod_10 = barrett_mul(d1, rgsw_c1_l_0[i], Q, mu);
            uint64_t prod_11 = barrett_mul(d1, rgsw_c1_l_1[i], Q, mu);

            // Accumulate: acc += external_product
            acc_c0[i] = mod_add(acc_c0[i], mod_add(prod_00, prod_10, Q), Q);
            acc_c1[i] = mod_add(acc_c1[i], mod_add(prod_01, prod_11, Q), Q);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    for (uint32_t i = local_id; i < N; i += threads) {
        o_c0[i] = acc_c0[i];
        o_c1[i] = acc_c1[i];
    }
}

// =============================================================================
// MULTI-RGSW BATCHED EXTERNAL PRODUCT
// =============================================================================
// For processing multiple RLWE x RGSW pairs in parallel (each pair independent)

kernel void external_product_multi_batched(
    device uint64_t* out_c0                  [[buffer(0)]],   // [batch, N]
    device uint64_t* out_c1                  [[buffer(1)]],   // [batch, N]
    device const uint64_t* rlwe_c0           [[buffer(2)]],   // [batch, N]
    device const uint64_t* rlwe_c1           [[buffer(3)]],   // [batch, N]
    device const uint64_t* rgsw_batch        [[buffer(4)]],   // [batch, 2, L, 2, N]
    constant BatchedExtProdParams& p         [[buffer(5)]],

    uint3 tid                                [[thread_position_in_threadgroup]],
    uint3 tgid                               [[threadgroup_position_in_grid]],
    uint3 tg_size                            [[threads_per_threadgroup]],

    threadgroup uint64_t* shared             [[threadgroup(0)]]
) {
    uint32_t batch_idx = tgid.x;
    if (batch_idx >= p.batch_size) return;

    uint32_t local_id = tid.x;
    uint32_t threads = tg_size.x;
    uint32_t N = p.N;
    uint32_t L = p.L;
    uint64_t Q = p.Q;
    uint64_t mu = p.barrett_mu;
    uint32_t base_log = p.base_log;
    uint64_t base_mask = p.base_mask;

    // Shared memory
    threadgroup uint64_t* decomp_c0 = shared;
    threadgroup uint64_t* decomp_c1 = shared + L * N;
    threadgroup uint64_t* acc_c0 = shared + 2 * L * N;
    threadgroup uint64_t* acc_c1 = shared + 2 * L * N + N;

    // Per-batch RGSW
    uint32_t rgsw_stride = 2 * L * 2 * N;
    device const uint64_t* rgsw = rgsw_batch + batch_idx * rgsw_stride;

    device const uint64_t* c0 = rlwe_c0 + batch_idx * N;
    device const uint64_t* c1 = rlwe_c1 + batch_idx * N;
    device uint64_t* o_c0 = out_c0 + batch_idx * N;
    device uint64_t* o_c1 = out_c1 + batch_idx * N;

    // Decompose
    for (uint32_t i = local_id; i < N; i += threads) {
        uint64_t val_c0 = c0[i];
        uint64_t val_c1 = c1[i];

        for (uint32_t l = 0; l < L; ++l) {
            decomp_c0[l * N + i] = (val_c0 >> (l * base_log)) & base_mask;
            decomp_c1[l * N + i] = (val_c1 >> (l * base_log)) & base_mask;
        }

        acc_c0[i] = 0;
        acc_c1[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate
    for (uint32_t l = 0; l < L; ++l) {
        device const uint64_t* rgsw_c0_l_0 = rgsw + (0 * L * 2 * N) + (l * 2 * N);
        device const uint64_t* rgsw_c0_l_1 = rgsw + (0 * L * 2 * N) + (l * 2 * N) + N;
        device const uint64_t* rgsw_c1_l_0 = rgsw + (1 * L * 2 * N) + (l * 2 * N);
        device const uint64_t* rgsw_c1_l_1 = rgsw + (1 * L * 2 * N) + (l * 2 * N) + N;

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
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    for (uint32_t i = local_id; i < N; i += threads) {
        o_c0[i] = acc_c0[i];
        o_c1[i] = acc_c1[i];
    }
}
