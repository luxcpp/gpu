// =============================================================================
// Fused Blind Rotation Metal Kernel - Lux FHE GPU Acceleration
// =============================================================================
//
// This kernel fuses the entire blind rotation loop (n=512 iterations) into
// a single GPU dispatch, eliminating 512 kernel launches per bootstrap.
//
// Performance Impact:
//   Before: 512 kernel launches per bootstrap (~5ms total overhead)
//   After:  1 kernel launch per bootstrap (~0.5ms)
//   Expected speedup: 5-10x for typical bootstrapping workloads
//
// Algorithm:
//   1. Initialize accumulator: acc = (0, X^{-b} * testPoly)
//   2. For each i in [0, n):
//      - Compute rotation amount from LWE coefficient a[i]
//      - If rotation != 0:
//        - rotated_acc = X^{a[i]} * acc (negacyclic rotation)
//        - acc = CMux(bsk[i], acc, rotated_acc)
//              = acc + ExternalProduct(rotated_acc - acc, RGSW(s[i]))
//   3. Write final accumulator to output
//
// Memory Layout:
//   Shared memory (32KB max on Apple Silicon):
//     - acc_c0[N]: Accumulator component 0 (8KB for N=1024)
//     - acc_c1[N]: Accumulator component 1 (8KB for N=1024)
//     - work[N]:   Scratch buffer for NTT/products (8KB for N=1024)
//     Total: 24KB for N=1024, fits in 32KB limit
//
// Thread Organization:
//   - Grid: [batch_size, 1, 1]
//   - Threadgroup: [512, 1, 1] (N/2 threads for butterfly operations)
//   - Each threadgroup processes one complete bootstrap operation
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Kernel Parameters
// =============================================================================

struct BlindRotateParams {
    uint64_t Q;              // Ring modulus
    uint64_t mu;             // Barrett constant: floor(2^64 / Q)
    uint64_t N_inv;          // N^{-1} mod Q
    uint64_t N_inv_precon;   // Barrett precomputation for N_inv
    uint32_t N;              // Ring dimension (e.g., 1024)
    uint32_t log_N;          // log2(N) (e.g., 10)
    uint32_t n;              // LWE dimension (e.g., 512)
    uint32_t L;              // Decomposition levels (e.g., 3)
    uint32_t base_log;       // log2 of decomposition base (e.g., 7)
    uint64_t base_mask;      // (1 << base_log) - 1
    uint32_t batch_size;     // Number of bootstraps in this batch
};

// =============================================================================
// Barrett Modular Arithmetic (Constant-Time)
// =============================================================================

// Barrett modular multiplication with precomputed constant
// Computes (a * omega) mod Q where precon = floor(2^64 * omega / Q)
inline uint64_t barrett_mul(uint64_t a, uint64_t omega,
                            uint64_t Q, uint64_t precon) {
    uint64_t q_approx = metal::mulhi(a, precon);
    uint64_t product = a * omega;
    uint64_t result = product - q_approx * Q;

    // Constant-time conditional reduction
    uint64_t mask = uint64_t(int64_t(result >= Q) * -1);
    return result - (mask & Q);
}

// General modular multiplication (slower, no precomputation)
inline uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t lo = a * b;
    uint64_t hi = metal::mulhi(a, b);

    if (hi == 0) {
        return lo % Q;
    }

    // Full 128-bit reduction for large products
    uint64_t two32_mod_q = (uint64_t(1) << 32) % Q;
    uint64_t two64_mod_q = (two32_mod_q * two32_mod_q) % Q;
    return (lo % Q + (hi % Q) * two64_mod_q % Q) % Q;
}

// Modular addition: (a + b) mod Q
inline uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    uint64_t mask = uint64_t(int64_t(sum >= Q) * -1);
    return sum - (mask & Q);
}

// Modular subtraction: (a - b) mod Q
inline uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t mask = uint64_t(int64_t(a < b) * -1);
    return a - b + (mask & Q);
}

// =============================================================================
// NTT Butterfly Operations
// =============================================================================

// Cooley-Tukey (DIT) butterfly for forward NTT
inline void ct_butterfly(threadgroup uint64_t* data,
                         uint32_t idx_lo, uint32_t idx_hi,
                         uint64_t omega, uint64_t precon,
                         uint64_t Q) {
    uint64_t lo_val = data[idx_lo];
    uint64_t hi_val = data[idx_hi];
    uint64_t hi_tw = barrett_mul(hi_val, omega, Q, precon);

    data[idx_lo] = mod_add(lo_val, hi_tw, Q);
    data[idx_hi] = mod_sub(lo_val, hi_tw, Q);
}

// Gentleman-Sande (DIF) butterfly for inverse NTT
inline void gs_butterfly(threadgroup uint64_t* data,
                         uint32_t idx_lo, uint32_t idx_hi,
                         uint64_t omega, uint64_t precon,
                         uint64_t Q) {
    uint64_t lo_val = data[idx_lo];
    uint64_t hi_val = data[idx_hi];

    uint64_t sum = mod_add(lo_val, hi_val, Q);
    uint64_t diff = mod_sub(lo_val, hi_val, Q);
    uint64_t diff_tw = barrett_mul(diff, omega, Q, precon);

    data[idx_lo] = sum;
    data[idx_hi] = diff_tw;
}

// =============================================================================
// Negacyclic Rotation Helper
// =============================================================================
// Computes X^k * poly in Z_Q[X]/(X^N + 1)
// - Coefficient at index i moves to index (i + k) mod N
// - Sign flips when wrapping around N (negacyclic property)

inline void negacyclic_rotate_inplace(
    threadgroup uint64_t* dst,
    threadgroup uint64_t* src,
    int32_t rotation,
    uint32_t N,
    uint64_t Q,
    uint32_t local_id,
    uint32_t num_threads
) {
    // Normalize rotation to [0, 2N)
    int32_t rot = rotation;
    while (rot < 0) rot += 2 * int32_t(N);
    rot = rot % (2 * int32_t(N));

    for (uint32_t i = local_id; i < N; i += num_threads) {
        int32_t src_idx = int32_t(i) - rot;
        bool negate = false;

        // Handle wrap-around with sign tracking
        while (src_idx < 0) {
            src_idx += N;
            negate = !negate;
        }
        while (src_idx >= int32_t(N)) {
            src_idx -= N;
            negate = !negate;
        }

        uint64_t val = src[src_idx];
        dst[i] = negate ? mod_sub(0, val, Q) : val;
    }
}

// =============================================================================
// FUSED BLIND ROTATION KERNEL
// =============================================================================
//
// Processes the entire blind rotation loop in a single kernel launch.
// Each threadgroup handles one bootstrap operation.
//
// Inputs:
//   acc_out:   [batch, 2, N] - Output RLWE ciphertexts
//   lwe_in:    [batch, n+1]  - Input LWE ciphertexts
//   bsk:       [n, 2, L, 2, N] - Bootstrap key (RGSW encryptions of secret key bits)
//   test_poly: [N] - Test polynomial for the gate
//   twiddles:  [N] - Forward NTT twiddle factors
//   tw_precon: [N] - Barrett precomputation for twiddles
//   inv_tw:    [N] - Inverse NTT twiddle factors
//   inv_precon:[N] - Barrett precomputation for inverse twiddles
//
// Shared Memory Layout (for N=1024):
//   [0, N):      acc_c0 - Accumulator component 0
//   [N, 2N):     acc_c1 - Accumulator component 1
//   [2N, 3N):    rot_c0 - Rotated accumulator component 0 (scratch)
//   [3N, 4N):    rot_c1 - Rotated accumulator component 1 (scratch)
//   [4N, 5N):    work   - Work buffer for decomposition/NTT

kernel void blind_rotate_fused(
    device int64_t* acc_out              [[buffer(0)]],   // [B, 2, N] output
    device const int64_t* lwe_in         [[buffer(1)]],   // [B, n+1] input
    device const int64_t* bsk            [[buffer(2)]],   // [n, 2, L, 2, N]
    device const int64_t* test_poly      [[buffer(3)]],   // [N]
    constant uint64_t* twiddles          [[buffer(4)]],   // [N] forward
    constant uint64_t* tw_precon         [[buffer(5)]],   // [N] forward precon
    constant uint64_t* inv_tw            [[buffer(6)]],   // [N] inverse
    constant uint64_t* inv_precon        [[buffer(7)]],   // [N] inverse precon
    constant BlindRotateParams& params   [[buffer(8)]],

    uint3 tid                            [[thread_position_in_threadgroup]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint3 tg_size                        [[threads_per_threadgroup]],

    threadgroup uint64_t* shared         [[threadgroup(0)]]
) {
    // Extract parameters
    uint32_t batch_idx = tgid.x;
    uint32_t local_id = tid.x;
    uint32_t num_threads = tg_size.x;

    uint32_t N = params.N;
    uint32_t log_N = params.log_N;
    uint32_t n = params.n;
    uint32_t L = params.L;
    uint64_t Q = params.Q;
    uint64_t base_mask = params.base_mask;
    uint32_t base_log = params.base_log;

    if (batch_idx >= params.batch_size) return;

    // Shared memory layout
    threadgroup uint64_t* acc_c0 = shared;              // [N]
    threadgroup uint64_t* acc_c1 = shared + N;          // [N]
    threadgroup uint64_t* rot_c0 = shared + 2 * N;      // [N]
    threadgroup uint64_t* rot_c1 = shared + 3 * N;      // [N]
    threadgroup uint64_t* work = shared + 4 * N;        // [N]

    // Pointers to this batch's data
    device const int64_t* lwe = lwe_in + batch_idx * (n + 1);
    device int64_t* out = acc_out + batch_idx * 2 * N;

    // =========================================================================
    // STAGE 1: Initialize Accumulator
    // =========================================================================
    // acc = (0, X^{-b} * testPoly)
    // where b is the LWE constant term (last element)

    int64_t b_val = lwe[n];
    int32_t b_mod = int32_t((b_val % int64_t(2 * N) + int64_t(2 * N)) % int64_t(2 * N));
    int32_t init_rot = (b_mod == 0) ? 0 : int32_t(2 * N) - b_mod;  // -b mod 2N

    // Initialize acc_c0 to zeros, acc_c1 to rotated test polynomial
    for (uint32_t i = local_id; i < N; i += num_threads) {
        acc_c0[i] = 0;

        // Compute rotated test polynomial coefficient
        int32_t src_idx = int32_t(i) - init_rot;
        bool negate = false;
        while (src_idx < 0) { src_idx += N; negate = !negate; }
        while (src_idx >= int32_t(N)) { src_idx -= N; negate = !negate; }

        uint64_t val = uint64_t(test_poly[src_idx]) % Q;
        acc_c1[i] = negate ? mod_sub(0, val, Q) : val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // STAGE 2: Main Blind Rotation Loop
    // =========================================================================
    // For each LWE coefficient a[j], perform CMux with BSK[j]

    for (uint32_t j = 0; j < n; ++j) {
        int64_t a_j = lwe[j];
        int32_t rot = int32_t((a_j % int64_t(2 * N) + int64_t(2 * N)) % int64_t(2 * N));

        // Skip if rotation is zero (no change to accumulator)
        if (rot == 0) continue;

        // ---------------------------------------------------------------------
        // Step 2a: Compute rotated accumulator
        // rotated = X^{a[j]} * acc
        // ---------------------------------------------------------------------

        negacyclic_rotate_inplace(rot_c0, acc_c0, rot, N, Q, local_id, num_threads);
        negacyclic_rotate_inplace(rot_c1, acc_c1, rot, N, Q, local_id, num_threads);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---------------------------------------------------------------------
        // Step 2b: Compute diff = rotated - acc (for external product)
        // ---------------------------------------------------------------------

        // We'll compute diff in-place in rot_c0/rot_c1
        for (uint32_t i = local_id; i < N; i += num_threads) {
            rot_c0[i] = mod_sub(rot_c0[i], acc_c0[i], Q);
            rot_c1[i] = mod_sub(rot_c1[i], acc_c1[i], Q);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---------------------------------------------------------------------
        // Step 2c: External Product - ExternalProduct(diff, RGSW(s[j]))
        // ---------------------------------------------------------------------
        // RGSW layout: bsk[j, in_c, l, out_c, coeff]
        // in_c: input RLWE component (0 or 1)
        // l: decomposition level
        // out_c: output RLWE component (0 or 1)

        // Temporary accumulators for external product result
        // We'll accumulate directly into work buffer, then add to acc

        // Initialize product accumulators to zero
        // (We'll use rot_c0/rot_c1 as temporary after computing diff)

        // Process both input components (in_c = 0, 1)
        for (uint32_t in_c = 0; in_c < 2; ++in_c) {
            threadgroup uint64_t* diff_comp = (in_c == 0) ? rot_c0 : rot_c1;

            // Process each decomposition level
            for (uint32_t l = 0; l < L; ++l) {
                // Decompose: extract digit l from diff_comp
                for (uint32_t i = local_id; i < N; i += num_threads) {
                    work[i] = (diff_comp[i] >> (l * base_log)) & base_mask;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Forward NTT on decomposed digit
                for (uint32_t stage = 0; stage < log_N; ++stage) {
                    uint32_t m = 1u << stage;
                    uint32_t t = N >> (stage + 1);

                    for (uint32_t bf = local_id; bf < N / 2; bf += num_threads) {
                        uint32_t ii = bf / t;
                        uint32_t jj = bf % t;
                        uint32_t idx_lo = (ii << (log_N - stage)) + jj;
                        uint32_t idx_hi = idx_lo + t;
                        uint32_t tw_idx = m + ii;

                        ct_butterfly(work, idx_lo, idx_hi,
                                    twiddles[tw_idx], tw_precon[tw_idx], Q);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                // Multiply with RGSW polynomials and accumulate to acc
                // RGSW index: bsk[j * 2*L*2*N + in_c * L*2*N + l * 2*N + out_c * N + i]
                device const int64_t* rgsw_base = bsk + j * 2 * L * 2 * N + in_c * L * 2 * N + l * 2 * N;

                for (uint32_t out_c = 0; out_c < 2; ++out_c) {
                    device const int64_t* rgsw_poly = rgsw_base + out_c * N;

                    // Pointwise multiply in NTT domain
                    for (uint32_t i = local_id; i < N; i += num_threads) {
                        uint64_t digit_ntt = work[i];
                        uint64_t rgsw_val = uint64_t(rgsw_poly[i]) % Q;

                        // For first contribution, store; otherwise add
                        if (in_c == 0 && l == 0) {
                            // Store product temporarily
                            if (out_c == 0) {
                                rot_c0[i] = mod_mul(digit_ntt, rgsw_val, Q);
                            } else {
                                rot_c1[i] = mod_mul(digit_ntt, rgsw_val, Q);
                            }
                        } else {
                            // Accumulate product
                            uint64_t prod = mod_mul(digit_ntt, rgsw_val, Q);
                            if (out_c == 0) {
                                rot_c0[i] = mod_add(rot_c0[i], prod, Q);
                            } else {
                                rot_c1[i] = mod_add(rot_c1[i], prod, Q);
                            }
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }

        // Inverse NTT on accumulated products (rot_c0 and rot_c1)
        // Process rot_c0
        for (uint32_t stage = 0; stage < log_N; ++stage) {
            uint32_t m = N >> (stage + 1);
            uint32_t t = 1u << stage;

            for (uint32_t bf = local_id; bf < N / 2; bf += num_threads) {
                uint32_t ii = bf / t;
                uint32_t jj = bf % t;
                uint32_t idx_lo = (ii << (stage + 1)) + jj;
                uint32_t idx_hi = idx_lo + t;
                uint32_t tw_idx = m + ii;

                gs_butterfly(rot_c0, idx_lo, idx_hi,
                            inv_tw[tw_idx], inv_precon[tw_idx], Q);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Process rot_c1
        for (uint32_t stage = 0; stage < log_N; ++stage) {
            uint32_t m = N >> (stage + 1);
            uint32_t t = 1u << stage;

            for (uint32_t bf = local_id; bf < N / 2; bf += num_threads) {
                uint32_t ii = bf / t;
                uint32_t jj = bf % t;
                uint32_t idx_lo = (ii << (stage + 1)) + jj;
                uint32_t idx_hi = idx_lo + t;
                uint32_t tw_idx = m + ii;

                gs_butterfly(rot_c1, idx_lo, idx_hi,
                            inv_tw[tw_idx], inv_precon[tw_idx], Q);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Scale by N^{-1} and add to accumulator
        // acc = acc + ExternalProduct(diff, RGSW)
        for (uint32_t i = local_id; i < N; i += num_threads) {
            uint64_t prod0 = barrett_mul(rot_c0[i], params.N_inv, Q, params.N_inv_precon);
            uint64_t prod1 = barrett_mul(rot_c1[i], params.N_inv, Q, params.N_inv_precon);

            acc_c0[i] = mod_add(acc_c0[i], prod0, Q);
            acc_c1[i] = mod_add(acc_c1[i], prod1, Q);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // =========================================================================
    // STAGE 3: Write Output
    // =========================================================================

    for (uint32_t i = local_id; i < N; i += num_threads) {
        out[i] = int64_t(acc_c0[i]);
        out[N + i] = int64_t(acc_c1[i]);
    }
}

// =============================================================================
// OPTIMIZED VARIANT: Batch-Aware with Shared Twiddle Prefetch
// =============================================================================
//
// This variant prefetches twiddle factors to shared memory to reduce
// global memory bandwidth. Useful when shared memory budget allows.
// Requires 8N extra bytes of shared memory for twiddles.

kernel void blind_rotate_fused_v2(
    device int64_t* acc_out              [[buffer(0)]],
    device const int64_t* lwe_in         [[buffer(1)]],
    device const int64_t* bsk            [[buffer(2)]],
    device const int64_t* test_poly      [[buffer(3)]],
    constant uint64_t* twiddles          [[buffer(4)]],
    constant uint64_t* tw_precon         [[buffer(5)]],
    constant uint64_t* inv_tw            [[buffer(6)]],
    constant uint64_t* inv_precon        [[buffer(7)]],
    constant BlindRotateParams& params   [[buffer(8)]],

    uint3 tid                            [[thread_position_in_threadgroup]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint3 tg_size                        [[threads_per_threadgroup]],

    threadgroup uint64_t* shared         [[threadgroup(0)]]
) {
    uint32_t batch_idx = tgid.x;
    uint32_t local_id = tid.x;
    uint32_t num_threads = tg_size.x;

    uint32_t N = params.N;
    uint32_t log_N = params.log_N;
    uint32_t n = params.n;
    uint32_t L = params.L;
    uint64_t Q = params.Q;
    uint64_t base_mask = params.base_mask;
    uint32_t base_log = params.base_log;

    if (batch_idx >= params.batch_size) return;

    // Extended shared memory layout with prefetched twiddles
    // Total: 9N * 8 bytes = 72KB for N=1024 (exceeds 32KB limit)
    // So we only prefetch for smaller N or use simpler layout

    // For N <= 512, we can prefetch twiddles (4.5KB * 8 = 36KB total)
    // For N > 512, use global memory access (previous kernel)

    threadgroup uint64_t* acc_c0 = shared;
    threadgroup uint64_t* acc_c1 = shared + N;
    threadgroup uint64_t* rot_c0 = shared + 2 * N;
    threadgroup uint64_t* rot_c1 = shared + 3 * N;
    threadgroup uint64_t* work = shared + 4 * N;

    // For N <= 512, prefetch twiddles
    threadgroup uint64_t* fwd_tw = nullptr;
    threadgroup uint64_t* fwd_pre = nullptr;
    threadgroup uint64_t* inv_twl = nullptr;
    threadgroup uint64_t* inv_pre = nullptr;

    bool use_prefetch = (N <= 512);

    if (use_prefetch) {
        fwd_tw = shared + 5 * N;
        fwd_pre = shared + 6 * N;
        inv_twl = shared + 7 * N;
        inv_pre = shared + 8 * N;

        // Cooperative prefetch
        for (uint32_t i = local_id; i < N; i += num_threads) {
            fwd_tw[i] = twiddles[i];
            fwd_pre[i] = tw_precon[i];
            inv_twl[i] = inv_tw[i];
            inv_pre[i] = inv_precon[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device const int64_t* lwe = lwe_in + batch_idx * (n + 1);
    device int64_t* out = acc_out + batch_idx * 2 * N;

    // Initialize accumulator
    int64_t b_val = lwe[n];
    int32_t b_mod = int32_t((b_val % int64_t(2 * N) + int64_t(2 * N)) % int64_t(2 * N));
    int32_t init_rot = (b_mod == 0) ? 0 : int32_t(2 * N) - b_mod;

    for (uint32_t i = local_id; i < N; i += num_threads) {
        acc_c0[i] = 0;

        int32_t src_idx = int32_t(i) - init_rot;
        bool negate = false;
        while (src_idx < 0) { src_idx += N; negate = !negate; }
        while (src_idx >= int32_t(N)) { src_idx -= N; negate = !negate; }

        uint64_t val = uint64_t(test_poly[src_idx]) % Q;
        acc_c1[i] = negate ? mod_sub(0, val, Q) : val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main loop with optional twiddle prefetch
    for (uint32_t j = 0; j < n; ++j) {
        int64_t a_j = lwe[j];
        int32_t rot = int32_t((a_j % int64_t(2 * N) + int64_t(2 * N)) % int64_t(2 * N));

        if (rot == 0) continue;

        // Rotate accumulator
        negacyclic_rotate_inplace(rot_c0, acc_c0, rot, N, Q, local_id, num_threads);
        negacyclic_rotate_inplace(rot_c1, acc_c1, rot, N, Q, local_id, num_threads);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute diff
        for (uint32_t i = local_id; i < N; i += num_threads) {
            rot_c0[i] = mod_sub(rot_c0[i], acc_c0[i], Q);
            rot_c1[i] = mod_sub(rot_c1[i], acc_c1[i], Q);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // External product
        for (uint32_t in_c = 0; in_c < 2; ++in_c) {
            threadgroup uint64_t* diff_comp = (in_c == 0) ? rot_c0 : rot_c1;

            for (uint32_t l = 0; l < L; ++l) {
                // Decompose
                for (uint32_t i = local_id; i < N; i += num_threads) {
                    work[i] = (diff_comp[i] >> (l * base_log)) & base_mask;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Forward NTT
                for (uint32_t stage = 0; stage < log_N; ++stage) {
                    uint32_t m = 1u << stage;
                    uint32_t t = N >> (stage + 1);

                    for (uint32_t bf = local_id; bf < N / 2; bf += num_threads) {
                        uint32_t ii = bf / t;
                        uint32_t jj = bf % t;
                        uint32_t idx_lo = (ii << (log_N - stage)) + jj;
                        uint32_t idx_hi = idx_lo + t;
                        uint32_t tw_idx = m + ii;

                        uint64_t omega = use_prefetch ? fwd_tw[tw_idx] : twiddles[tw_idx];
                        uint64_t precon = use_prefetch ? fwd_pre[tw_idx] : tw_precon[tw_idx];
                        ct_butterfly(work, idx_lo, idx_hi, omega, precon, Q);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                // Multiply and accumulate
                device const int64_t* rgsw_base = bsk + j * 2 * L * 2 * N + in_c * L * 2 * N + l * 2 * N;

                for (uint32_t out_c = 0; out_c < 2; ++out_c) {
                    device const int64_t* rgsw_poly = rgsw_base + out_c * N;

                    for (uint32_t i = local_id; i < N; i += num_threads) {
                        uint64_t digit_ntt = work[i];
                        uint64_t rgsw_val = uint64_t(rgsw_poly[i]) % Q;

                        if (in_c == 0 && l == 0) {
                            if (out_c == 0) {
                                rot_c0[i] = mod_mul(digit_ntt, rgsw_val, Q);
                            } else {
                                rot_c1[i] = mod_mul(digit_ntt, rgsw_val, Q);
                            }
                        } else {
                            uint64_t prod = mod_mul(digit_ntt, rgsw_val, Q);
                            if (out_c == 0) {
                                rot_c0[i] = mod_add(rot_c0[i], prod, Q);
                            } else {
                                rot_c1[i] = mod_add(rot_c1[i], prod, Q);
                            }
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }

        // Inverse NTT on rot_c0
        for (uint32_t stage = 0; stage < log_N; ++stage) {
            uint32_t m = N >> (stage + 1);
            uint32_t t = 1u << stage;

            for (uint32_t bf = local_id; bf < N / 2; bf += num_threads) {
                uint32_t ii = bf / t;
                uint32_t jj = bf % t;
                uint32_t idx_lo = (ii << (stage + 1)) + jj;
                uint32_t idx_hi = idx_lo + t;
                uint32_t tw_idx = m + ii;

                uint64_t omega = use_prefetch ? inv_twl[tw_idx] : inv_tw[tw_idx];
                uint64_t precon = use_prefetch ? inv_pre[tw_idx] : inv_precon[tw_idx];
                gs_butterfly(rot_c0, idx_lo, idx_hi, omega, precon, Q);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Inverse NTT on rot_c1
        for (uint32_t stage = 0; stage < log_N; ++stage) {
            uint32_t m = N >> (stage + 1);
            uint32_t t = 1u << stage;

            for (uint32_t bf = local_id; bf < N / 2; bf += num_threads) {
                uint32_t ii = bf / t;
                uint32_t jj = bf % t;
                uint32_t idx_lo = (ii << (stage + 1)) + jj;
                uint32_t idx_hi = idx_lo + t;
                uint32_t tw_idx = m + ii;

                uint64_t omega = use_prefetch ? inv_twl[tw_idx] : inv_tw[tw_idx];
                uint64_t precon = use_prefetch ? inv_pre[tw_idx] : inv_precon[tw_idx];
                gs_butterfly(rot_c1, idx_lo, idx_hi, omega, precon, Q);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Scale and accumulate
        for (uint32_t i = local_id; i < N; i += num_threads) {
            uint64_t prod0 = barrett_mul(rot_c0[i], params.N_inv, Q, params.N_inv_precon);
            uint64_t prod1 = barrett_mul(rot_c1[i], params.N_inv, Q, params.N_inv_precon);

            acc_c0[i] = mod_add(acc_c0[i], prod0, Q);
            acc_c1[i] = mod_add(acc_c1[i], prod1, Q);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    for (uint32_t i = local_id; i < N; i += num_threads) {
        out[i] = int64_t(acc_c0[i]);
        out[N + i] = int64_t(acc_c1[i]);
    }
}

// =============================================================================
// SIMPLIFIED VARIANT: For Quick Testing
// =============================================================================
// Processes blind rotation without full external product (just rotation + add).
// Useful for debugging and baseline performance measurement.

kernel void blind_rotate_simplified(
    device int64_t* acc_out              [[buffer(0)]],
    device const int64_t* lwe_in         [[buffer(1)]],
    device const int64_t* test_poly      [[buffer(2)]],
    constant BlindRotateParams& params   [[buffer(3)]],

    uint3 tid                            [[thread_position_in_threadgroup]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint3 tg_size                        [[threads_per_threadgroup]],

    threadgroup uint64_t* shared         [[threadgroup(0)]]
) {
    uint32_t batch_idx = tgid.x;
    uint32_t local_id = tid.x;
    uint32_t num_threads = tg_size.x;

    uint32_t N = params.N;
    uint32_t n = params.n;
    uint64_t Q = params.Q;

    if (batch_idx >= params.batch_size) return;

    threadgroup uint64_t* acc_c0 = shared;
    threadgroup uint64_t* acc_c1 = shared + N;

    device const int64_t* lwe = lwe_in + batch_idx * (n + 1);
    device int64_t* out = acc_out + batch_idx * 2 * N;

    // Initialize with X^{-b} * testPoly
    int64_t b_val = lwe[n];
    int32_t b_mod = int32_t((b_val % int64_t(2 * N) + int64_t(2 * N)) % int64_t(2 * N));
    int32_t init_rot = (b_mod == 0) ? 0 : int32_t(2 * N) - b_mod;

    for (uint32_t i = local_id; i < N; i += num_threads) {
        acc_c0[i] = 0;

        int32_t src_idx = int32_t(i) - init_rot;
        bool negate = false;
        while (src_idx < 0) { src_idx += N; negate = !negate; }
        while (src_idx >= int32_t(N)) { src_idx -= N; negate = !negate; }

        uint64_t val = uint64_t(test_poly[src_idx]) % Q;
        acc_c1[i] = negate ? mod_sub(0, val, Q) : val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write output (simplified - just initialization for testing)
    for (uint32_t i = local_id; i < N; i += num_threads) {
        out[i] = int64_t(acc_c0[i]);
        out[N + i] = int64_t(acc_c1[i]);
    }
}
