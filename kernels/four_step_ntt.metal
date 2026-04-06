// =============================================================================
// Four-Step NTT Optimized for Apple Metal Threadgroup Memory and SIMDgroup
// =============================================================================
//
// This implements the Four-Step NTT algorithm specifically tuned for:
// - 32KB Metal threadgroup memory (vs 48KB CUDA shared memory)
// - 32-lane SIMDgroup operations via threadgroup memory
// - Coalesced memory access patterns for unified memory
// - Integer-only arithmetic for FHE determinism
//
// Four-Step Algorithm for N = N1 * N2:
//   1. N2 parallel column NTTs of size N1
//   2. Twiddle multiplication by omega^(i*j)
//   3. Matrix transpose
//   4. N1 parallel row NTTs of size N2
//
// NOTE: Metal's simd_shuffle does NOT support uint64_t. Instead, we use
// threadgroup memory for all inter-lane communication, which is still very
// fast (~20ns latency, ~3TB/s bandwidth per SIMD on M3).
//
// See patent: PAT-FHE-010-four-step-ntt-metal.md
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-2-Clause
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Constants
// =============================================================================

constant uint32_t SIMD_SIZE = 32;            // Metal SIMDgroup size
constant uint32_t MAX_TILE_SIZE = 4096;      // 32KB / 8 bytes = 4096 uint64_t
constant uint32_t MAX_TILE_DIM = 64;         // sqrt(4096) = 64 for square tiles

// =============================================================================
// Parameters Structure
// =============================================================================

struct FourStepParams {
    uint64_t Q;             // Prime modulus
    uint64_t mu;            // Barrett constant: floor(2^64 / Q)
    uint64_t N_inv;         // N^{-1} mod Q for inverse NTT
    uint64_t N_inv_precon;  // Barrett precomputation for N_inv
    uint32_t N;             // Total ring dimension
    uint32_t N1;            // Column dimension for Four-Step
    uint32_t N2;            // Row dimension for Four-Step
    uint32_t log_N1;        // log2(N1)
    uint32_t log_N2;        // log2(N2)
    uint32_t tile_stride;   // Padded stride
    uint32_t batch_size;    // Number of polynomials to process
};

// =============================================================================
// Modular Arithmetic (Integer-Only for Determinism)
// =============================================================================

/**
 * @brief Barrett modular multiplication
 *
 * Computes (a * b) mod Q using Barrett reduction.
 * Requires: a, b < Q and Q < 2^62
 *
 * The precomputed constant precon = floor(2^64 * b / Q) allows
 * faster approximate division.
 */
inline uint64_t barrett_mul_precon(uint64_t a, uint64_t b, uint64_t Q, uint64_t precon) {
    // Approximate quotient: q ≈ (a * precon) >> 64
    uint64_t q_approx = metal::mulhi(a, precon);

    // Compute a * b - q_approx * Q
    uint64_t product = a * b;
    uint64_t result = product - q_approx * Q;

    // Conditional reduction (result may be in [0, 2Q))
    return (result >= Q) ? (result - Q) : result;
}

/**
 * @brief Simple Barrett multiplication without precomputation
 *
 * Used when precomputed constants are not available.
 */
inline uint64_t barrett_mul(uint64_t a, uint64_t b, uint64_t Q, uint64_t mu) {
    uint64_t lo = a * b;
    uint64_t q = metal::mulhi(lo, mu);
    uint64_t result = lo - q * Q;
    return (result >= Q) ? (result - Q) : result;
}

/**
 * @brief Modular addition: (a + b) mod Q
 */
inline uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    return (sum >= Q) ? (sum - Q) : sum;
}

/**
 * @brief Modular subtraction: (a - b) mod Q
 */
inline uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    return (a >= b) ? (a - b) : (a + Q - b);
}

// =============================================================================
// Cooley-Tukey Butterfly (for forward NTT)
// =============================================================================

/**
 * @brief Single Cooley-Tukey butterfly operation
 *
 * CT: (lo, hi) -> (lo + hi*tw, lo - hi*tw)
 */
inline void ct_butterfly(
    thread uint64_t& lo,
    thread uint64_t& hi,
    uint64_t tw,
    uint64_t tw_pre,
    uint64_t Q
) {
    uint64_t hi_tw = barrett_mul_precon(hi, tw, Q, tw_pre);
    uint64_t new_lo = mod_add(lo, hi_tw, Q);
    uint64_t new_hi = mod_sub(lo, hi_tw, Q);
    lo = new_lo;
    hi = new_hi;
}

/**
 * @brief Single Gentleman-Sande butterfly operation (for inverse NTT)
 *
 * GS: (lo, hi) -> (lo + hi, (lo - hi) * tw)
 */
inline void gs_butterfly(
    thread uint64_t& lo,
    thread uint64_t& hi,
    uint64_t tw,
    uint64_t tw_pre,
    uint64_t Q
) {
    uint64_t sum = mod_add(lo, hi, Q);
    uint64_t diff = mod_sub(lo, hi, Q);
    uint64_t diff_tw = barrett_mul_precon(diff, tw, Q, tw_pre);
    lo = sum;
    hi = diff_tw;
}

// =============================================================================
// In-Threadgroup NTT (for small dimensions that fit in threadgroup memory)
// =============================================================================

/**
 * @brief Forward NTT on data in threadgroup memory
 *
 * Performs complete NTT on column data using threadgroup memory.
 * Each thread handles multiple elements as needed.
 *
 * @param shared Pointer to threadgroup memory containing column data
 * @param stride Stride between consecutive elements (for column access)
 * @param N Size of NTT
 * @param log_N log2(N)
 * @param thread_idx Thread index within group
 * @param num_threads Number of threads in group
 * @param twiddles Forward twiddle factors
 * @param twiddle_precon Barrett precomputed constants
 * @param Q Prime modulus
 */
inline void threadgroup_ntt_forward(
    threadgroup uint64_t* shared,
    uint32_t stride,
    uint32_t N,
    uint32_t log_N,
    uint32_t thread_idx,
    uint32_t num_threads,
    constant uint64_t* twiddles,
    constant uint64_t* twiddle_precon,
    uint64_t Q
) {
    // Cooley-Tukey DIT NTT
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = 1u << stage;           // Number of groups
        uint32_t t = N >> (stage + 1);      // Half-size of each group

        uint32_t num_butterflies = N >> 1;
        uint32_t butterflies_per_thread = (num_butterflies + num_threads - 1) / num_threads;

        for (uint32_t b = 0; b < butterflies_per_thread; ++b) {
            uint32_t butterfly_idx = thread_idx + b * num_threads;
            if (butterfly_idx >= num_butterflies) break;

            uint32_t group = butterfly_idx / t;
            uint32_t j = butterfly_idx % t;

            uint32_t idx_lo = (group * 2 * t + j) * stride;
            uint32_t idx_hi = idx_lo + t * stride;

            uint32_t tw_idx = m + group;
            uint64_t tw = twiddles[tw_idx];
            uint64_t tw_pre = twiddle_precon[tw_idx];

            uint64_t lo = shared[idx_lo];
            uint64_t hi = shared[idx_hi];

            ct_butterfly(lo, hi, tw, tw_pre, Q);

            shared[idx_lo] = lo;
            shared[idx_hi] = hi;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/**
 * @brief Inverse NTT on data in threadgroup memory (Gentleman-Sande)
 */
inline void threadgroup_ntt_inverse(
    threadgroup uint64_t* shared,
    uint32_t stride,
    uint32_t N,
    uint32_t log_N,
    uint32_t thread_idx,
    uint32_t num_threads,
    constant uint64_t* twiddles,
    constant uint64_t* twiddle_precon,
    uint64_t Q
) {
    // Gentleman-Sande DIF INTT
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = N >> (stage + 1);      // Number of groups
        uint32_t t = 1u << stage;           // Half-size of each group

        uint32_t num_butterflies = N >> 1;
        uint32_t butterflies_per_thread = (num_butterflies + num_threads - 1) / num_threads;

        for (uint32_t b = 0; b < butterflies_per_thread; ++b) {
            uint32_t butterfly_idx = thread_idx + b * num_threads;
            if (butterfly_idx >= num_butterflies) break;

            uint32_t group = butterfly_idx / t;
            uint32_t j = butterfly_idx % t;

            uint32_t idx_lo = (group * 2 * t + j) * stride;
            uint32_t idx_hi = idx_lo + t * stride;

            uint32_t tw_idx = m + group;
            uint64_t tw = twiddles[tw_idx];
            uint64_t tw_pre = twiddle_precon[tw_idx];

            uint64_t lo = shared[idx_lo];
            uint64_t hi = shared[idx_hi];

            gs_butterfly(lo, hi, tw, tw_pre, Q);

            shared[idx_lo] = lo;
            shared[idx_hi] = hi;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// =============================================================================
// Step 1: Column NTTs (Forward)
// =============================================================================

/**
 * @brief Column NTT kernel for Four-Step algorithm
 *
 * Processes N2 parallel column NTTs of size N1.
 * Each threadgroup handles one tile.
 *
 * Thread organization:
 * - Threadgroup: up to 1024 threads
 * - Threadgroup memory: tile_N1 x tile_stride elements (with padding)
 */
kernel void four_step_column_ntt(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* twiddles [[buffer(1)]],
    constant uint64_t* twiddle_precon [[buffer(2)]],
    constant FourStepParams& params [[buffer(3)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_tg [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]],
    threadgroup uint64_t* shared [[threadgroup(0)]]
) {
    uint32_t thread_idx = thread_pos_in_tg.x + thread_pos_in_tg.y * tg_size.x + thread_pos_in_tg.z * tg_size.x * tg_size.y;
    uint32_t threadgroup_size = tg_size.x * tg_size.y * tg_size.z;

    uint32_t N1 = params.N1;
    uint32_t N2 = params.N2;
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint32_t batch_idx = tg_pos.z;
    uint32_t tile_row = tg_pos.y;
    uint32_t tile_col = tg_pos.x;
    uint32_t tile_stride = params.tile_stride;

    // Tile dimensions
    uint32_t TILE_N1 = min(N1, MAX_TILE_DIM);
    uint32_t TILE_N2 = min(N2, MAX_TILE_DIM);

    // Global offset for this batch and tile
    device uint64_t* batch_data = data + batch_idx * N;

    // Phase 1: Cooperative load tile into shared memory
    uint32_t elements_per_thread = (TILE_N1 * TILE_N2 + threadgroup_size - 1) / threadgroup_size;

    for (uint32_t e = 0; e < elements_per_thread; ++e) {
        uint32_t local_idx = thread_idx + e * threadgroup_size;
        if (local_idx >= TILE_N1 * TILE_N2) break;

        uint32_t local_row = local_idx / TILE_N2;
        uint32_t local_col = local_idx % TILE_N2;

        uint32_t global_row = tile_row * TILE_N1 + local_row;
        uint32_t global_col = tile_col * TILE_N2 + local_col;

        if (global_row < N1 && global_col < N2) {
            uint32_t global_idx = global_row * N2 + global_col;
            shared[local_row * tile_stride + local_col] = batch_data[global_idx];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Column NTTs
    // Each column is processed by all threads cooperatively
    uint32_t log_N1 = params.log_N1;

    for (uint32_t col = 0; col < TILE_N2; ++col) {
        // NTT on column `col` using all threads
        threadgroup_ntt_forward(
            shared + col,           // Start of column
            tile_stride,            // Stride = tile_stride (to next row)
            TILE_N1,                // Size
            log_N1,
            thread_idx,
            threadgroup_size,
            twiddles,
            twiddle_precon,
            Q
        );
    }

    // Phase 3: Write back to global memory
    for (uint32_t e = 0; e < elements_per_thread; ++e) {
        uint32_t local_idx = thread_idx + e * threadgroup_size;
        if (local_idx >= TILE_N1 * TILE_N2) break;

        uint32_t local_row = local_idx / TILE_N2;
        uint32_t local_col = local_idx % TILE_N2;

        uint32_t global_row = tile_row * TILE_N1 + local_row;
        uint32_t global_col = tile_col * TILE_N2 + local_col;

        if (global_row < N1 && global_col < N2) {
            uint32_t global_idx = global_row * N2 + global_col;
            batch_data[global_idx] = shared[local_row * tile_stride + local_col];
        }
    }
}

// =============================================================================
// Step 1: Column NTTs (Inverse)
// =============================================================================

kernel void four_step_column_intt(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* twiddles [[buffer(1)]],
    constant uint64_t* twiddle_precon [[buffer(2)]],
    constant FourStepParams& params [[buffer(3)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_tg [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]],
    threadgroup uint64_t* shared [[threadgroup(0)]]
) {
    uint32_t thread_idx = thread_pos_in_tg.x + thread_pos_in_tg.y * tg_size.x + thread_pos_in_tg.z * tg_size.x * tg_size.y;
    uint32_t threadgroup_size = tg_size.x * tg_size.y * tg_size.z;

    uint32_t N1 = params.N1;
    uint32_t N2 = params.N2;
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint32_t batch_idx = tg_pos.z;
    uint32_t tile_row = tg_pos.y;
    uint32_t tile_col = tg_pos.x;
    uint32_t tile_stride = params.tile_stride;

    uint32_t TILE_N1 = min(N1, MAX_TILE_DIM);
    uint32_t TILE_N2 = min(N2, MAX_TILE_DIM);

    device uint64_t* batch_data = data + batch_idx * N;

    // Load
    uint32_t elements_per_thread = (TILE_N1 * TILE_N2 + threadgroup_size - 1) / threadgroup_size;

    for (uint32_t e = 0; e < elements_per_thread; ++e) {
        uint32_t local_idx = thread_idx + e * threadgroup_size;
        if (local_idx >= TILE_N1 * TILE_N2) break;

        uint32_t local_row = local_idx / TILE_N2;
        uint32_t local_col = local_idx % TILE_N2;

        uint32_t global_row = tile_row * TILE_N1 + local_row;
        uint32_t global_col = tile_col * TILE_N2 + local_col;

        if (global_row < N1 && global_col < N2) {
            uint32_t global_idx = global_row * N2 + global_col;
            shared[local_row * tile_stride + local_col] = batch_data[global_idx];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inverse column NTTs
    uint32_t log_N1 = params.log_N1;

    for (uint32_t col = 0; col < TILE_N2; ++col) {
        threadgroup_ntt_inverse(
            shared + col,
            tile_stride,
            TILE_N1,
            log_N1,
            thread_idx,
            threadgroup_size,
            twiddles,
            twiddle_precon,
            Q
        );
    }

    // Write back
    for (uint32_t e = 0; e < elements_per_thread; ++e) {
        uint32_t local_idx = thread_idx + e * threadgroup_size;
        if (local_idx >= TILE_N1 * TILE_N2) break;

        uint32_t local_row = local_idx / TILE_N2;
        uint32_t local_col = local_idx % TILE_N2;

        uint32_t global_row = tile_row * TILE_N1 + local_row;
        uint32_t global_col = tile_col * TILE_N2 + local_col;

        if (global_row < N1 && global_col < N2) {
            uint32_t global_idx = global_row * N2 + global_col;
            batch_data[global_idx] = shared[local_row * tile_stride + local_col];
        }
    }
}

// =============================================================================
// Step 2+3: Fused Twiddle Multiplication and Transpose
// =============================================================================

/**
 * @brief Fused twiddle multiplication and transpose kernel
 *
 * Combines Step 2 (multiply by omega^(i*j)) with Step 3 (transpose).
 * Uses bank-conflict-free shared memory access with padding.
 *
 * Input: N1 x N2 matrix
 * Output: N2 x N1 matrix (transposed)
 */
kernel void four_step_twiddle_transpose(
    device uint64_t* output [[buffer(0)]],
    device const uint64_t* input [[buffer(1)]],
    constant uint64_t* twiddles [[buffer(2)]],
    constant uint64_t* twiddle_precon [[buffer(3)]],
    constant FourStepParams& params [[buffer(4)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_tg [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]],
    threadgroup uint64_t* shared [[threadgroup(0)]]
) {
    uint32_t thread_idx = thread_pos_in_tg.x + thread_pos_in_tg.y * tg_size.x + thread_pos_in_tg.z * tg_size.x * tg_size.y;
    uint32_t threadgroup_size = tg_size.x * tg_size.y * tg_size.z;

    uint32_t N1 = params.N1;
    uint32_t N2 = params.N2;
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint32_t batch_idx = tg_pos.z;
    uint32_t tile_row = tg_pos.y;
    uint32_t tile_col = tg_pos.x;
    uint32_t tile_stride = params.tile_stride;

    uint32_t TILE_DIM = MAX_TILE_DIM;

    device const uint64_t* batch_input = input + batch_idx * N;
    device uint64_t* batch_output = output + batch_idx * N;

    uint32_t elements_per_thread = (TILE_DIM * TILE_DIM + threadgroup_size - 1) / threadgroup_size;

    // Phase 1: Read, apply twiddle, write to shared (transposed)
    for (uint32_t e = 0; e < elements_per_thread; ++e) {
        uint32_t local_idx = thread_idx + e * threadgroup_size;
        if (local_idx >= TILE_DIM * TILE_DIM) break;

        uint32_t local_row = local_idx / TILE_DIM;
        uint32_t local_col = local_idx % TILE_DIM;

        uint32_t global_row = tile_row * TILE_DIM + local_row;
        uint32_t global_col = tile_col * TILE_DIM + local_col;

        if (global_row < N1 && global_col < N2) {
            uint32_t in_idx = global_row * N2 + global_col;
            uint64_t val = batch_input[in_idx];

            // Apply twiddle factor omega^(i*j)
            uint32_t tw_idx = global_row * N2 + global_col;
            uint64_t tw = twiddles[tw_idx];
            uint64_t tw_pre = twiddle_precon[tw_idx];
            val = barrett_mul_precon(val, tw, Q, tw_pre);

            // Store transposed: [local_col][local_row] instead of [local_row][local_col]
            shared[local_col * tile_stride + local_row] = val;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Read from shared (already transposed), write to output
    for (uint32_t e = 0; e < elements_per_thread; ++e) {
        uint32_t local_idx = thread_idx + e * threadgroup_size;
        if (local_idx >= TILE_DIM * TILE_DIM) break;

        uint32_t local_row = local_idx / TILE_DIM;
        uint32_t local_col = local_idx % TILE_DIM;

        // Output position: swap tile coordinates for transpose
        uint32_t out_row = tile_col * TILE_DIM + local_row;
        uint32_t out_col = tile_row * TILE_DIM + local_col;

        if (out_row < N2 && out_col < N1) {
            uint32_t out_idx = out_row * N1 + out_col;
            batch_output[out_idx] = shared[local_row * tile_stride + local_col];
        }
    }
}

/**
 * @brief Inverse twiddle and transpose for inverse NTT
 */
kernel void four_step_inv_twiddle_transpose(
    device uint64_t* output [[buffer(0)]],
    device const uint64_t* input [[buffer(1)]],
    constant uint64_t* twiddles [[buffer(2)]],
    constant uint64_t* twiddle_precon [[buffer(3)]],
    constant FourStepParams& params [[buffer(4)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_tg [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]],
    threadgroup uint64_t* shared [[threadgroup(0)]]
) {
    uint32_t thread_idx = thread_pos_in_tg.x + thread_pos_in_tg.y * tg_size.x + thread_pos_in_tg.z * tg_size.x * tg_size.y;
    uint32_t threadgroup_size = tg_size.x * tg_size.y * tg_size.z;

    uint32_t N1 = params.N1;
    uint32_t N2 = params.N2;
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint32_t batch_idx = tg_pos.z;
    uint32_t tile_row = tg_pos.y;
    uint32_t tile_col = tg_pos.x;
    uint32_t tile_stride = params.tile_stride;
    uint32_t TILE_DIM = MAX_TILE_DIM;

    device const uint64_t* batch_input = input + batch_idx * N;
    device uint64_t* batch_output = output + batch_idx * N;

    uint32_t elements_per_thread = (TILE_DIM * TILE_DIM + threadgroup_size - 1) / threadgroup_size;

    for (uint32_t e = 0; e < elements_per_thread; ++e) {
        uint32_t local_idx = thread_idx + e * threadgroup_size;
        if (local_idx >= TILE_DIM * TILE_DIM) break;

        uint32_t local_row = local_idx / TILE_DIM;
        uint32_t local_col = local_idx % TILE_DIM;

        uint32_t global_row = tile_row * TILE_DIM + local_row;
        uint32_t global_col = tile_col * TILE_DIM + local_col;

        if (global_row < N2 && global_col < N1) {
            uint32_t in_idx = global_row * N1 + global_col;
            uint64_t val = batch_input[in_idx];

            uint32_t tw_idx = global_row * N1 + global_col;
            uint64_t tw = twiddles[tw_idx];
            uint64_t tw_pre = twiddle_precon[tw_idx];
            val = barrett_mul_precon(val, tw, Q, tw_pre);

            shared[local_col * tile_stride + local_row] = val;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint32_t e = 0; e < elements_per_thread; ++e) {
        uint32_t local_idx = thread_idx + e * threadgroup_size;
        if (local_idx >= TILE_DIM * TILE_DIM) break;

        uint32_t local_row = local_idx / TILE_DIM;
        uint32_t local_col = local_idx % TILE_DIM;

        uint32_t out_row = tile_col * TILE_DIM + local_row;
        uint32_t out_col = tile_row * TILE_DIM + local_col;

        if (out_row < N1 && out_col < N2) {
            uint32_t out_idx = out_row * N2 + out_col;
            batch_output[out_idx] = shared[local_row * tile_stride + local_col];
        }
    }
}

// =============================================================================
// Step 4: Row NTTs (Forward)
// =============================================================================

kernel void four_step_row_ntt(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* twiddles [[buffer(1)]],
    constant uint64_t* twiddle_precon [[buffer(2)]],
    constant FourStepParams& params [[buffer(3)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_tg [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]],
    threadgroup uint64_t* shared [[threadgroup(0)]]
) {
    uint32_t thread_idx = thread_pos_in_tg.x + thread_pos_in_tg.y * tg_size.x + thread_pos_in_tg.z * tg_size.x * tg_size.y;
    uint32_t threadgroup_size = tg_size.x * tg_size.y * tg_size.z;

    uint32_t N1 = params.N1;
    uint32_t N2 = params.N2;
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint32_t batch_idx = tg_pos.z;
    uint32_t tile_row = tg_pos.y;
    uint32_t tile_col = tg_pos.x;
    uint32_t tile_stride = params.tile_stride;

    uint32_t TILE_N2 = min(N2, MAX_TILE_DIM);
    uint32_t TILE_N1 = min(N1, MAX_TILE_DIM);

    device uint64_t* batch_data = data + batch_idx * N;

    // Load tile
    uint32_t elements_per_thread = (TILE_N2 * TILE_N1 + threadgroup_size - 1) / threadgroup_size;

    for (uint32_t e = 0; e < elements_per_thread; ++e) {
        uint32_t local_idx = thread_idx + e * threadgroup_size;
        if (local_idx >= TILE_N2 * TILE_N1) break;

        uint32_t local_row = local_idx / TILE_N1;
        uint32_t local_col = local_idx % TILE_N1;

        uint32_t global_row = tile_row * TILE_N2 + local_row;
        uint32_t global_col = tile_col * TILE_N1 + local_col;

        if (global_row < N2 && global_col < N1) {
            uint32_t global_idx = global_row * N1 + global_col;
            shared[local_row * tile_stride + local_col] = batch_data[global_idx];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Row NTTs
    uint32_t log_N2 = params.log_N2;

    for (uint32_t row = 0; row < TILE_N2; ++row) {
        threadgroup_ntt_forward(
            shared + row * tile_stride,  // Start of row
            1,                           // Stride = 1 (consecutive elements)
            TILE_N1,                     // Size (rows are N1 elements after transpose)
            log_N2,                      // Note: using log_N2 for row size after transpose
            thread_idx,
            threadgroup_size,
            twiddles,
            twiddle_precon,
            Q
        );
    }

    // Write back
    for (uint32_t e = 0; e < elements_per_thread; ++e) {
        uint32_t local_idx = thread_idx + e * threadgroup_size;
        if (local_idx >= TILE_N2 * TILE_N1) break;

        uint32_t local_row = local_idx / TILE_N1;
        uint32_t local_col = local_idx % TILE_N1;

        uint32_t global_row = tile_row * TILE_N2 + local_row;
        uint32_t global_col = tile_col * TILE_N1 + local_col;

        if (global_row < N2 && global_col < N1) {
            uint32_t global_idx = global_row * N1 + global_col;
            batch_data[global_idx] = shared[local_row * tile_stride + local_col];
        }
    }
}

// =============================================================================
// Step 4: Row NTTs (Inverse)
// =============================================================================

kernel void four_step_row_intt(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* twiddles [[buffer(1)]],
    constant uint64_t* twiddle_precon [[buffer(2)]],
    constant FourStepParams& params [[buffer(3)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_tg [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]],
    threadgroup uint64_t* shared [[threadgroup(0)]]
) {
    uint32_t thread_idx = thread_pos_in_tg.x + thread_pos_in_tg.y * tg_size.x + thread_pos_in_tg.z * tg_size.x * tg_size.y;
    uint32_t threadgroup_size = tg_size.x * tg_size.y * tg_size.z;

    uint32_t N1 = params.N1;
    uint32_t N2 = params.N2;
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint32_t batch_idx = tg_pos.z;
    uint32_t tile_row = tg_pos.y;
    uint32_t tile_col = tg_pos.x;
    uint32_t tile_stride = params.tile_stride;

    uint32_t TILE_N1 = min(N1, MAX_TILE_DIM);
    uint32_t TILE_N2 = min(N2, MAX_TILE_DIM);

    device uint64_t* batch_data = data + batch_idx * N;

    uint32_t elements_per_thread = (TILE_N1 * TILE_N2 + threadgroup_size - 1) / threadgroup_size;

    for (uint32_t e = 0; e < elements_per_thread; ++e) {
        uint32_t local_idx = thread_idx + e * threadgroup_size;
        if (local_idx >= TILE_N1 * TILE_N2) break;

        uint32_t local_row = local_idx / TILE_N2;
        uint32_t local_col = local_idx % TILE_N2;

        uint32_t global_row = tile_row * TILE_N1 + local_row;
        uint32_t global_col = tile_col * TILE_N2 + local_col;

        if (global_row < N1 && global_col < N2) {
            uint32_t global_idx = global_row * N2 + global_col;
            shared[local_row * tile_stride + local_col] = batch_data[global_idx];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint32_t log_N1 = params.log_N1;

    for (uint32_t row = 0; row < TILE_N1; ++row) {
        threadgroup_ntt_inverse(
            shared + row * tile_stride,
            1,
            TILE_N2,
            log_N1,
            thread_idx,
            threadgroup_size,
            twiddles,
            twiddle_precon,
            Q
        );
    }

    for (uint32_t e = 0; e < elements_per_thread; ++e) {
        uint32_t local_idx = thread_idx + e * threadgroup_size;
        if (local_idx >= TILE_N1 * TILE_N2) break;

        uint32_t local_row = local_idx / TILE_N2;
        uint32_t local_col = local_idx % TILE_N2;

        uint32_t global_row = tile_row * TILE_N1 + local_row;
        uint32_t global_col = tile_col * TILE_N2 + local_col;

        if (global_row < N1 && global_col < N2) {
            uint32_t global_idx = global_row * N2 + global_col;
            batch_data[global_idx] = shared[local_row * tile_stride + local_col];
        }
    }
}

// =============================================================================
// N^{-1} Scaling Kernel
// =============================================================================

/**
 * @brief Scale all elements by N^{-1} for inverse NTT normalization
 */
kernel void four_step_scale_n_inv(
    device uint64_t* data [[buffer(0)]],
    constant FourStepParams& params [[buffer(1)]],
    uint global_idx [[thread_position_in_grid]]
) {
    uint32_t total_elements = params.N * params.batch_size;
    if (global_idx >= total_elements) return;

    uint64_t val = data[global_idx];
    uint64_t scaled = barrett_mul_precon(val, params.N_inv, params.Q, params.N_inv_precon);
    data[global_idx] = scaled;
}

// =============================================================================
// Pointwise Modular Multiplication
// =============================================================================

/**
 * @brief Element-wise multiplication of two polynomials in NTT domain
 */
kernel void four_step_pointwise_mul(
    device uint64_t* result [[buffer(0)]],
    device const uint64_t* a [[buffer(1)]],
    device const uint64_t* b [[buffer(2)]],
    constant FourStepParams& params [[buffer(3)]],
    uint global_idx [[thread_position_in_grid]]
) {
    uint32_t total_elements = params.N * params.batch_size;
    if (global_idx >= total_elements) return;

    uint64_t av = a[global_idx];
    uint64_t bv = b[global_idx];
    result[global_idx] = barrett_mul(av, bv, params.Q, params.mu);
}

// =============================================================================
// Complete Four-Step NTT (Fused for Small N)
// =============================================================================

/**
 * @brief Fused Four-Step NTT for N <= 4096
 *
 * When N fits entirely in threadgroup memory (N <= 4096 for 32KB),
 * we can process all four steps without returning to global memory.
 */
kernel void four_step_ntt_fused(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* col_twiddles [[buffer(1)]],
    constant uint64_t* col_tw_precon [[buffer(2)]],
    constant uint64_t* trans_twiddles [[buffer(3)]],
    constant uint64_t* trans_tw_precon [[buffer(4)]],
    constant uint64_t* row_twiddles [[buffer(5)]],
    constant uint64_t* row_tw_precon [[buffer(6)]],
    constant FourStepParams& params [[buffer(7)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_tg [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]],
    threadgroup uint64_t* shared [[threadgroup(0)]]
) {
    uint32_t thread_idx = thread_pos_in_tg.x + thread_pos_in_tg.y * tg_size.x + thread_pos_in_tg.z * tg_size.x * tg_size.y;
    uint32_t threadgroup_size = tg_size.x * tg_size.y * tg_size.z;

    uint32_t N1 = params.N1;
    uint32_t N2 = params.N2;
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint32_t batch_idx = tg_pos.x;
    uint32_t log_N1 = params.log_N1;
    uint32_t log_N2 = params.log_N2;

    device uint64_t* batch_data = data + batch_idx * N;

    // =========================================================================
    // Phase 1: Load entire polynomial into shared memory
    // =========================================================================
    uint32_t elements_per_thread = (N + threadgroup_size - 1) / threadgroup_size;

    for (uint32_t e = 0; e < elements_per_thread; ++e) {
        uint32_t local_idx = thread_idx + e * threadgroup_size;
        if (local_idx < N) {
            shared[local_idx] = batch_data[local_idx];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // Phase 2: Column NTTs (Step 1)
    // =========================================================================
    for (uint32_t col = 0; col < N2; ++col) {
        threadgroup_ntt_forward(
            shared + col,
            N2,  // Stride between rows
            N1,
            log_N1,
            thread_idx,
            threadgroup_size,
            col_twiddles,
            col_tw_precon,
            Q
        );
    }

    // =========================================================================
    // Phase 3: Twiddle multiplication (Step 2)
    // =========================================================================
    for (uint32_t e = 0; e < elements_per_thread; ++e) {
        uint32_t local_idx = thread_idx + e * threadgroup_size;
        if (local_idx < N) {
            uint32_t i = local_idx / N2;  // Row
            uint32_t j = local_idx % N2;  // Column
            uint64_t val = shared[local_idx];
            uint64_t tw = trans_twiddles[i * N2 + j];
            uint64_t tw_pre = trans_tw_precon[i * N2 + j];
            shared[local_idx] = barrett_mul_precon(val, tw, Q, tw_pre);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // Phase 4: In-place transpose in shared memory (Step 3)
    // =========================================================================
    // Use temporary storage for transpose (requires extra passes)
    // For square matrices (N1 == N2), we can do in-place swap
    if (N1 == N2) {
        for (uint32_t e = 0; e < elements_per_thread; ++e) {
            uint32_t local_idx = thread_idx + e * threadgroup_size;
            if (local_idx < N) {
                uint32_t row = local_idx / N2;
                uint32_t col = local_idx % N2;
                // Only swap upper triangle
                if (row < col) {
                    uint32_t idx1 = row * N2 + col;
                    uint32_t idx2 = col * N1 + row;
                    uint64_t temp = shared[idx1];
                    shared[idx1] = shared[idx2];
                    shared[idx2] = temp;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // Phase 5: Row NTTs (Step 4)
    // =========================================================================
    for (uint32_t row = 0; row < N1; ++row) {
        threadgroup_ntt_forward(
            shared + row * N2,
            1,  // Consecutive elements
            N2,
            log_N2,
            thread_idx,
            threadgroup_size,
            row_twiddles,
            row_tw_precon,
            Q
        );
    }

    // =========================================================================
    // Phase 6: Write back to global memory
    // =========================================================================
    for (uint32_t e = 0; e < elements_per_thread; ++e) {
        uint32_t local_idx = thread_idx + e * threadgroup_size;
        if (local_idx < N) {
            batch_data[local_idx] = shared[local_idx];
        }
    }
}
