// =============================================================================
// Unified Memory NTT Metal Shaders
// =============================================================================
//
// Zero-copy NTT kernels for Apple Silicon's unified memory architecture.
//
// Key innovations:
// 1. Direct operation on MTLResourceStorageModeShared buffers
// 2. No explicit memory transfers - CPU and GPU share physical memory
// 3. Double-buffered streaming for overlapped execution
// 4. Persistent twiddle cache eliminates repeated uploads
//
// Memory model:
// - All buffers use StorageModeShared (unified memory)
// - CPU writes are immediately visible to GPU (cache coherent)
// - GPU writes are immediately visible to CPU after command completion
// - No explicit synchronization needed for sequential access
//
// Performance characteristics:
// - Unified memory bandwidth: ~200 GB/s (M3 Pro/Max)
// - Latency: ~100ns (GPU to memory)
// - Zero PCIe transfer overhead (vs discrete GPU)
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Constants and Structures
// =============================================================================

// Maximum twiddles in threadgroup shared memory (32KB / 8 bytes)
constant uint32_t MAX_SHARED_TWIDDLES = 4096;

// Threadgroup sizes optimized for Apple Silicon
constant uint32_t SIMD_WIDTH = 32;
constant uint32_t MAX_THREADGROUP_SIZE = 1024;

// NTT parameters structure (matches host-side struct)
struct NTTParams {
    uint64_t Q;            // Prime modulus
    uint64_t mu;           // Barrett constant: floor(2^64 / Q)
    uint64_t N_inv;        // N^{-1} mod Q
    uint64_t N_inv_precon; // Barrett precomputation for N_inv
    uint32_t N;            // Ring dimension
    uint32_t log_N;        // log2(N)
    uint32_t stage;        // Current NTT stage
    uint32_t batch;        // Batch size
};

// Streaming configuration for double-buffering
struct StreamConfig {
    uint32_t buffer_index;     // Current buffer (0 or 1)
    uint32_t polynomials;      // Number of polynomials in batch
    uint32_t stages_complete;  // Stages completed in current pass
    uint32_t total_stages;     // Total stages to execute
};

// =============================================================================
// Modular Arithmetic (Optimized for Unified Memory Access)
// =============================================================================

// Barrett multiplication: (a * b) mod Q
// Optimized for unified memory where data stays resident
inline uint64_t barrett_mul_unified(uint64_t a, uint64_t b, uint64_t Q, uint64_t mu) {
    uint64_t lo = a * b;
    uint64_t q = mulhi(lo, mu);
    uint64_t result = lo - q * Q;

    // Branch-free conditional subtraction
    // Unified memory has good latency, but avoiding branches helps GPU pipeline
    uint64_t mask = (result >= Q) ? ~0ULL : 0ULL;
    result -= (Q & mask);

    return result;
}

// Modular addition with branch-free reduction
inline uint64_t mod_add_unified(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    uint64_t mask = (sum >= Q) ? ~0ULL : 0ULL;
    return sum - (Q & mask);
}

// Modular subtraction with branch-free correction
inline uint64_t mod_sub_unified(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t diff = a - b;
    uint64_t mask = (a < b) ? ~0ULL : 0ULL;
    return diff + (Q & mask);
}

// =============================================================================
// Core Butterfly Operations
// =============================================================================

// Cooley-Tukey butterfly (forward NTT)
// (lo, hi) -> (lo + hi*tw, lo - hi*tw)
inline void ct_butterfly(thread uint64_t& lo, thread uint64_t& hi,
                          uint64_t tw, uint64_t Q, uint64_t mu) {
    uint64_t hi_tw = barrett_mul_unified(hi, tw, Q, mu);
    uint64_t new_lo = mod_add_unified(lo, hi_tw, Q);
    uint64_t new_hi = mod_sub_unified(lo, hi_tw, Q);
    lo = new_lo;
    hi = new_hi;
}

// Gentleman-Sande butterfly (inverse NTT)
// (lo, hi) -> (lo + hi, (lo - hi) * tw)
inline void gs_butterfly(thread uint64_t& lo, thread uint64_t& hi,
                          uint64_t tw, uint64_t Q, uint64_t mu) {
    uint64_t sum = mod_add_unified(lo, hi, Q);
    uint64_t diff = mod_sub_unified(lo, hi, Q);
    lo = sum;
    hi = barrett_mul_unified(diff, tw, Q, mu);
}

// =============================================================================
// Unified Memory Forward NTT Stage
// =============================================================================
//
// Single stage of Cooley-Tukey NTT operating directly on unified memory.
// No explicit memory transfers - data stays in shared physical memory.
//
// Threading model:
// - One threadgroup per batch element
// - Threads cooperatively process butterflies within polynomial
// - Twiddles prefetched to threadgroup memory for fast access

kernel void unified_ntt_forward_stage(
    device uint64_t* data [[buffer(0)]],          // Polynomial data (unified memory)
    device const uint64_t* twiddles [[buffer(1)]], // Twiddle factors (unified memory, persistent)
    constant NTTParams& params [[buffer(2)]],
    threadgroup uint64_t* shared_tw [[threadgroup(0)]], // Twiddle cache
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint32_t stage = params.stage;
    uint32_t batch_idx = tg_id;

    // Stage parameters
    uint32_t m = 1u << stage;           // Number of twiddle groups
    uint32_t t = N >> (stage + 1);      // Butterflies per group

    // =========================================================================
    // Phase 1: Cooperative Twiddle Prefetch
    // =========================================================================
    // Load twiddles into threadgroup shared memory.
    // With unified memory, this is a cache-to-cache transfer (very fast).
    // Benefit: Each twiddle loaded once, reused by all threads in group.

    uint32_t tw_to_load = min(m, MAX_SHARED_TWIDDLES);
    uint32_t loads_per_thread = (tw_to_load + tg_size - 1) / tg_size;

    for (uint32_t i = 0; i < loads_per_thread; ++i) {
        uint32_t tw_idx = tid + i * tg_size;
        if (tw_idx < tw_to_load) {
            // Twiddles stored as: twiddles[m + i] for stage s with m = 2^s groups
            shared_tw[tw_idx] = twiddles[m + tw_idx];
        }
    }

    // Barrier: ensure all twiddles loaded before butterfly phase
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // Phase 2: Butterfly Computation
    // =========================================================================
    // Each thread processes multiple butterflies.
    // Unified memory provides coherent access without explicit sync.

    device uint64_t* batch_data = data + batch_idx * N;
    uint32_t butterflies_total = N / 2;
    uint32_t butterflies_per_thread = (butterflies_total + tg_size - 1) / tg_size;

    for (uint32_t b = 0; b < butterflies_per_thread; ++b) {
        uint32_t butterfly_idx = tid + b * tg_size;
        if (butterfly_idx >= butterflies_total) break;

        // Compute indices for this butterfly
        uint32_t group = butterfly_idx / t;
        uint32_t elem = butterfly_idx % t;

        uint32_t idx_lo = (group << (params.log_N - stage)) + elem;
        uint32_t idx_hi = idx_lo + t;

        // Load data from unified memory (no explicit transfer)
        uint64_t lo = batch_data[idx_lo];
        uint64_t hi = batch_data[idx_hi];

        // Get twiddle from shared memory (or global if too many)
        uint64_t tw = (group < MAX_SHARED_TWIDDLES) ? shared_tw[group] : twiddles[m + group];

        // Butterfly computation
        ct_butterfly(lo, hi, tw, Q, mu);

        // Write back to unified memory (immediately visible after completion)
        batch_data[idx_lo] = lo;
        batch_data[idx_hi] = hi;
    }
}

// =============================================================================
// Unified Memory Inverse NTT Stage
// =============================================================================

kernel void unified_ntt_inverse_stage(
    device uint64_t* data [[buffer(0)]],
    device const uint64_t* twiddles [[buffer(1)]],
    constant NTTParams& params [[buffer(2)]],
    threadgroup uint64_t* shared_tw [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint32_t stage = params.stage;
    uint32_t batch_idx = tg_id;

    // GS butterfly parameters
    uint32_t m = N >> (stage + 1);      // Number of twiddle groups
    uint32_t t = 1u << stage;           // Butterflies per group

    // Phase 1: Cooperative twiddle prefetch
    uint32_t tw_to_load = min(m, MAX_SHARED_TWIDDLES);
    uint32_t loads_per_thread = (tw_to_load + tg_size - 1) / tg_size;

    for (uint32_t i = 0; i < loads_per_thread; ++i) {
        uint32_t tw_idx = tid + i * tg_size;
        if (tw_idx < tw_to_load) {
            shared_tw[tw_idx] = twiddles[m + tw_idx];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Butterfly computation
    device uint64_t* batch_data = data + batch_idx * N;
    uint32_t butterflies_total = N / 2;
    uint32_t butterflies_per_thread = (butterflies_total + tg_size - 1) / tg_size;

    for (uint32_t b = 0; b < butterflies_per_thread; ++b) {
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
}

// =============================================================================
// Fused Multi-Stage NTT (Optimal for Small N)
// =============================================================================
//
// For N <= 4096, all twiddles fit in threadgroup memory.
// This kernel executes all log(N) stages without returning to host,
// eliminating kernel launch overhead between stages.
//
// Key optimization: Twiddles loaded once, stages execute in sequence.

kernel void unified_ntt_forward_fused(
    device uint64_t* data [[buffer(0)]],
    device const uint64_t* twiddles [[buffer(1)]],
    constant NTTParams& params [[buffer(2)]],
    threadgroup uint64_t* shared_tw [[threadgroup(0)]],
    threadgroup uint64_t* shared_data [[threadgroup(1)]], // Local polynomial copy
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint32_t log_N = params.log_N;
    uint32_t batch_idx = tg_id;

    device uint64_t* batch_data = data + batch_idx * N;

    // =========================================================================
    // Phase 1: Load ALL twiddles into shared memory
    // =========================================================================
    // For N=4096, we need ~4095 twiddles = 32KB (fits in M3 shared memory)

    uint32_t total_twiddles = N - 1;  // Sum of 2^0 + 2^1 + ... + 2^(log_N-1)
    uint32_t loads_per_thread = (total_twiddles + tg_size - 1) / tg_size;

    for (uint32_t i = 0; i < loads_per_thread; ++i) {
        uint32_t tw_idx = tid + i * tg_size;
        if (tw_idx < total_twiddles && tw_idx < MAX_SHARED_TWIDDLES) {
            // Twiddles stored contiguously starting at index 1
            shared_tw[tw_idx] = twiddles[tw_idx + 1];
        }
    }

    // =========================================================================
    // Phase 2: Load polynomial into shared memory
    // =========================================================================
    // Coalesced load for maximum bandwidth utilization

    uint32_t loads_data = (N + tg_size - 1) / tg_size;
    for (uint32_t i = 0; i < loads_data; ++i) {
        uint32_t idx = tid + i * tg_size;
        if (idx < N) {
            shared_data[idx] = batch_data[idx];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // Phase 3: Execute ALL NTT stages in shared memory
    // =========================================================================

    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = 1u << stage;
        uint32_t t = N >> (stage + 1);
        uint32_t tw_offset = m - 1;  // Offset to stage's twiddles in shared array

        uint32_t butterflies_per_thread = (N / 2 + tg_size - 1) / tg_size;

        for (uint32_t b = 0; b < butterflies_per_thread; ++b) {
            uint32_t butterfly_idx = tid + b * tg_size;
            if (butterfly_idx >= N / 2) break;

            uint32_t group = butterfly_idx / t;
            uint32_t elem = butterfly_idx % t;

            uint32_t idx_lo = (group << (log_N - stage)) + elem;
            uint32_t idx_hi = idx_lo + t;

            // All accesses from shared memory (ultra-fast)
            uint64_t lo = shared_data[idx_lo];
            uint64_t hi = shared_data[idx_hi];
            uint64_t tw = shared_tw[tw_offset + group];

            ct_butterfly(lo, hi, tw, Q, mu);

            shared_data[idx_lo] = lo;
            shared_data[idx_hi] = hi;
        }

        // Barrier between stages
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // =========================================================================
    // Phase 4: Write result back to unified memory
    // =========================================================================

    for (uint32_t i = 0; i < loads_data; ++i) {
        uint32_t idx = tid + i * tg_size;
        if (idx < N) {
            batch_data[idx] = shared_data[idx];
        }
    }
}

// =============================================================================
// Fused Inverse NTT
// =============================================================================

kernel void unified_ntt_inverse_fused(
    device uint64_t* data [[buffer(0)]],
    device const uint64_t* twiddles [[buffer(1)]],
    constant NTTParams& params [[buffer(2)]],
    threadgroup uint64_t* shared_tw [[threadgroup(0)]],
    threadgroup uint64_t* shared_data [[threadgroup(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint64_t N_inv = params.N_inv;
    uint32_t log_N = params.log_N;
    uint32_t batch_idx = tg_id;

    device uint64_t* batch_data = data + batch_idx * N;

    // Load twiddles
    uint32_t total_twiddles = N - 1;
    uint32_t loads_per_thread = (total_twiddles + tg_size - 1) / tg_size;

    for (uint32_t i = 0; i < loads_per_thread; ++i) {
        uint32_t tw_idx = tid + i * tg_size;
        if (tw_idx < total_twiddles && tw_idx < MAX_SHARED_TWIDDLES) {
            shared_tw[tw_idx] = twiddles[tw_idx + 1];
        }
    }

    // Load polynomial
    uint32_t loads_data = (N + tg_size - 1) / tg_size;
    for (uint32_t i = 0; i < loads_data; ++i) {
        uint32_t idx = tid + i * tg_size;
        if (idx < N) {
            shared_data[idx] = batch_data[idx];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Execute all inverse stages
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = N >> (stage + 1);
        uint32_t t = 1u << stage;
        uint32_t tw_offset = m - 1;

        uint32_t butterflies_per_thread = (N / 2 + tg_size - 1) / tg_size;

        for (uint32_t b = 0; b < butterflies_per_thread; ++b) {
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

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Scale by N^{-1} and write back
    for (uint32_t i = 0; i < loads_data; ++i) {
        uint32_t idx = tid + i * tg_size;
        if (idx < N) {
            uint64_t val = shared_data[idx];
            batch_data[idx] = barrett_mul_unified(val, N_inv, Q, mu);
        }
    }
}

// =============================================================================
// N^{-1} Scaling Kernel (for staged inverse NTT)
// =============================================================================

kernel void unified_scale_ninv(
    device uint64_t* data [[buffer(0)]],
    constant NTTParams& params [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint total_threads [[threads_per_grid]]
) {
    uint32_t total = params.N * params.batch;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint64_t N_inv = params.N_inv;

    for (uint32_t i = tid; i < total; i += total_threads) {
        data[i] = barrett_mul_unified(data[i], N_inv, Q, mu);
    }
}

// =============================================================================
// Pointwise Modular Multiplication
// =============================================================================
// Operates directly on unified memory - result immediately available to CPU.

kernel void unified_pointwise_mul(
    device uint64_t* result [[buffer(0)]],
    device const uint64_t* a [[buffer(1)]],
    device const uint64_t* b [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint total_threads [[threads_per_grid]]
) {
    uint32_t total = params.N * params.batch;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;

    for (uint32_t i = tid; i < total; i += total_threads) {
        result[i] = barrett_mul_unified(a[i], b[i], Q, mu);
    }
}

// =============================================================================
// Pointwise Modular Addition
// =============================================================================

kernel void unified_pointwise_add(
    device uint64_t* result [[buffer(0)]],
    device const uint64_t* a [[buffer(1)]],
    device const uint64_t* b [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint total_threads [[threads_per_grid]]
) {
    uint32_t total = params.N * params.batch;
    uint64_t Q = params.Q;

    for (uint32_t i = tid; i < total; i += total_threads) {
        result[i] = mod_add_unified(a[i], b[i], Q);
    }
}

// =============================================================================
// Pointwise Modular Subtraction
// =============================================================================

kernel void unified_pointwise_sub(
    device uint64_t* result [[buffer(0)]],
    device const uint64_t* a [[buffer(1)]],
    device const uint64_t* b [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint total_threads [[threads_per_grid]]
) {
    uint32_t total = params.N * params.batch;
    uint64_t Q = params.Q;

    for (uint32_t i = tid; i < total; i += total_threads) {
        result[i] = mod_sub_unified(a[i], b[i], Q);
    }
}

// =============================================================================
// Double-Buffer Streaming Support
// =============================================================================
//
// These kernels support overlapped execution using double-buffering.
// While GPU processes buffer A, CPU can prepare data in buffer B.

struct DoubleBufferParams {
    uint32_t active_buffer;   // 0 or 1
    uint32_t buffer_size;     // N * batch per buffer
    uint32_t ready_flag;      // Set when buffer is ready for GPU
    uint32_t complete_flag;   // Set when GPU is done
};

kernel void unified_ntt_forward_stream(
    device uint64_t* buffer0 [[buffer(0)]],
    device uint64_t* buffer1 [[buffer(1)]],
    device const uint64_t* twiddles [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    constant DoubleBufferParams& stream [[buffer(4)]],
    threadgroup uint64_t* shared_tw [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    // Select active buffer
    device uint64_t* data = (stream.active_buffer == 0) ? buffer0 : buffer1;

    // Execute single stage (called log_N times by host)
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint32_t stage = params.stage;
    uint32_t batch_idx = tg_id;

    uint32_t m = 1u << stage;
    uint32_t t = N >> (stage + 1);

    // Prefetch twiddles
    uint32_t tw_to_load = min(m, MAX_SHARED_TWIDDLES);
    uint32_t loads_per_thread = (tw_to_load + tg_size - 1) / tg_size;

    for (uint32_t i = 0; i < loads_per_thread; ++i) {
        uint32_t tw_idx = tid + i * tg_size;
        if (tw_idx < tw_to_load) {
            shared_tw[tw_idx] = twiddles[m + tw_idx];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Butterfly computation
    device uint64_t* batch_data = data + batch_idx * N;
    uint32_t butterflies_per_thread = (N / 2 + tg_size - 1) / tg_size;

    for (uint32_t b = 0; b < butterflies_per_thread; ++b) {
        uint32_t butterfly_idx = tid + b * tg_size;
        if (butterfly_idx >= N / 2) break;

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
}

// =============================================================================
// SIMD-Optimized Butterfly (For SIMD-Width Processing)
// =============================================================================
//
// Process 32 butterflies in parallel using SIMD groups.
// Optimal for large polynomials with many independent butterflies.

kernel void unified_ntt_forward_simd(
    device uint64_t* data [[buffer(0)]],
    device const uint64_t* twiddles [[buffer(1)]],
    constant NTTParams& params [[buffer(2)]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint simd_groups [[simdgroups_per_threadgroup]]
) {
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint32_t stage = params.stage;
    uint32_t batch_idx = tg_id;

    uint32_t m = 1u << stage;
    uint32_t t = N >> (stage + 1);

    device uint64_t* batch_data = data + batch_idx * N;

    // Each SIMD group processes SIMD_WIDTH butterflies
    uint32_t butterflies_per_simd = SIMD_WIDTH;
    uint32_t simd_groups_needed = (N / 2 + butterflies_per_simd - 1) / butterflies_per_simd;

    for (uint32_t sg = simd_id; sg < simd_groups_needed; sg += simd_groups) {
        uint32_t butterfly_idx = sg * SIMD_WIDTH + simd_lane;
        if (butterfly_idx >= N / 2) continue;

        uint32_t group = butterfly_idx / t;
        uint32_t elem = butterfly_idx % t;

        uint32_t idx_lo = (group << (params.log_N - stage)) + elem;
        uint32_t idx_hi = idx_lo + t;

        uint64_t lo = batch_data[idx_lo];
        uint64_t hi = batch_data[idx_hi];
        uint64_t tw = twiddles[m + group];

        ct_butterfly(lo, hi, tw, Q, mu);

        batch_data[idx_lo] = lo;
        batch_data[idx_hi] = hi;
    }
}

// =============================================================================
// Memory Copy Utilities (For Hybrid Approaches)
// =============================================================================
//
// Even with unified memory, explicit copy can help for:
// 1. Warming up cache hierarchy
// 2. Prefetching next batch while processing current
// 3. Reordering data for coalesced access

kernel void unified_memcpy(
    device uint64_t* dst [[buffer(0)]],
    device const uint64_t* src [[buffer(1)]],
    constant uint32_t& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint total_threads [[threads_per_grid]]
) {
    for (uint32_t i = tid; i < count; i += total_threads) {
        dst[i] = src[i];
    }
}

// Vectorized copy using SIMD (4x uint64_t at a time)
kernel void unified_memcpy_vec4(
    device uint64_t* dst [[buffer(0)]],
    device const uint64_t* src [[buffer(1)]],
    constant uint32_t& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint total_threads [[threads_per_grid]]
) {
    uint32_t vec_count = count / 4;

    // Process 4 elements at a time
    for (uint32_t i = tid; i < vec_count; i += total_threads) {
        uint32_t base = i * 4;
        dst[base + 0] = src[base + 0];
        dst[base + 1] = src[base + 1];
        dst[base + 2] = src[base + 2];
        dst[base + 3] = src[base + 3];
    }

    // Handle remainder
    uint32_t remainder_start = vec_count * 4;
    for (uint32_t i = remainder_start + tid; i < count; i += total_threads) {
        dst[i] = src[i];
    }
}

// =============================================================================
// Benchmark Kernel (Measure Unified Memory Bandwidth)
// =============================================================================

kernel void benchmark_unified_bandwidth(
    device uint64_t* data [[buffer(0)]],
    constant uint32_t& count [[buffer(1)]],
    constant uint32_t& iterations [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint total_threads [[threads_per_grid]]
) {
    uint64_t sum = 0;

    for (uint32_t iter = 0; iter < iterations; ++iter) {
        for (uint32_t i = tid; i < count; i += total_threads) {
            sum += data[i];
        }
    }

    // Prevent optimization
    if (sum == 0xDEADBEEF) {
        data[tid] = sum;
    }
}
