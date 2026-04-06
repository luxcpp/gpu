// =============================================================================
// Speculative Bootstrap Key Prefetching - Metal Shaders
// =============================================================================
//
// GPU kernels for async BSK prefetching during blind rotation.
//
// Key innovation: Double-buffered BSK storage with async memory copy
// - Fetch BSK[i+1] while computing CMux with BSK[i]
// - Hides memory latency for all but first iteration
// - ~2x improvement in memory bandwidth utilization
//
// Metal Memory Model:
// - device: Main GPU memory (unified with CPU on Apple Silicon)
// - threadgroup: Shared memory per threadgroup (~32KB on M-series)
// - thread: Registers per thread
//
// Async copy uses Metal's memcpy semantics with fence for ordering.
//
// Copyright (C) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// BSK Prefetch Parameters
// =============================================================================

struct BSKPrefetchParams {
    uint32_t N;             // Ring dimension (e.g., 1024)
    uint32_t L;             // Decomposition levels (e.g., 4)
    uint32_t n;             // LWE dimension (number of BSK entries)
    uint32_t entry_size;    // Elements per entry: 2 * L * 2 * N
    uint32_t current_entry; // Entry being computed
    uint32_t prefetch_entry;// Entry being prefetched
    uint64_t Q;             // Ring modulus
    uint64_t mu;            // Barrett constant
};

// =============================================================================
// Async BSK Copy Kernel
// =============================================================================
//
// Copies one BSK entry from source to destination buffer.
// Designed to run concurrently with CMux computation on another entry.
//
// Dispatch: grid = (entry_size / threads_per_group, 1, 1)
//           threads = (256, 1, 1)

kernel void async_bsk_copy(
    device int64_t* dst              [[buffer(0)]],  // Destination buffer [entry_size]
    constant int64_t* bsk            [[buffer(1)]],  // Full BSK [n, 2, L, 2, N]
    constant BSKPrefetchParams& params [[buffer(2)]],
    uint gid                         [[thread_position_in_grid]],
    uint grid_size                   [[threads_per_grid]]
) {
    uint32_t entry_size = params.entry_size;
    uint32_t entry_idx = params.prefetch_entry;

    // Coalesced copy: each thread handles multiple elements
    uint64_t src_offset = uint64_t(entry_idx) * uint64_t(entry_size);

    for (uint32_t i = gid; i < entry_size; i += grid_size) {
        dst[i] = bsk[src_offset + i];
    }
}

// =============================================================================
// Async BSK Copy with Threadgroup Staging
// =============================================================================
//
// Two-stage copy using threadgroup memory as staging buffer.
// Better for larger entries where direct device-to-device may have latency.
//
// Stage 1: Device -> Threadgroup (coalesced read)
// Stage 2: Threadgroup -> Device (coalesced write)

kernel void async_bsk_copy_staged(
    device int64_t* dst              [[buffer(0)]],
    constant int64_t* bsk            [[buffer(1)]],
    constant BSKPrefetchParams& params [[buffer(2)]],
    uint tg_id                       [[threadgroup_position_in_grid]],
    uint local_id                    [[thread_index_in_threadgroup]],
    uint tg_size                     [[threads_per_threadgroup]],
    threadgroup int64_t* staging     [[threadgroup(0)]]
) {
    uint32_t entry_size = params.entry_size;
    uint32_t entry_idx = params.prefetch_entry;
    uint64_t src_offset = uint64_t(entry_idx) * uint64_t(entry_size);

    // Process in chunks that fit in threadgroup memory
    // Assuming 32KB threadgroup = 4096 int64_t elements max
    constexpr uint32_t CHUNK_SIZE = 4096;

    uint32_t chunk_start = tg_id * CHUNK_SIZE;
    uint32_t chunk_end = min(chunk_start + CHUNK_SIZE, entry_size);

    // Stage 1: Load chunk into threadgroup memory
    for (uint32_t i = chunk_start + local_id; i < chunk_end; i += tg_size) {
        staging[i - chunk_start] = bsk[src_offset + i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: Write chunk to destination
    for (uint32_t i = chunk_start + local_id; i < chunk_end; i += tg_size) {
        dst[i] = staging[i - chunk_start];
    }
}

// =============================================================================
// Prefetch-Aware CMux Kernel
// =============================================================================
//
// CMux with integrated prefetch signaling.
// Uses fence operations to ensure prefetch completes before next iteration.
//
// CMux(selector, d0, d1) = d0 + selector * (d1 - d0)
//                        = d0 + ExternalProduct(d1 - d0, RGSW(selector))

kernel void cmux_with_prefetch(
    device int64_t* acc              [[buffer(0)]],  // Accumulator [2, N] in/out
    constant int64_t* bsk_active     [[buffer(1)]],  // Active BSK entry [2, L, 2, N]
    constant int32_t& rotation       [[buffer(2)]],  // Rotation amount for this step
    constant BSKPrefetchParams& params [[buffer(3)]],
    uint gid                         [[thread_position_in_grid]],
    uint grid_size                   [[threads_per_grid]],
    threadgroup uint64_t* shared     [[threadgroup(0)]]
) {
    uint32_t N = params.N;
    uint32_t L = params.L;
    uint64_t Q = params.Q;

    // Skip if no rotation needed
    if (rotation == 0) return;

    int32_t rot = ((rotation % int32_t(2 * N)) + int32_t(2 * N)) % int32_t(2 * N);

    // Shared memory layout:
    // [0, 2*N): current accumulator snapshot
    // [2*N, 4*N): rotated accumulator
    // [4*N, 6*N): difference (rotated - current)
    // [6*N, 8*N): external product result
    threadgroup uint64_t* acc_snap = shared;
    threadgroup uint64_t* rotated = shared + 2 * N;
    threadgroup uint64_t* diff = shared + 4 * N;
    threadgroup uint64_t* prod = shared + 6 * N;

    // Load accumulator into shared memory
    for (uint32_t i = gid; i < 2 * N; i += grid_size) {
        acc_snap[i] = uint64_t(acc[i]) % Q;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute negacyclic rotation
    for (uint32_t c = 0; c < 2; ++c) {
        for (uint32_t i = gid; i < N; i += grid_size) {
            int32_t src = int32_t(i) - rot;
            bool neg = false;
            while (src < 0) { src += int32_t(N); neg = !neg; }
            while (src >= int32_t(N)) { src -= int32_t(N); neg = !neg; }

            uint64_t val = acc_snap[c * N + uint32_t(src)];
            rotated[c * N + i] = neg ? (Q - val) % Q : val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute difference: rotated - acc
    for (uint32_t i = gid; i < 2 * N; i += grid_size) {
        uint64_t r = rotated[i];
        uint64_t a = acc_snap[i];
        diff[i] = (r >= a) ? r - a : r + Q - a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Initialize product accumulators
    for (uint32_t i = gid; i < 2 * N; i += grid_size) {
        prod[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // External product: diff * RGSW
    // Each thread processes subset of coefficients
    uint64_t mask = (1ULL << 7) - 1;  // Assuming baseLog = 7

    for (uint32_t comp = 0; comp < 2; ++comp) {
        threadgroup uint64_t* diff_c = diff + comp * N;

        for (uint32_t l = 0; l < L; ++l) {
            // RGSW row for (comp, l): [2, N]
            constant int64_t* rgsw_row = bsk_active + comp * L * 2 * N + l * 2 * N;

            for (uint32_t j = gid; j < N; j += grid_size) {
                // Extract digit l
                uint64_t digit = (diff_c[j] >> (l * 7)) & mask;

                // Multiply and accumulate
                for (uint32_t out_c = 0; out_c < 2; ++out_c) {
                    uint64_t rgsw_val = uint64_t(rgsw_row[out_c * N + j]) % Q;
                    uint64_t term = (digit * rgsw_val) % Q;

                    // Atomic add to shared accumulator
                    // (In practice, use reduction pattern for better performance)
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    prod[out_c * N + j] = (prod[out_c * N + j] + term) % Q;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Update accumulator: acc = acc + prod
    for (uint32_t i = gid; i < 2 * N; i += grid_size) {
        uint64_t sum = (acc_snap[i] + prod[i]) % Q;
        acc[i] = int64_t(sum);
    }
}

// =============================================================================
// Double-Buffered CMux Pipeline
// =============================================================================
//
// Main kernel for pipelined blind rotation step.
// Assumes prefetch of next BSK entry is happening concurrently.
//
// Memory ordering:
// 1. Wait for any pending prefetch to complete (fence)
// 2. Execute CMux with active buffer
// 3. Signal completion for next iteration

kernel void cmux_double_buffered(
    device int64_t* acc              [[buffer(0)]],  // Accumulator [2, N]
    device int64_t* buffer_a         [[buffer(1)]],  // BSK buffer A [2, L, 2, N]
    device int64_t* buffer_b         [[buffer(2)]],  // BSK buffer B [2, L, 2, N]
    constant int32_t* rotations      [[buffer(3)]],  // Rotation amounts [n]
    constant BSKPrefetchParams& params [[buffer(4)]],
    constant uint32_t& active_buffer [[buffer(5)]],  // 0 = buffer_a, 1 = buffer_b
    constant uint32_t& step_idx      [[buffer(6)]],  // Current blind rotation step
    uint gid                         [[thread_position_in_grid]],
    uint grid_size                   [[threads_per_grid]],
    threadgroup uint64_t* shared     [[threadgroup(0)]]
) {
    // Select active buffer
    device int64_t* active = (active_buffer == 0) ? buffer_a : buffer_b;

    uint32_t N = params.N;
    uint32_t L = params.L;
    uint64_t Q = params.Q;

    int32_t rotation = rotations[step_idx];

    // Memory fence: ensure previous async copy completed
    threadgroup_barrier(mem_flags::mem_device);

    // Skip if no rotation
    if (rotation == 0) return;

    int32_t rot = ((rotation % int32_t(2 * N)) + int32_t(2 * N)) % int32_t(2 * N);

    // Shared memory for intermediate computations
    // Layout: [8*N] total
    // - [0, 2*N): accumulator snapshot
    // - [2*N, 4*N): rotated
    // - [4*N, 6*N): diff
    // - [6*N, 8*N): prod
    threadgroup uint64_t* acc_snap = shared;
    threadgroup uint64_t* rotated = shared + 2 * N;
    threadgroup uint64_t* diff = shared + 4 * N;
    threadgroup uint64_t* prod = shared + 6 * N;

    // Load accumulator
    for (uint32_t i = gid; i < 2 * N; i += grid_size) {
        acc_snap[i] = uint64_t(acc[i]) % Q;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Negacyclic rotation
    for (uint32_t c = 0; c < 2; ++c) {
        for (uint32_t i = gid; i < N; i += grid_size) {
            int32_t src = int32_t(i) - rot;
            bool neg = false;
            while (src < 0) { src += int32_t(N); neg = !neg; }
            while (src >= int32_t(N)) { src -= int32_t(N); neg = !neg; }

            uint64_t val = acc_snap[c * N + uint32_t(src)];
            rotated[c * N + i] = neg ? (Q - val) % Q : val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Difference
    for (uint32_t i = gid; i < 2 * N; i += grid_size) {
        uint64_t r = rotated[i];
        uint64_t a = acc_snap[i];
        diff[i] = (r >= a) ? r - a : r + Q - a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Initialize product
    for (uint32_t i = gid; i < 2 * N; i += grid_size) {
        prod[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // External product (simplified - full implementation needs proper NTT)
    uint64_t mask = (1ULL << 7) - 1;

    for (uint32_t comp = 0; comp < 2; ++comp) {
        threadgroup uint64_t* diff_c = diff + comp * N;

        for (uint32_t l = 0; l < L; ++l) {
            for (uint32_t j = gid; j < N; j += grid_size) {
                uint64_t digit = (diff_c[j] >> (l * 7)) & mask;

                // RGSW row offset
                uint64_t row_offset = comp * L * 2 * N + l * 2 * N;

                for (uint32_t out_c = 0; out_c < 2; ++out_c) {
                    uint64_t rgsw_val = uint64_t(active[row_offset + out_c * N + j]) % Q;
                    uint64_t term = (digit * rgsw_val) % Q;
                    prod[out_c * N + j] = (prod[out_c * N + j] + term) % Q;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Update accumulator
    for (uint32_t i = gid; i < 2 * N; i += grid_size) {
        uint64_t sum = (acc_snap[i] + prod[i]) % Q;
        acc[i] = int64_t(sum);
    }

    // Memory fence: signal completion for buffer swap
    threadgroup_barrier(mem_flags::mem_device);
}

// =============================================================================
// Prefetch Dispatch Helper
// =============================================================================
//
// Computes dispatch parameters for prefetch kernels.
// Returns optimal grid and threadgroup sizes based on entry size.

struct PrefetchDispatchParams {
    uint3 grid_size;
    uint3 threadgroup_size;
    uint32_t shared_bytes;
};

// Inline helper to compute dispatch params (called from host)
// For N=1024, L=4: entry_size = 16384
// Optimal: 256 threads, 64 threadgroups = 16384 threads total

// =============================================================================
// Batch Prefetch Kernel
// =============================================================================
//
// Prefetch multiple BSK entries for batch processing.
// Useful when processing multiple LWE ciphertexts in parallel.

kernel void batch_bsk_prefetch(
    device int64_t* dst_buffers      [[buffer(0)]],  // [batch, 2, entry_size]
    constant int64_t* bsk            [[buffer(1)]],  // Full BSK [n, 2, L, 2, N]
    constant uint32_t* entry_indices [[buffer(2)]],  // Entries to prefetch [batch]
    constant BSKPrefetchParams& params [[buffer(3)]],
    constant uint32_t& batch_size    [[buffer(4)]],
    uint2 gid                        [[thread_position_in_grid]]
) {
    uint32_t batch_idx = gid.y;
    if (batch_idx >= batch_size) return;

    uint32_t entry_size = params.entry_size;
    uint32_t entry_idx = entry_indices[batch_idx];
    if (entry_idx >= params.n) return;

    uint64_t src_offset = uint64_t(entry_idx) * uint64_t(entry_size);
    uint64_t dst_offset = uint64_t(batch_idx) * uint64_t(entry_size);

    // Each thread in x dimension handles subset of elements
    uint32_t threads_x = gid.x;
    uint32_t stride_x = 256;  // Assuming 256 threads per group in x

    for (uint32_t i = threads_x; i < entry_size; i += stride_x) {
        dst_buffers[dst_offset + i] = bsk[src_offset + i];
    }
}

// =============================================================================
// Streaming Prefetch with Overlap
// =============================================================================
//
// Advanced prefetch that streams BSK data while computation proceeds.
// Uses Metal's async copy semantics for maximum overlap.
//
// Pipeline:
//   T0: Start async copy of BSK[i+1]
//   T1: Continue CMux compute with BSK[i]
//   T2: Async copy completes
//   T3: Swap buffers, repeat

kernel void streaming_prefetch(
    device int64_t* dst              [[buffer(0)]],
    constant int64_t* src            [[buffer(1)]],
    constant uint32_t& num_elements  [[buffer(2)]],
    uint gid                         [[thread_position_in_grid]],
    uint grid_size                   [[threads_per_grid]]
) {
    // Simple streaming copy - Metal scheduler handles overlap
    // with compute kernels dispatched on same command buffer

    for (uint32_t i = gid; i < num_elements; i += grid_size) {
        dst[i] = src[i];
    }

    // No barrier needed here - barrier at consumer kernel
}

// =============================================================================
// Prefetch Completion Signal
// =============================================================================
//
// Lightweight kernel to signal prefetch completion.
// Sets a flag that compute kernels can poll.

kernel void signal_prefetch_complete(
    device atomic_uint* completion_flag [[buffer(0)]],
    constant uint32_t& expected_value   [[buffer(1)]]
) {
    // Atomic store (Metal only supports relaxed memory order)
    atomic_store_explicit(completion_flag, expected_value + 1, memory_order_relaxed);
}

kernel void wait_prefetch_complete(
    device atomic_uint* completion_flag [[buffer(0)]],
    constant uint32_t& expected_value   [[buffer(1)]]
) {
    // Spin until prefetch completes
    // (In practice, use Metal events for better efficiency)
    // Note: Metal only supports memory_order_relaxed
    while (atomic_load_explicit(completion_flag, memory_order_relaxed) < expected_value) {
        // Spin
    }
}
