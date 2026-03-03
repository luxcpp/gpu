// =============================================================================
// Speculative Bootstrap Key Prefetching - CUDA Kernels
// =============================================================================
// CUDA port of bsk_prefetch.metal -- byte-identical output.
//
// Double-buffered BSK storage with async memory copy for blind rotation.
//
// Copyright (C) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#ifdef __CUDA_ARCH__
#define BSK_DEVICE __device__ __forceinline__
#else
#define BSK_DEVICE inline
#define __global__
#define __shared__
static inline void __syncthreads() {}
static inline void __threadfence() {}
#endif

struct BSKPrefetchParams {
    uint32_t N;
    uint32_t L;
    uint32_t n;
    uint32_t entry_size;
    uint32_t current_entry;
    uint32_t prefetch_entry;
    uint64_t Q;
    uint64_t mu;
};

// =============================================================================
// Async BSK Copy Kernel
// =============================================================================

extern "C" __global__ void async_bsk_copy(
    int64_t* dst, const int64_t* bsk, const BSKPrefetchParams params)
{
#ifdef __CUDA_ARCH__
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t grid_size = gridDim.x * blockDim.x;
    uint32_t entry_size = params.entry_size;
    uint32_t entry_idx = params.prefetch_entry;
    uint64_t src_offset = (uint64_t)entry_idx * (uint64_t)entry_size;

    for (uint32_t i = gid; i < entry_size; i += grid_size) {
        dst[i] = bsk[src_offset + i];
    }
#endif
}

// =============================================================================
// Async BSK Copy with Shared Memory Staging
// =============================================================================

extern "C" __global__ void async_bsk_copy_staged(
    int64_t* dst, const int64_t* bsk, const BSKPrefetchParams params)
{
#ifdef __CUDA_ARCH__
    extern __shared__ int64_t staging[];

    uint32_t tg_id = blockIdx.x;
    uint32_t local_id = threadIdx.x;
    uint32_t tg_size = blockDim.x;
    uint32_t entry_size = params.entry_size;
    uint32_t entry_idx = params.prefetch_entry;
    uint64_t src_offset = (uint64_t)entry_idx * (uint64_t)entry_size;

    const uint32_t CHUNK_SIZE = 4096;
    uint32_t chunk_start = tg_id * CHUNK_SIZE;
    uint32_t chunk_end = min(chunk_start + CHUNK_SIZE, entry_size);

    // Stage 1: Load chunk into shared memory
    for (uint32_t i = chunk_start + local_id; i < chunk_end; i += tg_size) {
        staging[i - chunk_start] = bsk[src_offset + i];
    }
    __syncthreads();

    // Stage 2: Write chunk to destination
    for (uint32_t i = chunk_start + local_id; i < chunk_end; i += tg_size) {
        dst[i] = staging[i - chunk_start];
    }
#endif
}

// =============================================================================
// Prefetch-Aware CMux Kernel
// =============================================================================

extern "C" __global__ void cmux_with_prefetch(
    int64_t* acc, const int64_t* bsk_active,
    int32_t rotation, const BSKPrefetchParams params)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared[];

    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t grid_size = gridDim.x * blockDim.x;
    uint32_t N = params.N;
    uint32_t L = params.L;
    uint64_t Q = params.Q;

    if (rotation == 0) return;

    int32_t rot = ((rotation % (int32_t)(2 * N)) + (int32_t)(2 * N)) % (int32_t)(2 * N);

    uint64_t* acc_snap = shared;
    uint64_t* rotated = shared + 2 * N;
    uint64_t* diff = shared + 4 * N;
    uint64_t* prod = shared + 6 * N;

    // Load accumulator
    for (uint32_t i = gid; i < 2 * N; i += grid_size) {
        acc_snap[i] = (uint64_t)acc[i] % Q;
    }
    __syncthreads();

    // Negacyclic rotation
    for (uint32_t c = 0; c < 2; ++c) {
        for (uint32_t i = gid; i < N; i += grid_size) {
            int32_t src = (int32_t)i - rot;
            bool neg = false;
            while (src < 0) { src += (int32_t)N; neg = !neg; }
            while (src >= (int32_t)N) { src -= (int32_t)N; neg = !neg; }
            uint64_t val = acc_snap[c * N + (uint32_t)src];
            rotated[c * N + i] = neg ? (Q - val) % Q : val;
        }
    }
    __syncthreads();

    // Compute difference
    for (uint32_t i = gid; i < 2 * N; i += grid_size) {
        uint64_t r = rotated[i], a = acc_snap[i];
        diff[i] = (r >= a) ? r - a : r + Q - a;
    }
    __syncthreads();

    // Initialize product
    for (uint32_t i = gid; i < 2 * N; i += grid_size) {
        prod[i] = 0;
    }
    __syncthreads();

    // External product
    uint64_t mask = (1ULL << 7) - 1;
    for (uint32_t comp = 0; comp < 2; ++comp) {
        uint64_t* diff_c = diff + comp * N;
        for (uint32_t l = 0; l < L; ++l) {
            const int64_t* rgsw_row = bsk_active + comp * L * 2 * N + l * 2 * N;
            for (uint32_t j = gid; j < N; j += grid_size) {
                uint64_t digit = (diff_c[j] >> (l * 7)) & mask;
                for (uint32_t out_c = 0; out_c < 2; ++out_c) {
                    uint64_t rgsw_val = (uint64_t)rgsw_row[out_c * N + j] % Q;
                    uint64_t term = (digit * rgsw_val) % Q;
                    __syncthreads();
                    prod[out_c * N + j] = (prod[out_c * N + j] + term) % Q;
                }
            }
        }
    }
    __syncthreads();

    // Update accumulator
    for (uint32_t i = gid; i < 2 * N; i += grid_size) {
        uint64_t sum = (acc_snap[i] + prod[i]) % Q;
        acc[i] = (int64_t)sum;
    }
#endif
}

// =============================================================================
// Double-Buffered CMux Pipeline
// =============================================================================

extern "C" __global__ void cmux_double_buffered(
    int64_t* acc, int64_t* buffer_a, int64_t* buffer_b,
    const int32_t* rotations, const BSKPrefetchParams params,
    uint32_t active_buffer, uint32_t step_idx)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared[];

    int64_t* active = (active_buffer == 0) ? buffer_a : buffer_b;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t grid_size = gridDim.x * blockDim.x;
    uint32_t N = params.N;
    uint32_t L = params.L;
    uint64_t Q = params.Q;

    int32_t rotation = rotations[step_idx];

    __threadfence();

    if (rotation == 0) return;

    int32_t rot = ((rotation % (int32_t)(2 * N)) + (int32_t)(2 * N)) % (int32_t)(2 * N);

    uint64_t* acc_snap = shared;
    uint64_t* rotated = shared + 2 * N;
    uint64_t* diff = shared + 4 * N;
    uint64_t* prod = shared + 6 * N;

    for (uint32_t i = gid; i < 2 * N; i += grid_size)
        acc_snap[i] = (uint64_t)acc[i] % Q;
    __syncthreads();

    for (uint32_t c = 0; c < 2; ++c) {
        for (uint32_t i = gid; i < N; i += grid_size) {
            int32_t src = (int32_t)i - rot;
            bool neg = false;
            while (src < 0) { src += (int32_t)N; neg = !neg; }
            while (src >= (int32_t)N) { src -= (int32_t)N; neg = !neg; }
            uint64_t val = acc_snap[c * N + (uint32_t)src];
            rotated[c * N + i] = neg ? (Q - val) % Q : val;
        }
    }
    __syncthreads();

    for (uint32_t i = gid; i < 2 * N; i += grid_size) {
        uint64_t r = rotated[i], a = acc_snap[i];
        diff[i] = (r >= a) ? r - a : r + Q - a;
    }
    __syncthreads();

    for (uint32_t i = gid; i < 2 * N; i += grid_size)
        prod[i] = 0;
    __syncthreads();

    uint64_t mask = (1ULL << 7) - 1;
    for (uint32_t comp = 0; comp < 2; ++comp) {
        uint64_t* diff_c = diff + comp * N;
        for (uint32_t l = 0; l < L; ++l) {
            for (uint32_t j = gid; j < N; j += grid_size) {
                uint64_t digit = (diff_c[j] >> (l * 7)) & mask;
                uint64_t row_offset = comp * L * 2 * N + l * 2 * N;
                for (uint32_t out_c = 0; out_c < 2; ++out_c) {
                    uint64_t rgsw_val = (uint64_t)active[row_offset + out_c * N + j] % Q;
                    uint64_t term = (digit * rgsw_val) % Q;
                    prod[out_c * N + j] = (prod[out_c * N + j] + term) % Q;
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = gid; i < 2 * N; i += grid_size) {
        uint64_t sum = (acc_snap[i] + prod[i]) % Q;
        acc[i] = (int64_t)sum;
    }
    __threadfence();
#endif
}

// =============================================================================
// Batch Prefetch Kernel
// =============================================================================

extern "C" __global__ void batch_bsk_prefetch(
    int64_t* dst_buffers, const int64_t* bsk,
    const uint32_t* entry_indices, const BSKPrefetchParams params, uint32_t batch_size)
{
#ifdef __CUDA_ARCH__
    uint32_t batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    uint32_t entry_size = params.entry_size;
    uint32_t entry_idx = entry_indices[batch_idx];
    if (entry_idx >= params.n) return;

    uint64_t src_offset = (uint64_t)entry_idx * (uint64_t)entry_size;
    uint64_t dst_offset = (uint64_t)batch_idx * (uint64_t)entry_size;

    uint32_t threads_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride_x = gridDim.x * blockDim.x;

    for (uint32_t i = threads_x; i < entry_size; i += stride_x) {
        dst_buffers[dst_offset + i] = bsk[src_offset + i];
    }
#endif
}

// =============================================================================
// Streaming Prefetch
// =============================================================================

extern "C" __global__ void streaming_prefetch(
    int64_t* dst, const int64_t* src, uint32_t num_elements)
{
#ifdef __CUDA_ARCH__
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t grid_size = gridDim.x * blockDim.x;
    for (uint32_t i = gid; i < num_elements; i += grid_size) {
        dst[i] = src[i];
    }
#endif
}

// =============================================================================
// Prefetch Completion Signal (using atomics)
// =============================================================================

extern "C" __global__ void signal_prefetch_complete(
    uint32_t* completion_flag, uint32_t expected_value)
{
#ifdef __CUDA_ARCH__
    atomicExch(completion_flag, expected_value + 1);
#endif
}

extern "C" __global__ void wait_prefetch_complete(
    volatile uint32_t* completion_flag, uint32_t expected_value)
{
#ifdef __CUDA_ARCH__
    while (*completion_flag < expected_value) {
        // Spin -- in practice use CUDA events
    }
#endif
}
