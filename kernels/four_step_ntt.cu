// =============================================================================
// Four-Step NTT Optimized for CUDA
// =============================================================================
// CUDA port of four_step_ntt.metal -- byte-identical arithmetic output.
//
// Four-Step Algorithm for N = N1 * N2:
//   1. N2 parallel column NTTs of size N1
//   2. Twiddle multiplication by omega^(i*j)
//   3. Matrix transpose
//   4. N1 parallel row NTTs of size N2
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-2-Clause

#include <cstdint>

#ifdef __CUDA_ARCH__
#define NTT_DEVICE __device__ __forceinline__
#else
#define NTT_DEVICE inline
#define __global__
#define __shared__
static inline uint64_t __umul64hi(uint64_t a, uint64_t b) {
    __uint128_t r = (__uint128_t)a * b; return (uint64_t)(r >> 64);
}
static inline void __syncthreads() {}
#endif

static const uint32_t MAX_TILE_SIZE = 4096;
static const uint32_t MAX_TILE_DIM = 64;

struct FourStepParams {
    uint64_t Q;
    uint64_t mu;
    uint64_t N_inv;
    uint64_t N_inv_precon;
    uint32_t N;
    uint32_t N1;
    uint32_t N2;
    uint32_t log_N1;
    uint32_t log_N2;
    uint32_t tile_stride;
    uint32_t batch_size;
};

NTT_DEVICE uint64_t barrett_mul_precon(uint64_t a, uint64_t b, uint64_t Q, uint64_t precon) {
    uint64_t q_approx = __umul64hi(a, precon);
    uint64_t product = a * b;
    uint64_t result = product - q_approx * Q;
    return (result >= Q) ? (result - Q) : result;
}

NTT_DEVICE uint64_t barrett_mul(uint64_t a, uint64_t b, uint64_t Q, uint64_t mu) {
    uint64_t lo = a * b;
    uint64_t q = __umul64hi(lo, mu);
    uint64_t result = lo - q * Q;
    return (result >= Q) ? (result - Q) : result;
}

NTT_DEVICE uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    return (sum >= Q) ? (sum - Q) : sum;
}

NTT_DEVICE uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    return (a >= b) ? (a - b) : (a + Q - b);
}

NTT_DEVICE void ct_butterfly(uint64_t& lo, uint64_t& hi, uint64_t tw, uint64_t tw_pre, uint64_t Q) {
    uint64_t hi_tw = barrett_mul_precon(hi, tw, Q, tw_pre);
    uint64_t new_lo = mod_add(lo, hi_tw, Q);
    uint64_t new_hi = mod_sub(lo, hi_tw, Q);
    lo = new_lo;
    hi = new_hi;
}

NTT_DEVICE void gs_butterfly(uint64_t& lo, uint64_t& hi, uint64_t tw, uint64_t tw_pre, uint64_t Q) {
    uint64_t sum = mod_add(lo, hi, Q);
    uint64_t diff = mod_sub(lo, hi, Q);
    lo = sum;
    hi = barrett_mul_precon(diff, tw, Q, tw_pre);
}

// In-shared-memory NTT helpers (stride access for column NTTs)
NTT_DEVICE void threadgroup_ntt_forward(
    uint64_t* shared, uint32_t stride, uint32_t N, uint32_t log_N,
    uint32_t thread_idx, uint32_t num_threads,
    const uint64_t* twiddles, const uint64_t* twiddle_precon, uint64_t Q)
{
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = 1u << stage;
        uint32_t t = N >> (stage + 1);
        uint32_t num_butterflies = N >> 1;
        uint32_t bpt = (num_butterflies + num_threads - 1) / num_threads;
        for (uint32_t b = 0; b < bpt; ++b) {
            uint32_t bi = thread_idx + b * num_threads;
            if (bi >= num_butterflies) break;
            uint32_t group = bi / t;
            uint32_t j = bi % t;
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
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
    }
}

NTT_DEVICE void threadgroup_ntt_inverse(
    uint64_t* shared, uint32_t stride, uint32_t N, uint32_t log_N,
    uint32_t thread_idx, uint32_t num_threads,
    const uint64_t* twiddles, const uint64_t* twiddle_precon, uint64_t Q)
{
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = N >> (stage + 1);
        uint32_t t = 1u << stage;
        uint32_t num_butterflies = N >> 1;
        uint32_t bpt = (num_butterflies + num_threads - 1) / num_threads;
        for (uint32_t b = 0; b < bpt; ++b) {
            uint32_t bi = thread_idx + b * num_threads;
            if (bi >= num_butterflies) break;
            uint32_t group = bi / t;
            uint32_t j = bi % t;
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
#ifdef __CUDA_ARCH__
        __syncthreads();
#endif
    }
}

// =============================================================================
// Step 1: Column NTTs (Forward)
// =============================================================================

extern "C" __global__ void four_step_column_ntt(
    uint64_t* data, const uint64_t* twiddles, const uint64_t* twiddle_precon,
    const FourStepParams params)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared[];
    uint32_t thread_idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t threadgroup_size = blockDim.x * blockDim.y * blockDim.z;
    uint32_t N1 = params.N1;
    uint32_t N2 = params.N2;
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint32_t batch_idx = blockIdx.z;
    uint32_t tile_row = blockIdx.y;
    uint32_t tile_col = blockIdx.x;
    uint32_t tile_stride = params.tile_stride;
    uint32_t TILE_N1 = min(N1, MAX_TILE_DIM);
    uint32_t TILE_N2 = min(N2, MAX_TILE_DIM);
    uint64_t* batch_data = data + batch_idx * N;

    uint32_t ept = (TILE_N1 * TILE_N2 + threadgroup_size - 1) / threadgroup_size;
    for (uint32_t e = 0; e < ept; ++e) {
        uint32_t li = thread_idx + e * threadgroup_size;
        if (li >= TILE_N1 * TILE_N2) break;
        uint32_t lr = li / TILE_N2;
        uint32_t lc = li % TILE_N2;
        uint32_t gr = tile_row * TILE_N1 + lr;
        uint32_t gc = tile_col * TILE_N2 + lc;
        if (gr < N1 && gc < N2)
            shared[lr * tile_stride + lc] = batch_data[gr * N2 + gc];
    }
    __syncthreads();

    uint32_t log_N1 = params.log_N1;
    for (uint32_t col = 0; col < TILE_N2; ++col) {
        threadgroup_ntt_forward(shared + col, tile_stride, TILE_N1, log_N1,
                                thread_idx, threadgroup_size, twiddles, twiddle_precon, Q);
    }

    for (uint32_t e = 0; e < ept; ++e) {
        uint32_t li = thread_idx + e * threadgroup_size;
        if (li >= TILE_N1 * TILE_N2) break;
        uint32_t lr = li / TILE_N2;
        uint32_t lc = li % TILE_N2;
        uint32_t gr = tile_row * TILE_N1 + lr;
        uint32_t gc = tile_col * TILE_N2 + lc;
        if (gr < N1 && gc < N2)
            batch_data[gr * N2 + gc] = shared[lr * tile_stride + lc];
    }
#endif
}

// =============================================================================
// Step 1: Column NTTs (Inverse)
// =============================================================================

extern "C" __global__ void four_step_column_intt(
    uint64_t* data, const uint64_t* twiddles, const uint64_t* twiddle_precon,
    const FourStepParams params)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared[];
    uint32_t thread_idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t threadgroup_size = blockDim.x * blockDim.y * blockDim.z;
    uint32_t N1 = params.N1, N2 = params.N2, N = params.N;
    uint64_t Q = params.Q;
    uint32_t batch_idx = blockIdx.z;
    uint32_t tile_row = blockIdx.y, tile_col = blockIdx.x;
    uint32_t tile_stride = params.tile_stride;
    uint32_t TILE_N1 = min(N1, MAX_TILE_DIM);
    uint32_t TILE_N2 = min(N2, MAX_TILE_DIM);
    uint64_t* batch_data = data + batch_idx * N;

    uint32_t ept = (TILE_N1 * TILE_N2 + threadgroup_size - 1) / threadgroup_size;
    for (uint32_t e = 0; e < ept; ++e) {
        uint32_t li = thread_idx + e * threadgroup_size;
        if (li >= TILE_N1 * TILE_N2) break;
        uint32_t lr = li / TILE_N2, lc = li % TILE_N2;
        uint32_t gr = tile_row * TILE_N1 + lr, gc = tile_col * TILE_N2 + lc;
        if (gr < N1 && gc < N2)
            shared[lr * tile_stride + lc] = batch_data[gr * N2 + gc];
    }
    __syncthreads();

    for (uint32_t col = 0; col < TILE_N2; ++col) {
        threadgroup_ntt_inverse(shared + col, tile_stride, TILE_N1, params.log_N1,
                                thread_idx, threadgroup_size, twiddles, twiddle_precon, Q);
    }

    for (uint32_t e = 0; e < ept; ++e) {
        uint32_t li = thread_idx + e * threadgroup_size;
        if (li >= TILE_N1 * TILE_N2) break;
        uint32_t lr = li / TILE_N2, lc = li % TILE_N2;
        uint32_t gr = tile_row * TILE_N1 + lr, gc = tile_col * TILE_N2 + lc;
        if (gr < N1 && gc < N2)
            batch_data[gr * N2 + gc] = shared[lr * tile_stride + lc];
    }
#endif
}

// =============================================================================
// Step 2+3: Fused Twiddle Multiplication and Transpose
// =============================================================================

extern "C" __global__ void four_step_twiddle_transpose(
    uint64_t* output, const uint64_t* input,
    const uint64_t* twiddles, const uint64_t* twiddle_precon,
    const FourStepParams params)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared[];
    uint32_t thread_idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t threadgroup_size = blockDim.x * blockDim.y * blockDim.z;
    uint32_t N1 = params.N1, N2 = params.N2, N = params.N;
    uint64_t Q = params.Q;
    uint32_t batch_idx = blockIdx.z;
    uint32_t tile_row = blockIdx.y, tile_col = blockIdx.x;
    uint32_t tile_stride = params.tile_stride;
    uint32_t TILE_DIM = MAX_TILE_DIM;
    const uint64_t* batch_input = input + batch_idx * N;
    uint64_t* batch_output = output + batch_idx * N;

    uint32_t ept = (TILE_DIM * TILE_DIM + threadgroup_size - 1) / threadgroup_size;
    for (uint32_t e = 0; e < ept; ++e) {
        uint32_t li = thread_idx + e * threadgroup_size;
        if (li >= TILE_DIM * TILE_DIM) break;
        uint32_t lr = li / TILE_DIM, lc = li % TILE_DIM;
        uint32_t gr = tile_row * TILE_DIM + lr, gc = tile_col * TILE_DIM + lc;
        if (gr < N1 && gc < N2) {
            uint32_t in_idx = gr * N2 + gc;
            uint64_t val = batch_input[in_idx];
            uint32_t tw_idx = gr * N2 + gc;
            val = barrett_mul_precon(val, twiddles[tw_idx], Q, twiddle_precon[tw_idx]);
            shared[lc * tile_stride + lr] = val;  // transposed store
        }
    }
    __syncthreads();

    for (uint32_t e = 0; e < ept; ++e) {
        uint32_t li = thread_idx + e * threadgroup_size;
        if (li >= TILE_DIM * TILE_DIM) break;
        uint32_t lr = li / TILE_DIM, lc = li % TILE_DIM;
        uint32_t out_row = tile_col * TILE_DIM + lr;
        uint32_t out_col = tile_row * TILE_DIM + lc;
        if (out_row < N2 && out_col < N1)
            batch_output[out_row * N1 + out_col] = shared[lr * tile_stride + lc];
    }
#endif
}

// =============================================================================
// Step 4: Row NTTs (Forward)
// =============================================================================

extern "C" __global__ void four_step_row_ntt(
    uint64_t* data, const uint64_t* twiddles, const uint64_t* twiddle_precon,
    const FourStepParams params)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared[];
    uint32_t thread_idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t threadgroup_size = blockDim.x * blockDim.y * blockDim.z;
    uint32_t N1 = params.N1, N2 = params.N2, N = params.N;
    uint64_t Q = params.Q;
    uint32_t batch_idx = blockIdx.z;
    uint32_t tile_row = blockIdx.y, tile_col = blockIdx.x;
    uint32_t tile_stride = params.tile_stride;
    uint32_t TILE_N2 = min(N2, MAX_TILE_DIM);
    uint32_t TILE_N1 = min(N1, MAX_TILE_DIM);
    uint64_t* batch_data = data + batch_idx * N;

    uint32_t ept = (TILE_N2 * TILE_N1 + threadgroup_size - 1) / threadgroup_size;
    for (uint32_t e = 0; e < ept; ++e) {
        uint32_t li = thread_idx + e * threadgroup_size;
        if (li >= TILE_N2 * TILE_N1) break;
        uint32_t lr = li / TILE_N1, lc = li % TILE_N1;
        uint32_t gr = tile_row * TILE_N2 + lr, gc = tile_col * TILE_N1 + lc;
        if (gr < N2 && gc < N1)
            shared[lr * tile_stride + lc] = batch_data[gr * N1 + gc];
    }
    __syncthreads();

    for (uint32_t row = 0; row < TILE_N2; ++row) {
        threadgroup_ntt_forward(shared + row * tile_stride, 1, TILE_N1, params.log_N2,
                                thread_idx, threadgroup_size, twiddles, twiddle_precon, Q);
    }

    for (uint32_t e = 0; e < ept; ++e) {
        uint32_t li = thread_idx + e * threadgroup_size;
        if (li >= TILE_N2 * TILE_N1) break;
        uint32_t lr = li / TILE_N1, lc = li % TILE_N1;
        uint32_t gr = tile_row * TILE_N2 + lr, gc = tile_col * TILE_N1 + lc;
        if (gr < N2 && gc < N1)
            batch_data[gr * N1 + gc] = shared[lr * tile_stride + lc];
    }
#endif
}

// =============================================================================
// Scaling and Pointwise
// =============================================================================

extern "C" __global__ void four_step_scale_n_inv(uint64_t* data, const FourStepParams params)
{
#ifdef __CUDA_ARCH__
    uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_elements = params.N * params.batch_size;
    if (global_idx >= total_elements) return;
    data[global_idx] = barrett_mul_precon(data[global_idx], params.N_inv, params.Q, params.N_inv_precon);
#endif
}

extern "C" __global__ void four_step_pointwise_mul(
    uint64_t* result, const uint64_t* a, const uint64_t* b, const FourStepParams params)
{
#ifdef __CUDA_ARCH__
    uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_elements = params.N * params.batch_size;
    if (global_idx >= total_elements) return;
    result[global_idx] = barrett_mul(a[global_idx], b[global_idx], params.Q, params.mu);
#endif
}

// =============================================================================
// Fused Four-Step NTT (N <= 4096)
// =============================================================================

extern "C" __global__ void four_step_ntt_fused(
    uint64_t* data,
    const uint64_t* col_twiddles, const uint64_t* col_tw_precon,
    const uint64_t* trans_twiddles, const uint64_t* trans_tw_precon,
    const uint64_t* row_twiddles, const uint64_t* row_tw_precon,
    const FourStepParams params)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared[];
    uint32_t thread_idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t threadgroup_size = blockDim.x * blockDim.y * blockDim.z;
    uint32_t N1 = params.N1, N2 = params.N2, N = params.N;
    uint64_t Q = params.Q;
    uint32_t batch_idx = blockIdx.x;
    uint32_t log_N1 = params.log_N1, log_N2 = params.log_N2;
    uint64_t* batch_data = data + batch_idx * N;

    // Load entire polynomial
    uint32_t ept = (N + threadgroup_size - 1) / threadgroup_size;
    for (uint32_t e = 0; e < ept; ++e) {
        uint32_t li = thread_idx + e * threadgroup_size;
        if (li < N) shared[li] = batch_data[li];
    }
    __syncthreads();

    // Column NTTs
    for (uint32_t col = 0; col < N2; ++col) {
        threadgroup_ntt_forward(shared + col, N2, N1, log_N1,
                                thread_idx, threadgroup_size, col_twiddles, col_tw_precon, Q);
    }

    // Twiddle multiplication
    for (uint32_t e = 0; e < ept; ++e) {
        uint32_t li = thread_idx + e * threadgroup_size;
        if (li < N) {
            uint32_t i = li / N2;
            uint32_t j = li % N2;
            shared[li] = barrett_mul_precon(shared[li], trans_twiddles[i * N2 + j], Q, trans_tw_precon[i * N2 + j]);
        }
    }
    __syncthreads();

    // In-place transpose (square case)
    if (N1 == N2) {
        for (uint32_t e = 0; e < ept; ++e) {
            uint32_t li = thread_idx + e * threadgroup_size;
            if (li < N) {
                uint32_t row = li / N2, col = li % N2;
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
    __syncthreads();

    // Row NTTs
    for (uint32_t row = 0; row < N1; ++row) {
        threadgroup_ntt_forward(shared + row * N2, 1, N2, log_N2,
                                thread_idx, threadgroup_size, row_twiddles, row_tw_precon, Q);
    }

    // Write back
    for (uint32_t e = 0; e < ept; ++e) {
        uint32_t li = thread_idx + e * threadgroup_size;
        if (li < N) batch_data[li] = shared[li];
    }
#endif
}
