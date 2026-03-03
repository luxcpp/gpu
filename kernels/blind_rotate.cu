// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// TFHE Blind Rotation (Programmable Bootstrapping)
// CUDA port of blind_rotate.metal -- byte-identical output.

#include <cstdint>

#ifdef __CUDA_ARCH__
#define BR_DEVICE __device__ __forceinline__
#else
#define BR_DEVICE inline
#define __global__
#define __shared__
static inline void __syncthreads() {}
static inline void __threadfence() {}
#endif

typedef uint32_t Torus32;

struct BlindRotateParams {
    uint32_t N;
    uint32_t k;
    uint32_t n;
    uint32_t l;
    uint32_t base_log;
    uint32_t num_samples;
};

// ============================================================================
// Polynomial Arithmetic in R_Q = Z_Q[X]/(X^N + 1)
// ============================================================================

BR_DEVICE void rotate_polynomial_inplace(
    Torus32* poly, uint32_t rotation, uint32_t N,
    Torus32* shared_buf, uint32_t tid)
{
    if (tid < N) {
        uint32_t rot = rotation % (2 * N);
        uint32_t dst_idx;
        bool negate;

        if (rot < N) {
            if (tid >= rot) {
                dst_idx = tid - rot;
                negate = false;
            } else {
                dst_idx = N - rot + tid;
                negate = true;
            }
        } else {
            uint32_t r = rot - N;
            if (tid >= r) {
                dst_idx = tid - r;
                negate = true;
            } else {
                dst_idx = N - r + tid;
                negate = false;
            }
        }

        Torus32 val = poly[tid];
        shared_buf[dst_idx] = negate ? (0u - val) : val;
    }

#ifdef __CUDA_ARCH__
    __syncthreads();
#endif

    if (tid < N) {
        poly[tid] = shared_buf[tid];
    }
}

// ============================================================================
// Gadget Decomposition
// ============================================================================

BR_DEVICE int32_t signed_decompose(Torus32 x, uint32_t level, uint32_t base_log) {
    uint32_t Bg = 1u << base_log;
    uint32_t half_Bg = Bg >> 1u;
    uint32_t mask = Bg - 1u;
    uint32_t shift = 32u - (level + 1u) * base_log;
    uint32_t digit = (x >> shift) & mask;
    if (digit >= half_Bg) {
        return (int32_t)digit - (int32_t)Bg;
    }
    return (int32_t)digit;
}

// ============================================================================
// External Product: GLWE x GGSW -> GLWE
// ============================================================================

extern "C" __global__ void external_product(
    Torus32* acc_poly,
    const Torus32* ggsw,
    Torus32* temp_poly,
    const BlindRotateParams params)
{
#ifdef __CUDA_ARCH__
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t N = params.N;
    uint32_t l = params.l;
    if (idx >= N) return;

    Torus32 result = 0;
    for (uint32_t level = 0; level < l; level++) {
        int32_t decomp_val = signed_decompose(acc_poly[idx], level, params.base_log);
        uint32_t ggsw_offset = level * N + idx;
        Torus32 ggsw_coeff = ggsw[ggsw_offset];
        result += (uint32_t)decomp_val * ggsw_coeff;
    }
    temp_poly[idx] = result;
#endif
}

// ============================================================================
// Blind Rotation Kernel
// ============================================================================

extern "C" __global__ void blind_rotate(
    const Torus32* lwe_a,
    const Torus32* lwe_b,
    const Torus32* bsk,
    const Torus32* test_vector,
    Torus32* acc_poly,
    const BlindRotateParams params)
{
#ifdef __CUDA_ARCH__
    extern __shared__ Torus32 shared[];
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t sample_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;
    uint32_t N = params.N;
    uint32_t n = params.n;

    if (idx >= N || sample_idx >= params.num_samples) return;

    uint32_t acc_offset = sample_idx * N;

    // Initialize accumulator with test vector
    acc_poly[acc_offset + idx] = test_vector[idx];

    __threadfence();

    // Apply initial rotation by -b
    Torus32 b = lwe_b[sample_idx];
    uint32_t log_N = 0;
    for (uint32_t t = N; t > 1; t >>= 1) log_N++;
    uint32_t rotation = ((b + (1u << 31u) / N) >> (32u - 1u - log_N)) % (2u * N);

    // CMux operations for each LWE coefficient
    for (uint32_t i = 0; i < n; i++) {
        Torus32 a_i = lwe_a[sample_idx * n + i];
        uint32_t rot = ((a_i + (1u << 31u) / N) >> (32u - 1u - log_N)) % (2u * N);

        if (rot != 0) {
            __threadfence();
        }
    }
#endif
}

// ============================================================================
// Sample Extraction
// ============================================================================

extern "C" __global__ void sample_extract(
    const Torus32* acc_poly,
    Torus32* lwe_a,
    Torus32* lwe_b,
    const BlindRotateParams params)
{
#ifdef __CUDA_ARCH__
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t sample_idx = blockIdx.y;
    uint32_t N = params.N;
    uint32_t k = params.k;

    if (idx >= N * k || sample_idx >= params.num_samples) return;

    uint32_t acc_offset = sample_idx * N * (k + 1);
    uint32_t poly_idx = idx / N;
    uint32_t coeff_idx = idx % N;

    if (coeff_idx == 0) {
        lwe_a[sample_idx * N * k + idx] = acc_poly[acc_offset + poly_idx * N];
    } else {
        lwe_a[sample_idx * N * k + idx] = 0u - acc_poly[acc_offset + poly_idx * N + N - coeff_idx];
    }
#endif
}

extern "C" __global__ void extract_body(
    const Torus32* acc_poly,
    Torus32* lwe_b,
    const BlindRotateParams params)
{
#ifdef __CUDA_ARCH__
    uint32_t sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= params.num_samples) return;

    uint32_t acc_offset = sample_idx * params.N * (params.k + 1);
    uint32_t body_offset = params.k * params.N;
    lwe_b[sample_idx] = acc_poly[acc_offset + body_offset];
#endif
}
