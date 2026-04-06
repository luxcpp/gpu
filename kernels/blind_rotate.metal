// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// TFHE Blind Rotation (Programmable Bootstrapping)
// Optimized for Apple Silicon GPUs

#include <metal_stdlib>
using namespace metal;

typedef uint Torus32;

struct BlindRotateParams {
    uint N;              // Polynomial degree
    uint k;              // GLWE dimension
    uint n;              // LWE dimension
    uint l;              // Decomposition levels
    uint base_log;       // Base log for decomposition
    uint num_samples;    // Batch size
};

// ============================================================================
// Polynomial Arithmetic in R_Q = Z_Q[X]/(X^N + 1)
// ============================================================================

inline void rotate_polynomial_inplace(
    device Torus32* poly,
    uint rotation,
    uint N,
    threadgroup Torus32* shared,
    uint tid  // passed from kernel
) {
    
    if (tid < N) {
        uint rot = rotation % (2 * N);
        uint dst_idx;
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
            uint r = rot - N;
            if (tid >= r) {
                dst_idx = tid - r;
                negate = true;
            } else {
                dst_idx = N - r + tid;
                negate = false;
            }
        }
        
        Torus32 val = poly[tid];
        shared[dst_idx] = negate ? (0u - val) : val;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid < N) {
        poly[tid] = shared[tid];
    }
}

// ============================================================================
// Gadget Decomposition
// ============================================================================

inline int signed_decompose(Torus32 x, uint level, uint base_log) {
    uint Bg = 1u << base_log;
    uint half_Bg = Bg >> 1u;
    uint mask = Bg - 1u;
    
    uint shift = 32u - (level + 1u) * base_log;
    uint digit = (x >> shift) & mask;
    
    if (digit >= half_Bg) {
        return int(digit) - int(Bg);
    }
    return int(digit);
}

// ============================================================================
// External Product: GLWE x GGSW -> GLWE
// ============================================================================

kernel void external_product(
    device Torus32* acc_poly [[buffer(0)]],
    device const Torus32* ggsw [[buffer(1)]],
    device Torus32* temp_poly [[buffer(2)]],
    constant BlindRotateParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint idx = gid;
    uint N = params.N;
    uint l = params.l;
    
    if (idx >= N) return;
    
    Torus32 result = 0;
    
    for (uint level = 0; level < l; level++) {
        int decomp_val = signed_decompose(acc_poly[idx], level, params.base_log);
        uint ggsw_offset = level * N + idx;
        Torus32 ggsw_coeff = ggsw[ggsw_offset];
        result += uint(decomp_val) * ggsw_coeff;
    }
    
    temp_poly[idx] = result;
}

// ============================================================================
// Blind Rotation Kernel
// ============================================================================

kernel void blind_rotate(
    device const Torus32* lwe_a [[buffer(0)]],
    device const Torus32* lwe_b [[buffer(1)]],
    device const Torus32* bsk [[buffer(2)]],
    device const Torus32* test_vector [[buffer(3)]],
    device Torus32* acc_poly [[buffer(4)]],
    constant BlindRotateParams& params [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]],
    threadgroup Torus32* shared [[threadgroup(0)]]
) {
    uint idx = gid.x;
    uint sample_idx = gid.y;
    uint tid = tid_vec.x;
    uint N = params.N;
    uint n = params.n;
    
    if (idx >= N || sample_idx >= params.num_samples) return;
    
    uint acc_offset = sample_idx * N;
    
    // Initialize accumulator with test vector
    acc_poly[acc_offset + idx] = test_vector[idx];
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Apply initial rotation by -b
    Torus32 b = lwe_b[sample_idx];
    uint log_N = 0;
    for (uint t = N; t > 1; t >>= 1) log_N++;
    uint rotation = ((b + (1u << 31u) / N) >> (32u - 1u - log_N)) % (2u * N);
    
    // Rotate test vector by -b
    // (Would need threadgroup sync and shared memory)
    
    // CMux operations for each LWE coefficient
    for (uint i = 0; i < n; i++) {
        Torus32 a_i = lwe_a[sample_idx * n + i];
        uint rot = ((a_i + (1u << 31u) / N) >> (32u - 1u - log_N)) % (2u * N);
        
        if (rot != 0) {
            // External product with BSK[i]
            threadgroup_barrier(mem_flags::mem_device);
        }
    }
}

// ============================================================================
// Sample Extraction
// ============================================================================

kernel void sample_extract(
    device const Torus32* acc_poly [[buffer(0)]],
    device Torus32* lwe_a [[buffer(1)]],
    device Torus32* lwe_b [[buffer(2)]],
    constant BlindRotateParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint idx = gid.x;
    uint sample_idx = gid.y;
    uint N = params.N;
    uint k = params.k;
    
    if (idx >= N * k || sample_idx >= params.num_samples) return;
    
    uint acc_offset = sample_idx * N * (k + 1);
    uint poly_idx = idx / N;
    uint coeff_idx = idx % N;
    
    if (coeff_idx == 0) {
        lwe_a[sample_idx * N * k + idx] = acc_poly[acc_offset + poly_idx * N];
    } else {
        lwe_a[sample_idx * N * k + idx] = 0u - acc_poly[acc_offset + poly_idx * N + N - coeff_idx];
    }
}

kernel void extract_body(
    device const Torus32* acc_poly [[buffer(0)]],
    device Torus32* lwe_b [[buffer(1)]],
    constant BlindRotateParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint sample_idx = gid;
    
    if (sample_idx >= params.num_samples) return;
    
    uint acc_offset = sample_idx * params.N * (params.k + 1);
    uint body_offset = params.k * params.N;
    
    lwe_b[sample_idx] = acc_poly[acc_offset + body_offset];
}
