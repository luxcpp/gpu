// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Modular Reduction Kernels for ML-DSA (Dilithium) and ML-KEM (Kyber)
// - Barrett reduction for q=8380417 and q=3329
// - Centered reduction to [-q/2, q/2]
// - Batch coefficient reduction
// - Montgomery conversion
//
// Part of the Lux Network GPU acceleration library

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants
// ============================================================================

// Dilithium parameters
constant int32_t DILITHIUM_Q = 8380417;
constant uint32_t DILITHIUM_QINV = 58728449;   // q^(-1) mod 2^32
constant int32_t DILITHIUM_Q_HALF = 4190208;   // (q-1)/2
constant int64_t DILITHIUM_BARRETT_V = 8396807; // floor(2^46 / q)
constant int32_t DILITHIUM_R2 = 2365951;       // 2^64 mod q (Montgomery R^2)

// Kyber parameters
constant int16_t KYBER_Q = 3329;
constant uint16_t KYBER_QINV = 62209;          // q^(-1) mod 2^16
constant int16_t KYBER_Q_HALF = 1664;          // (q-1)/2
constant int32_t KYBER_BARRETT_V = 20159;      // floor(2^26 / q) + 1
constant int16_t KYBER_R2 = 1353;              // 2^32 mod q (Montgomery R^2)

// ============================================================================
// Barrett Reduction for Dilithium (q = 8380417)
// ============================================================================

// Barrett reduction: reduce a to [0, q)
// Input: a in (-2^31, 2^31)
// Output: a mod q in [0, q)
METAL_FUNC int32_t barrett_reduce_dilithium(int32_t a) {
    int32_t t = (int32_t)((DILITHIUM_BARRETT_V * (int64_t)a) >> 46);
    t *= DILITHIUM_Q;
    return a - t;
}

// Full reduction ensuring output in [0, q)
METAL_FUNC int32_t full_reduce_dilithium(int32_t a) {
    int32_t t = barrett_reduce_dilithium(a);
    t += (t >> 31) & DILITHIUM_Q;  // If negative, add q
    t -= DILITHIUM_Q;
    t += (t >> 31) & DILITHIUM_Q;  // If negative, add q
    return t;
}

// Centered reduction: reduce to [-q/2, q/2]
METAL_FUNC int32_t centered_reduce_dilithium(int32_t a) {
    int32_t t = full_reduce_dilithium(a);
    // If t > q/2, return t - q (making it negative)
    t -= (t > DILITHIUM_Q_HALF) ? DILITHIUM_Q : 0;
    return t;
}

// ============================================================================
// Barrett Reduction for Kyber (q = 3329)
// ============================================================================

// Barrett reduction: reduce a to approximately [0, q)
METAL_FUNC int16_t barrett_reduce_kyber(int16_t a) {
    int16_t t = (int16_t)((KYBER_BARRETT_V * (int32_t)a + (1 << 25)) >> 26);
    t *= KYBER_Q;
    return a - t;
}

// Full reduction ensuring output in [0, q)
METAL_FUNC int16_t full_reduce_kyber(int16_t a) {
    int16_t t = barrett_reduce_kyber(a);
    t += (t >> 15) & KYBER_Q;  // If negative, add q
    t -= KYBER_Q;
    t += (t >> 15) & KYBER_Q;  // If negative, add q
    return t;
}

// Centered reduction: reduce to [-(q-1)/2, q/2]
METAL_FUNC int16_t centered_reduce_kyber(int16_t a) {
    int16_t t = full_reduce_kyber(a);
    t -= (t > KYBER_Q_HALF) ? KYBER_Q : 0;
    return t;
}

// ============================================================================
// Montgomery Reduction
// ============================================================================

// Montgomery reduction for Dilithium
// Input: a in [0, q^2)
// Output: a * R^(-1) mod q where R = 2^32
METAL_FUNC int32_t montgomery_reduce_dilithium(int64_t a) {
    int32_t t = (int32_t)a * (int32_t)DILITHIUM_QINV;
    return (int32_t)((a - (int64_t)t * (int64_t)DILITHIUM_Q) >> 32);
}

// Montgomery reduction for Kyber
// Input: a in [0, q^2 * 2^16)
// Output: a * R^(-1) mod q where R = 2^16
METAL_FUNC int16_t montgomery_reduce_kyber(int32_t a) {
    int16_t t = (int16_t)a * (int16_t)KYBER_QINV;
    return (int16_t)((a - (int32_t)t * (int32_t)KYBER_Q) >> 16);
}

// ============================================================================
// Batch Reduction Kernels - Dilithium
// ============================================================================

// Barrett reduction batch
kernel void reduce_barrett_dilithium(
    device int32_t* data [[buffer(0)]],
    constant uint32_t& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    data[gid] = barrett_reduce_dilithium(data[gid]);
}

// Full reduction batch (ensure [0, q))
kernel void reduce_full_dilithium(
    device int32_t* data [[buffer(0)]],
    constant uint32_t& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    data[gid] = full_reduce_dilithium(data[gid]);
}

// Centered reduction batch (to [-q/2, q/2])
kernel void reduce_centered_dilithium(
    device int32_t* data [[buffer(0)]],
    constant uint32_t& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    data[gid] = centered_reduce_dilithium(data[gid]);
}

// ============================================================================
// Batch Reduction Kernels - Kyber
// ============================================================================

// Barrett reduction batch
kernel void reduce_barrett_kyber(
    device int16_t* data [[buffer(0)]],
    constant uint32_t& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    data[gid] = barrett_reduce_kyber(data[gid]);
}

// Full reduction batch
kernel void reduce_full_kyber(
    device int16_t* data [[buffer(0)]],
    constant uint32_t& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    data[gid] = full_reduce_kyber(data[gid]);
}

// Centered reduction batch
kernel void reduce_centered_kyber(
    device int16_t* data [[buffer(0)]],
    constant uint32_t& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    data[gid] = centered_reduce_kyber(data[gid]);
}

// ============================================================================
// Montgomery Domain Conversion
// ============================================================================

// Convert to Montgomery domain: a -> a*R mod q
kernel void to_montgomery_dilithium(
    device int32_t* data [[buffer(0)]],
    constant uint32_t& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    // Multiply by R^2 then reduce to get a*R
    data[gid] = montgomery_reduce_dilithium((int64_t)data[gid] * (int64_t)DILITHIUM_R2);
}

// Convert from Montgomery domain: a*R -> a mod q
kernel void from_montgomery_dilithium(
    device int32_t* data [[buffer(0)]],
    constant uint32_t& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    data[gid] = montgomery_reduce_dilithium((int64_t)data[gid]);
}

// Convert to Montgomery domain for Kyber
kernel void to_montgomery_kyber(
    device int16_t* data [[buffer(0)]],
    constant uint32_t& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    data[gid] = montgomery_reduce_kyber((int32_t)data[gid] * (int32_t)KYBER_R2);
}

// Convert from Montgomery domain for Kyber
kernel void from_montgomery_kyber(
    device int16_t* data [[buffer(0)]],
    constant uint32_t& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    data[gid] = montgomery_reduce_kyber((int32_t)data[gid]);
}

// ============================================================================
// Compression/Decompression for Kyber
// ============================================================================

// Compress: round(2^d / q * x) mod 2^d
kernel void compress_kyber(
    device uint8_t* output [[buffer(0)]],
    device const int16_t* input [[buffer(1)]],
    constant uint32_t& size [[buffer(2)]],
    constant uint32_t& d [[buffer(3)]],  // Compression parameter (1, 4, 5, 10, 11)
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    int16_t x = full_reduce_kyber(input[gid]);
    
    // Compute round((2^d * x) / q)
    uint32_t t = (uint32_t)x << d;
    t += KYBER_Q / 2;  // For rounding
    t /= KYBER_Q;
    t &= (1u << d) - 1;
    
    output[gid] = (uint8_t)t;
}

// Decompress: round(q / 2^d * x)
kernel void decompress_kyber(
    device int16_t* output [[buffer(0)]],
    device const uint8_t* input [[buffer(1)]],
    constant uint32_t& size [[buffer(2)]],
    constant uint32_t& d [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    uint32_t x = (uint32_t)input[gid];
    
    // Compute round((q * x) / 2^d)
    uint32_t t = x * KYBER_Q + (1u << (d - 1));
    t >>= d;
    
    output[gid] = (int16_t)t;
}

// ============================================================================
// Dilithium-Specific Reductions
// ============================================================================

// HighBits: extract high d bits after dividing by 2*gamma2
kernel void highbits_dilithium(
    device int32_t* output [[buffer(0)]],
    device const int32_t* input [[buffer(1)]],
    constant uint32_t& size [[buffer(2)]],
    constant int32_t& gamma2 [[buffer(3)]],  // gamma2 = (q-1)/(2*k)
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    int32_t a = full_reduce_dilithium(input[gid]);
    int32_t two_gamma2 = 2 * gamma2;
    
    // a0 = a mod 2*gamma2 (centered)
    int32_t a0 = a % two_gamma2;
    if (a0 > gamma2) a0 -= two_gamma2;
    
    // a1 = (a - a0) / (2*gamma2)
    // But we need to handle the edge case where a0 = q - 1
    if (a - a0 == DILITHIUM_Q - 1) {
        output[gid] = 0;
    } else {
        output[gid] = (a - a0) / two_gamma2;
    }
}

// LowBits: extract low bits after modular reduction
kernel void lowbits_dilithium(
    device int32_t* output [[buffer(0)]],
    device const int32_t* input [[buffer(1)]],
    constant uint32_t& size [[buffer(2)]],
    constant int32_t& gamma2 [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    int32_t a = full_reduce_dilithium(input[gid]);
    int32_t two_gamma2 = 2 * gamma2;
    
    // a0 = a mod 2*gamma2 (centered in [-gamma2, gamma2])
    int32_t a0 = a % two_gamma2;
    if (a0 > gamma2) a0 -= two_gamma2;
    
    // Handle edge case
    if (a - a0 == DILITHIUM_Q - 1) {
        a0 -= 1;
    }
    
    output[gid] = a0;
}

// MakeHint: compute hint bit for verification
kernel void make_hint_dilithium(
    device uint8_t* hint [[buffer(0)]],
    device const int32_t* z [[buffer(1)]],      // Low bits of w - c*s2
    device const int32_t* r [[buffer(2)]],      // Low bits of w
    constant uint32_t& size [[buffer(3)]],
    constant int32_t& gamma2 [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    int32_t z0 = z[gid];
    int32_t r0 = r[gid];
    
    // Hint is 1 if HighBits differ
    // Simplified: hint = 1 if |z0| > gamma2 or |r0| > gamma2
    bool h = (z0 > gamma2) || (z0 < -gamma2) || (r0 > gamma2) || (r0 < -gamma2);
    
    hint[gid] = h ? 1 : 0;
}

// UseHint: recover high bits using hint
kernel void use_hint_dilithium(
    device int32_t* output [[buffer(0)]],
    device const int32_t* input [[buffer(1)]],
    device const uint8_t* hint [[buffer(2)]],
    constant uint32_t& size [[buffer(3)]],
    constant int32_t& gamma2 [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    int32_t a = full_reduce_dilithium(input[gid]);
    int32_t two_gamma2 = 2 * gamma2;
    
    int32_t a0 = a % two_gamma2;
    if (a0 > gamma2) a0 -= two_gamma2;
    
    int32_t a1;
    if (a - a0 == DILITHIUM_Q - 1) {
        a1 = 0;
    } else {
        a1 = (a - a0) / two_gamma2;
    }
    
    // Apply hint
    if (hint[gid]) {
        int32_t max_a1 = (DILITHIUM_Q - 1) / two_gamma2;
        if (a0 > 0) {
            a1 = (a1 + 1) % (max_a1 + 1);
        } else {
            a1 = (a1 + max_a1) % (max_a1 + 1);
        }
    }
    
    output[gid] = a1;
}

// ============================================================================
// Freeze: Ensure coefficients are in canonical form [0, q)
// ============================================================================

kernel void freeze_dilithium(
    device int32_t* data [[buffer(0)]],
    constant uint32_t& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    data[gid] = full_reduce_dilithium(data[gid]);
}

kernel void freeze_kyber(
    device int16_t* data [[buffer(0)]],
    constant uint32_t& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    data[gid] = full_reduce_kyber(data[gid]);
}

// ============================================================================
// Conditional Subtraction (for lazy reduction)
// ============================================================================

// Subtract q if coefficient >= q (lazy reduction cleanup)
kernel void cond_sub_q_dilithium(
    device int32_t* data [[buffer(0)]],
    constant uint32_t& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    int32_t a = data[gid];
    a -= (a >= DILITHIUM_Q) ? DILITHIUM_Q : 0;
    data[gid] = a;
}

kernel void cond_sub_q_kyber(
    device int16_t* data [[buffer(0)]],
    constant uint32_t& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    int16_t a = data[gid];
    a -= (a >= KYBER_Q) ? KYBER_Q : 0;
    data[gid] = a;
}
