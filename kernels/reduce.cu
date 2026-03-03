// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Modular Reduction Kernels for ML-DSA (Dilithium) and ML-KEM (Kyber) - CUDA
// Matches reduce.metal output byte-for-byte
//
// - Barrett reduction for q=8380417 and q=3329
// - Centered reduction to [-q/2, q/2]
// - Batch coefficient reduction
// - Montgomery conversion
// - Compression/decompression (Kyber)
// - HighBits/LowBits/Hints (Dilithium)

#include <cstdint>

#ifdef __CUDA_ARCH__

// ============================================================================
// Constants
// ============================================================================

// Dilithium parameters
static __device__ const int32_t DILITHIUM_Q = 8380417;
static __device__ const uint32_t DILITHIUM_QINV = 58728449;   // q^(-1) mod 2^32
static __device__ const int32_t DILITHIUM_Q_HALF = 4190208;   // (q-1)/2
static __device__ const int64_t DILITHIUM_BARRETT_V = 8396807; // floor(2^46 / q)
static __device__ const int32_t DILITHIUM_R2 = 2365951;       // 2^64 mod q (Montgomery R^2)

// Kyber parameters
static __device__ const int16_t KYBER_Q = 3329;
static __device__ const uint16_t KYBER_QINV = 62209;          // q^(-1) mod 2^16
static __device__ const int16_t KYBER_Q_HALF = 1664;          // (q-1)/2
static __device__ const int32_t KYBER_BARRETT_V = 20159;      // floor(2^26 / q) + 1
static __device__ const int16_t KYBER_R2 = 1353;              // 2^32 mod q (Montgomery R^2)

// ============================================================================
// Barrett Reduction for Dilithium (q = 8380417)
// ============================================================================

__device__ __forceinline__
int32_t barrett_reduce_dilithium(int32_t a) {
    int32_t t = (int32_t)((DILITHIUM_BARRETT_V * (int64_t)a) >> 46);
    t *= DILITHIUM_Q;
    return a - t;
}

__device__ __forceinline__
int32_t full_reduce_dilithium(int32_t a) {
    int32_t t = barrett_reduce_dilithium(a);
    t += (t >> 31) & DILITHIUM_Q;
    t -= DILITHIUM_Q;
    t += (t >> 31) & DILITHIUM_Q;
    return t;
}

__device__ __forceinline__
int32_t centered_reduce_dilithium(int32_t a) {
    int32_t t = full_reduce_dilithium(a);
    t -= (t > DILITHIUM_Q_HALF) ? DILITHIUM_Q : 0;
    return t;
}

// ============================================================================
// Barrett Reduction for Kyber (q = 3329)
// ============================================================================

__device__ __forceinline__
int16_t barrett_reduce_kyber(int16_t a) {
    int16_t t = (int16_t)((KYBER_BARRETT_V * (int32_t)a + (1 << 25)) >> 26);
    t *= KYBER_Q;
    return a - t;
}

__device__ __forceinline__
int16_t full_reduce_kyber(int16_t a) {
    int16_t t = barrett_reduce_kyber(a);
    t += (t >> 15) & KYBER_Q;
    t -= KYBER_Q;
    t += (t >> 15) & KYBER_Q;
    return t;
}

__device__ __forceinline__
int16_t centered_reduce_kyber(int16_t a) {
    int16_t t = full_reduce_kyber(a);
    t -= (t > KYBER_Q_HALF) ? KYBER_Q : 0;
    return t;
}

// ============================================================================
// Montgomery Reduction
// ============================================================================

__device__ __forceinline__
int32_t montgomery_reduce_dilithium(int64_t a) {
    int32_t t = (int32_t)a * (int32_t)DILITHIUM_QINV;
    return (int32_t)((a - (int64_t)t * (int64_t)DILITHIUM_Q) >> 32);
}

__device__ __forceinline__
int16_t montgomery_reduce_kyber(int32_t a) {
    int16_t t = (int16_t)a * (int16_t)KYBER_QINV;
    return (int16_t)((a - (int32_t)t * (int32_t)KYBER_Q) >> 16);
}

// ============================================================================
// Batch Reduction Kernels - Dilithium
// ============================================================================

extern "C" __global__
void reduce_barrett_dilithium(int32_t* data, uint32_t size) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    data[gid] = barrett_reduce_dilithium(data[gid]);
}

extern "C" __global__
void reduce_full_dilithium(int32_t* data, uint32_t size) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    data[gid] = full_reduce_dilithium(data[gid]);
}

extern "C" __global__
void reduce_centered_dilithium(int32_t* data, uint32_t size) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    data[gid] = centered_reduce_dilithium(data[gid]);
}

// ============================================================================
// Batch Reduction Kernels - Kyber
// ============================================================================

extern "C" __global__
void reduce_barrett_kyber(int16_t* data, uint32_t size) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    data[gid] = barrett_reduce_kyber(data[gid]);
}

extern "C" __global__
void reduce_full_kyber(int16_t* data, uint32_t size) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    data[gid] = full_reduce_kyber(data[gid]);
}

extern "C" __global__
void reduce_centered_kyber(int16_t* data, uint32_t size) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    data[gid] = centered_reduce_kyber(data[gid]);
}

// ============================================================================
// Montgomery Domain Conversion
// ============================================================================

extern "C" __global__
void to_montgomery_dilithium(int32_t* data, uint32_t size) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    data[gid] = montgomery_reduce_dilithium((int64_t)data[gid] * (int64_t)DILITHIUM_R2);
}

extern "C" __global__
void from_montgomery_dilithium(int32_t* data, uint32_t size) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    data[gid] = montgomery_reduce_dilithium((int64_t)data[gid]);
}

extern "C" __global__
void to_montgomery_kyber(int16_t* data, uint32_t size) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    data[gid] = montgomery_reduce_kyber((int32_t)data[gid] * (int32_t)KYBER_R2);
}

extern "C" __global__
void from_montgomery_kyber(int16_t* data, uint32_t size) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    data[gid] = montgomery_reduce_kyber((int32_t)data[gid]);
}

// ============================================================================
// Compression/Decompression for Kyber
// ============================================================================

extern "C" __global__
void compress_kyber(uint8_t* output, const int16_t* input,
                    uint32_t size, uint32_t d) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;

    int16_t x = full_reduce_kyber(input[gid]);

    // round((2^d * x) / q)
    uint32_t t = (uint32_t)x << d;
    t += KYBER_Q / 2;  // rounding
    t /= KYBER_Q;
    t &= (1u << d) - 1;

    output[gid] = (uint8_t)t;
}

extern "C" __global__
void decompress_kyber(int16_t* output, const uint8_t* input,
                      uint32_t size, uint32_t d) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;

    uint32_t x = (uint32_t)input[gid];

    // round((q * x) / 2^d)
    uint32_t t = x * KYBER_Q + (1u << (d - 1));
    t >>= d;

    output[gid] = (int16_t)t;
}

// ============================================================================
// Dilithium-Specific Reductions
// ============================================================================

extern "C" __global__
void highbits_dilithium(int32_t* output, const int32_t* input,
                        uint32_t size, int32_t gamma2) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;

    int32_t a = full_reduce_dilithium(input[gid]);
    int32_t two_gamma2 = 2 * gamma2;

    int32_t a0 = a % two_gamma2;
    if (a0 > gamma2) a0 -= two_gamma2;

    if (a - a0 == DILITHIUM_Q - 1) {
        output[gid] = 0;
    } else {
        output[gid] = (a - a0) / two_gamma2;
    }
}

extern "C" __global__
void lowbits_dilithium(int32_t* output, const int32_t* input,
                       uint32_t size, int32_t gamma2) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;

    int32_t a = full_reduce_dilithium(input[gid]);
    int32_t two_gamma2 = 2 * gamma2;

    int32_t a0 = a % two_gamma2;
    if (a0 > gamma2) a0 -= two_gamma2;

    if (a - a0 == DILITHIUM_Q - 1) {
        a0 -= 1;
    }

    output[gid] = a0;
}

extern "C" __global__
void make_hint_dilithium(uint8_t* hint, const int32_t* z, const int32_t* r,
                         uint32_t size, int32_t gamma2) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;

    int32_t z0 = z[gid];
    int32_t r0 = r[gid];

    bool h = (z0 > gamma2) || (z0 < -gamma2) || (r0 > gamma2) || (r0 < -gamma2);
    hint[gid] = h ? 1 : 0;
}

extern "C" __global__
void use_hint_dilithium(int32_t* output, const int32_t* input,
                        const uint8_t* hint, uint32_t size, int32_t gamma2) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
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

extern "C" __global__
void freeze_dilithium(int32_t* data, uint32_t size) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    data[gid] = full_reduce_dilithium(data[gid]);
}

extern "C" __global__
void freeze_kyber(int16_t* data, uint32_t size) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    data[gid] = full_reduce_kyber(data[gid]);
}

// ============================================================================
// Conditional Subtraction (lazy reduction cleanup)
// ============================================================================

extern "C" __global__
void cond_sub_q_dilithium(int32_t* data, uint32_t size) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    int32_t a = data[gid];
    a -= (a >= DILITHIUM_Q) ? DILITHIUM_Q : 0;
    data[gid] = a;
}

extern "C" __global__
void cond_sub_q_kyber(int16_t* data, uint32_t size) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    int16_t a = data[gid];
    a -= (a >= KYBER_Q) ? KYBER_Q : 0;
    data[gid] = a;
}

#endif // __CUDA_ARCH__
