// PAT-FHE-012: EVM256 Parallel Processing - CUDA Implementation
// uint256 operations using 4x uint64 limbs (little-endian)
// Matches evm256.metal output byte-for-byte
//
// Copyright (C) 2024-2026 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>

#ifdef __CUDA_ARCH__

#define LIMBS 4

struct uint256_t {
    unsigned long long limbs[LIMBS];
};

// ============================================================================
// Helper functions for multi-limb arithmetic
// ============================================================================

__device__ __forceinline__
unsigned long long add_carry(unsigned long long a, unsigned long long b,
                             unsigned long long* carry) {
    unsigned long long sum = a + b;
    unsigned long long c1 = sum < a ? 1ULL : 0ULL;
    unsigned long long result = sum + *carry;
    unsigned long long c2 = result < sum ? 1ULL : 0ULL;
    *carry = c1 | c2;
    return result;
}

__device__ __forceinline__
unsigned long long sub_borrow(unsigned long long a, unsigned long long b,
                              unsigned long long* borrow) {
    unsigned long long diff = a - b;
    unsigned long long b1 = diff > a ? 1ULL : 0ULL;
    unsigned long long result = diff - *borrow;
    unsigned long long b2 = result > diff ? 1ULL : 0ULL;
    *borrow = b1 | b2;
    return result;
}

// Full 64x64 -> 128 bit multiplication
__device__ __forceinline__
void mul64_wide(unsigned long long a, unsigned long long b,
                unsigned long long* hi, unsigned long long* lo) {
    unsigned long long a_lo = a & 0xFFFFFFFFULL;
    unsigned long long a_hi = a >> 32;
    unsigned long long b_lo = b & 0xFFFFFFFFULL;
    unsigned long long b_hi = b >> 32;

    unsigned long long p0 = a_lo * b_lo;
    unsigned long long p1 = a_lo * b_hi;
    unsigned long long p2 = a_hi * b_lo;
    unsigned long long p3 = a_hi * b_hi;

    unsigned long long cy = ((p0 >> 32) + (p1 & 0xFFFFFFFFULL) + (p2 & 0xFFFFFFFFULL)) >> 32;
    *lo = p0 + (p1 << 32) + (p2 << 32);
    *hi = p3 + (p1 >> 32) + (p2 >> 32) + cy;
}

// ============================================================================
// Compare and zero-check helpers
// ============================================================================

__device__ __forceinline__
int cmp256(const uint256_t* a, const uint256_t* b) {
    for (int i = LIMBS - 1; i >= 0; i--) {
        if (a->limbs[i] > b->limbs[i]) return 1;
        if (a->limbs[i] < b->limbs[i]) return -1;
    }
    return 0;
}

__device__ __forceinline__
bool is_zero(const uint256_t* a) {
    for (int i = 0; i < LIMBS; i++) {
        if (a->limbs[i] != 0) return false;
    }
    return true;
}

// ============================================================================
// Div256 Implementation (long division)
// ============================================================================

__device__ __forceinline__
void div256_impl(const uint256_t* numerator, const uint256_t* denominator,
                 uint256_t* quotient, uint256_t* remainder) {
    if (is_zero(denominator)) {
        for (int i = 0; i < LIMBS; i++) {
            quotient->limbs[i] = 0;
            remainder->limbs[i] = 0;
        }
        return;
    }

    uint256_t q, r;
    for (int i = 0; i < LIMBS; i++) {
        q.limbs[i] = 0;
        r.limbs[i] = 0;
    }

    for (int i = 255; i >= 0; i--) {
        // r <<= 1
        unsigned long long carry = 0;
        for (int j = 0; j < LIMBS; j++) {
            unsigned long long temp = (r.limbs[j] << 1) | carry;
            carry = r.limbs[j] >> 63;
            r.limbs[j] = temp;
        }

        // r[0] |= numerator bit i
        int limb_idx = i / 64;
        int bit_idx = i % 64;
        unsigned long long bit = (numerator->limbs[limb_idx] >> bit_idx) & 1ULL;
        r.limbs[0] |= bit;

        // if r >= denominator
        if (cmp256(&r, denominator) >= 0) {
            // r -= denominator
            unsigned long long borrow = 0;
            for (int j = 0; j < LIMBS; j++) {
                r.limbs[j] = sub_borrow(r.limbs[j], denominator->limbs[j], &borrow);
            }

            // q[i] = 1
            limb_idx = i / 64;
            bit_idx = i % 64;
            q.limbs[limb_idx] |= (1ULL << bit_idx);
        }
    }

    *quotient = q;
    *remainder = r;
}

// ============================================================================
// Montgomery reduction
// ============================================================================

__device__ __forceinline__
void montgomery_reduce(const unsigned long long* t, const uint256_t* m,
                       unsigned long long m_inv, uint256_t* result) {
    unsigned long long a[LIMBS * 2];
    for (int i = 0; i < LIMBS * 2; i++) {
        a[i] = t[i];
    }

    for (int i = 0; i < LIMBS; i++) {
        unsigned long long u = a[i] * m_inv;
        unsigned long long carry = 0;

        for (int j = 0; j < LIMBS; j++) {
            unsigned long long hi, lo;
            mul64_wide(u, m->limbs[j], &hi, &lo);

            unsigned long long sum = a[i + j] + lo + carry;
            carry = (sum < a[i + j]) ? 1ULL : 0ULL;
            carry += hi;
            a[i + j] = sum;
        }

        for (int j = LIMBS; j < LIMBS * 2 - i && carry; j++) {
            unsigned long long sum = a[i + j] + carry;
            carry = (sum < a[i + j]) ? 1ULL : 0ULL;
            a[i + j] = sum;
        }
    }

    // Result is in upper half
    bool needs_sub = false;
    for (int i = LIMBS - 1; i >= 0; i--) {
        if (a[LIMBS + i] > m->limbs[i]) {
            needs_sub = true;
            break;
        }
        if (a[LIMBS + i] < m->limbs[i]) break;
    }

    if (needs_sub) {
        unsigned long long borrow = 0;
        for (int i = 0; i < LIMBS; i++) {
            result->limbs[i] = sub_borrow(a[LIMBS + i], m->limbs[i], &borrow);
        }
    } else {
        for (int i = 0; i < LIMBS; i++) {
            result->limbs[i] = a[LIMBS + i];
        }
    }
}

// ============================================================================
// Kernel: Batch Add256
// ============================================================================

extern "C" __global__
void cuda_add256(
    const uint256_t* __restrict__ a,
    const uint256_t* __restrict__ b,
    uint256_t* __restrict__ result,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    unsigned long long carry = 0;
    for (int i = 0; i < LIMBS; i++) {
        result[idx].limbs[i] = add_carry(a[idx].limbs[i], b[idx].limbs[i], &carry);
    }
}

// ============================================================================
// Kernel: Batch Sub256
// ============================================================================

extern "C" __global__
void cuda_sub256(
    const uint256_t* __restrict__ a,
    const uint256_t* __restrict__ b,
    uint256_t* __restrict__ result,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    unsigned long long borrow = 0;
    for (int i = 0; i < LIMBS; i++) {
        result[idx].limbs[i] = sub_borrow(a[idx].limbs[i], b[idx].limbs[i], &borrow);
    }
}

// ============================================================================
// Kernel: Batch Mul256
// ============================================================================

extern "C" __global__
void cuda_mul256(
    const uint256_t* __restrict__ a,
    const uint256_t* __restrict__ b,
    uint256_t* __restrict__ result,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Schoolbook multiplication with 8-limb intermediate result
    unsigned long long product[LIMBS * 2];
    for (int i = 0; i < LIMBS * 2; i++) {
        product[i] = 0;
    }

    for (int i = 0; i < LIMBS; i++) {
        unsigned long long carry = 0;
        for (int j = 0; j < LIMBS; j++) {
            unsigned long long hi, lo;
            mul64_wide(a[idx].limbs[i], b[idx].limbs[j], &hi, &lo);

            unsigned long long sum = product[i + j] + lo + carry;
            carry = (sum < product[i + j]) ? 1ULL : 0ULL;
            carry += hi;
            product[i + j] = sum;
        }
        product[i + LIMBS] += carry;
    }

    // Take lower 256 bits
    for (int i = 0; i < LIMBS; i++) {
        result[idx].limbs[i] = product[i];
    }
}

// ============================================================================
// Kernel: Batch Div256
// ============================================================================

extern "C" __global__
void cuda_div256(
    const uint256_t* __restrict__ a,
    const uint256_t* __restrict__ b,
    uint256_t* __restrict__ result,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint256_t numerator = a[idx];
    uint256_t denominator = b[idx];
    uint256_t quotient, remainder;

    div256_impl(&numerator, &denominator, &quotient, &remainder);
    result[idx] = quotient;
}

// ============================================================================
// Kernel: Batch Mod256
// ============================================================================

extern "C" __global__
void cuda_mod256(
    const uint256_t* __restrict__ a,
    const uint256_t* __restrict__ b,
    uint256_t* __restrict__ result,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint256_t numerator = a[idx];
    uint256_t denominator = b[idx];
    uint256_t quotient, remainder;

    div256_impl(&numerator, &denominator, &quotient, &remainder);
    result[idx] = remainder;
}

// ============================================================================
// Kernel: Montgomery Multiplication
// ============================================================================

extern "C" __global__
void cuda_mont_mul(
    const uint256_t* __restrict__ a,
    const uint256_t* __restrict__ b,
    const uint256_t* __restrict__ m,
    const unsigned long long* __restrict__ m_inv,
    uint256_t* __restrict__ result,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Compute full 512-bit product
    unsigned long long product[LIMBS * 2];
    for (int i = 0; i < LIMBS * 2; i++) {
        product[i] = 0;
    }

    for (int i = 0; i < LIMBS; i++) {
        unsigned long long carry = 0;
        for (int j = 0; j < LIMBS; j++) {
            unsigned long long hi, lo;
            mul64_wide(a[idx].limbs[i], b[idx].limbs[j], &hi, &lo);

            unsigned long long sum = product[i + j] + lo + carry;
            carry = (sum < product[i + j]) ? 1ULL : 0ULL;
            carry += hi;
            product[i + j] = sum;
        }
        product[i + LIMBS] += carry;
    }

    // Montgomery reduce
    uint256_t mod = m[idx];
    uint256_t res;
    montgomery_reduce(product, &mod, *m_inv, &res);
    result[idx] = res;
}

// ============================================================================
// Kernel: Modular Exponentiation (square-and-multiply)
// ============================================================================

extern "C" __global__
void cuda_modexp256(
    const uint256_t* __restrict__ base,
    const uint256_t* __restrict__ exponent,
    const uint256_t* __restrict__ modulus,
    uint256_t* __restrict__ result,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint256_t res;
    res.limbs[0] = 1;
    res.limbs[1] = 0;
    res.limbs[2] = 0;
    res.limbs[3] = 0;

    uint256_t b = base[idx];
    uint256_t mod = modulus[idx];

    for (int i = 0; i < 256; i++) {
        int limb_idx = i / 64;
        int bit_idx = i % 64;
        unsigned long long bit = (exponent[idx].limbs[limb_idx] >> bit_idx) & 1ULL;

        if (bit) {
            // res = (res * b) % modulus
            unsigned long long product[LIMBS * 2];
            for (int k = 0; k < LIMBS * 2; k++) {
                product[k] = 0;
            }

            for (int j = 0; j < LIMBS; j++) {
                unsigned long long carry = 0;
                for (int k = 0; k < LIMBS; k++) {
                    unsigned long long hi, lo;
                    mul64_wide(res.limbs[j], b.limbs[k], &hi, &lo);

                    unsigned long long sum = product[j + k] + lo + carry;
                    carry = (sum < product[j + k]) ? 1ULL : 0ULL;
                    carry += hi;
                    product[j + k] = sum;
                }
                product[j + LIMBS] += carry;
            }

            uint256_t temp;
            for (int j = 0; j < LIMBS; j++) {
                temp.limbs[j] = product[j];
            }

            uint256_t quot, rem;
            div256_impl(&temp, &mod, &quot, &rem);
            res = rem;
        }

        // b = (b * b) % modulus
        unsigned long long product[LIMBS * 2];
        for (int k = 0; k < LIMBS * 2; k++) {
            product[k] = 0;
        }

        for (int j = 0; j < LIMBS; j++) {
            unsigned long long carry = 0;
            for (int k = 0; k < LIMBS; k++) {
                unsigned long long hi, lo;
                mul64_wide(b.limbs[j], b.limbs[k], &hi, &lo);

                unsigned long long sum = product[j + k] + lo + carry;
                carry = (sum < product[j + k]) ? 1ULL : 0ULL;
                carry += hi;
                product[j + k] = sum;
            }
            product[j + LIMBS] += carry;
        }

        uint256_t temp;
        for (int j = 0; j < LIMBS; j++) {
            temp.limbs[j] = product[j];
        }

        uint256_t quot, rem;
        div256_impl(&temp, &mod, &quot, &rem);
        b = rem;
    }

    result[idx] = res;
}

#endif // __CUDA_ARCH__
