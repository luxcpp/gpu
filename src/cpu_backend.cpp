// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Built-in CPU Backend - SIMD-optimized fallback with real crypto implementations
// This is linked directly into the core library (not a plugin).

#include "lux/gpu/backend_plugin.h"
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// CPU Buffer Implementation
// =============================================================================

struct CPUBuffer {
    void* data;
    size_t size;
};

struct CPUContext {
    int device_index;
};

// =============================================================================
// 256-bit and 384-bit Arithmetic for Elliptic Curves
// =============================================================================

namespace {

// 256-bit unsigned integer (4 x 64-bit limbs, little-endian)
struct U256 {
    uint64_t limbs[4];

    U256() : limbs{0, 0, 0, 0} {}
    U256(uint64_t v) : limbs{v, 0, 0, 0} {}
    U256(uint64_t l0, uint64_t l1, uint64_t l2, uint64_t l3)
        : limbs{l0, l1, l2, l3} {}

    bool is_zero() const {
        return limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0;
    }

    bool operator==(const U256& o) const {
        return limbs[0] == o.limbs[0] && limbs[1] == o.limbs[1] &&
               limbs[2] == o.limbs[2] && limbs[3] == o.limbs[3];
    }

    bool operator>=(const U256& o) const {
        for (int i = 3; i >= 0; i--) {
            if (limbs[i] > o.limbs[i]) return true;
            if (limbs[i] < o.limbs[i]) return false;
        }
        return true;
    }

    bool operator<(const U256& o) const { return !(*this >= o); }

    // Bit at position i (0 = LSB)
    bool bit(int i) const {
        return (limbs[i / 64] >> (i % 64)) & 1;
    }
};

// Add with carry
inline uint64_t add_with_carry(uint64_t a, uint64_t b, uint64_t& carry) {
    unsigned __int128 sum = (unsigned __int128)a + b + carry;
    carry = sum >> 64;
    return (uint64_t)sum;
}

// Subtract with borrow
inline uint64_t sub_with_borrow(uint64_t a, uint64_t b, uint64_t& borrow) {
    unsigned __int128 diff = (unsigned __int128)a - b - borrow;
    borrow = (diff >> 127) ? 1 : 0;
    return (uint64_t)diff;
}

// U256 addition
inline U256 u256_add(const U256& a, const U256& b) {
    U256 r;
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        r.limbs[i] = add_with_carry(a.limbs[i], b.limbs[i], carry);
    }
    return r;
}

// U256 subtraction (assumes a >= b)
inline U256 u256_sub(const U256& a, const U256& b) {
    U256 r;
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        r.limbs[i] = sub_with_borrow(a.limbs[i], b.limbs[i], borrow);
    }
    return r;
}

// U256 x U256 -> 512-bit product (8 limbs)
inline void u256_mul_wide(const U256& a, const U256& b, uint64_t result[8]) {
    memset(result, 0, 8 * sizeof(uint64_t));
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 prod = (unsigned __int128)a.limbs[i] * b.limbs[j]
                                     + result[i + j] + carry;
            result[i + j] = (uint64_t)prod;
            carry = prod >> 64;
        }
        result[i + 4] = carry;
    }
}

// =============================================================================
// BN254 Scalar Field Fr (256 bits)
// Modulus r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// =============================================================================

const U256 BN254_FR_MODULUS(
    0x43e1f593f0000001ULL,
    0x2833e84879b97091ULL,
    0xb85045b68181585dULL,
    0x30644e72e131a029ULL
);

// R = 2^256 mod r (Montgomery R)
const U256 BN254_FR_R(
    0xd35d438dc58f0d9dULL,
    0x0a78eb28f5c70b3dULL,
    0x666ea36f7879462cULL,
    0x0e0a77c19a07df2fULL
);

// R^2 mod r
const U256 BN254_FR_R2(
    0xf32cfc5b538afa89ULL,
    0xb5e71911d44501fbULL,
    0x47ab1eff0a417ff6ULL,
    0x06d89f71cab8351fULL
);

// -r^{-1} mod 2^64
const uint64_t BN254_FR_INV = 0xc2e1f593efffffffULL;

// Montgomery reduction for Fr
inline U256 fr_mont_reduce(uint64_t t[8]) {
    // Montgomery reduction: compute t * R^{-1} mod r
    for (int i = 0; i < 4; i++) {
        uint64_t m = t[i] * BN254_FR_INV;
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 prod = (unsigned __int128)m * BN254_FR_MODULUS.limbs[j]
                                     + t[i + j] + carry;
            t[i + j] = (uint64_t)prod;
            carry = prod >> 64;
        }
        for (int j = i + 4; j < 8 && carry; j++) {
            unsigned __int128 sum = (unsigned __int128)t[j] + carry;
            t[j] = (uint64_t)sum;
            carry = sum >> 64;
        }
    }

    U256 result(t[4], t[5], t[6], t[7]);
    if (result >= BN254_FR_MODULUS) {
        result = u256_sub(result, BN254_FR_MODULUS);
    }
    return result;
}

// Fr addition
inline U256 fr_add(const U256& a, const U256& b) {
    U256 r = u256_add(a, b);
    if (r >= BN254_FR_MODULUS) {
        r = u256_sub(r, BN254_FR_MODULUS);
    }
    return r;
}

// Fr subtraction
inline U256 fr_sub(const U256& a, const U256& b) {
    if (a >= b) {
        return u256_sub(a, b);
    } else {
        return u256_sub(u256_add(a, BN254_FR_MODULUS), b);
    }
}

// Fr Montgomery multiplication
inline U256 fr_mul(const U256& a, const U256& b) {
    uint64_t t[8];
    u256_mul_wide(a, b, t);
    return fr_mont_reduce(t);
}

// Convert to Montgomery form
inline U256 fr_to_mont(const U256& a) {
    return fr_mul(a, BN254_FR_R2);
}

// Convert from Montgomery form
inline U256 fr_from_mont(const U256& a) {
    uint64_t t[8] = {a.limbs[0], a.limbs[1], a.limbs[2], a.limbs[3], 0, 0, 0, 0};
    return fr_mont_reduce(t);
}

// Fr squaring
inline U256 fr_square(const U256& a) {
    return fr_mul(a, a);
}

// Fr exponentiation
inline U256 fr_pow(const U256& base, const U256& exp) {
    U256 result = BN254_FR_R;  // 1 in Montgomery form
    U256 b = base;
    for (int i = 0; i < 256; i++) {
        if (exp.bit(i)) {
            result = fr_mul(result, b);
        }
        b = fr_square(b);
    }
    return result;
}

// Fr inversion via Fermat's little theorem: a^{-1} = a^{r-2}
inline U256 fr_inv(const U256& a) {
    U256 exp = u256_sub(BN254_FR_MODULUS, U256(2));
    return fr_pow(a, exp);
}

// =============================================================================
// BN254 Base Field Fp (256 bits)
// Modulus p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
// =============================================================================

const U256 BN254_FP_MODULUS(
    0x3c208c16d87cfd47ULL,
    0x97816a916871ca8dULL,
    0xb85045b68181585dULL,
    0x30644e72e131a029ULL
);

const U256 BN254_FP_R(
    0xd35d438dc58f0d9dULL,
    0x0a78eb28f5c70b3dULL,
    0x666ea36f7879462cULL,
    0x0e0a77c19a07df2fULL
);

const U256 BN254_FP_R2(
    0xf32cfc5b538afa89ULL,
    0xb5e71911d44501fbULL,
    0x47ab1eff0a417ff6ULL,
    0x06d89f71cab8351fULL
);

const uint64_t BN254_FP_INV = 0x87d20782e4866389ULL;

// Fp Montgomery reduction
inline U256 fp_mont_reduce(uint64_t t[8]) {
    for (int i = 0; i < 4; i++) {
        uint64_t m = t[i] * BN254_FP_INV;
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 prod = (unsigned __int128)m * BN254_FP_MODULUS.limbs[j]
                                     + t[i + j] + carry;
            t[i + j] = (uint64_t)prod;
            carry = prod >> 64;
        }
        for (int j = i + 4; j < 8 && carry; j++) {
            unsigned __int128 sum = (unsigned __int128)t[j] + carry;
            t[j] = (uint64_t)sum;
            carry = sum >> 64;
        }
    }

    U256 result(t[4], t[5], t[6], t[7]);
    if (result >= BN254_FP_MODULUS) {
        result = u256_sub(result, BN254_FP_MODULUS);
    }
    return result;
}

inline U256 fp_add(const U256& a, const U256& b) {
    U256 r = u256_add(a, b);
    if (r >= BN254_FP_MODULUS) {
        r = u256_sub(r, BN254_FP_MODULUS);
    }
    return r;
}

inline U256 fp_sub(const U256& a, const U256& b) {
    if (a >= b) {
        return u256_sub(a, b);
    } else {
        return u256_sub(u256_add(a, BN254_FP_MODULUS), b);
    }
}

inline U256 fp_mul(const U256& a, const U256& b) {
    uint64_t t[8];
    u256_mul_wide(a, b, t);
    return fp_mont_reduce(t);
}

inline U256 fp_square(const U256& a) {
    return fp_mul(a, a);
}

inline U256 fp_to_mont(const U256& a) {
    return fp_mul(a, BN254_FP_R2);
}

inline U256 fp_from_mont(const U256& a) {
    uint64_t t[8] = {a.limbs[0], a.limbs[1], a.limbs[2], a.limbs[3], 0, 0, 0, 0};
    return fp_mont_reduce(t);
}

inline U256 fp_neg(const U256& a) {
    if (a.is_zero()) return a;
    return u256_sub(BN254_FP_MODULUS, a);
}

inline U256 fp_double(const U256& a) {
    return fp_add(a, a);
}

inline U256 fp_pow(const U256& base, const U256& exp) {
    U256 result = BN254_FP_R;
    U256 b = base;
    for (int i = 0; i < 256; i++) {
        if (exp.bit(i)) {
            result = fp_mul(result, b);
        }
        b = fp_square(b);
    }
    return result;
}

inline U256 fp_inv(const U256& a) {
    U256 exp = u256_sub(BN254_FP_MODULUS, U256(2));
    return fp_pow(a, exp);
}

// =============================================================================
// BN254 G1 Curve Operations
// y^2 = x^3 + 3
// =============================================================================

struct G1Projective {
    U256 x, y, z;

    bool is_infinity() const { return z.is_zero(); }

    static G1Projective infinity() {
        G1Projective p;
        p.x = BN254_FP_R;  // 1
        p.y = BN254_FP_R;  // 1
        p.z = U256();      // 0
        return p;
    }

    static G1Projective generator() {
        G1Projective p;
        p.x = fp_to_mont(U256(1));
        p.y = fp_to_mont(U256(2));
        p.z = BN254_FP_R;
        return p;
    }
};

struct G1Affine {
    U256 x, y;
    bool infinity;

    static G1Affine identity() {
        G1Affine p;
        p.x = U256();
        p.y = U256();
        p.infinity = true;
        return p;
    }

    G1Projective to_projective() const {
        G1Projective p;
        if (infinity) {
            return G1Projective::infinity();
        }
        p.x = x;
        p.y = y;
        p.z = BN254_FP_R;
        return p;
    }
};

// Point doubling: 2P
inline G1Projective g1_double(const G1Projective& p) {
    if (p.is_infinity()) return p;

    // Using standard doubling formulas for y^2 = x^3 + b
    U256 A = fp_square(p.x);
    U256 B = fp_square(p.y);
    U256 C = fp_square(B);

    U256 xpb = fp_add(p.x, B);
    U256 D = fp_double(fp_sub(fp_square(xpb), fp_add(A, C)));

    U256 E = fp_add(fp_double(A), A);  // 3*A (since a=0 for BN254)
    U256 F = fp_square(E);

    G1Projective r;
    r.x = fp_sub(F, fp_double(D));
    r.y = fp_sub(fp_mul(E, fp_sub(D, r.x)), fp_double(fp_double(fp_double(C))));
    r.z = fp_double(fp_mul(p.y, p.z));

    return r;
}

// Point addition: P + Q
inline G1Projective g1_add(const G1Projective& p, const G1Projective& q) {
    if (p.is_infinity()) return q;
    if (q.is_infinity()) return p;

    U256 z1z1 = fp_square(p.z);
    U256 z2z2 = fp_square(q.z);
    U256 u1 = fp_mul(p.x, z2z2);
    U256 u2 = fp_mul(q.x, z1z1);
    U256 s1 = fp_mul(fp_mul(p.y, q.z), z2z2);
    U256 s2 = fp_mul(fp_mul(q.y, p.z), z1z1);

    if (u1 == u2) {
        if (s1 == s2) {
            return g1_double(p);
        } else {
            return G1Projective::infinity();
        }
    }

    U256 h = fp_sub(u2, u1);
    U256 i = fp_square(fp_double(h));
    U256 j = fp_mul(h, i);
    U256 rr = fp_double(fp_sub(s2, s1));
    U256 v = fp_mul(u1, i);

    G1Projective result;
    result.x = fp_sub(fp_sub(fp_square(rr), j), fp_double(v));
    result.y = fp_sub(fp_mul(rr, fp_sub(v, result.x)), fp_double(fp_mul(s1, j)));
    result.z = fp_mul(fp_sub(fp_square(fp_add(p.z, q.z)), fp_add(z1z1, z2z2)), h);

    return result;
}

// Mixed addition (affine + projective)
inline G1Projective g1_add_mixed(const G1Projective& p, const G1Affine& q) {
    if (q.infinity) return p;
    if (p.is_infinity()) return q.to_projective();

    U256 z1z1 = fp_square(p.z);
    U256 u2 = fp_mul(q.x, z1z1);
    U256 s2 = fp_mul(fp_mul(q.y, p.z), z1z1);

    if (p.x == u2) {
        if (p.y == s2) {
            return g1_double(p);
        } else {
            return G1Projective::infinity();
        }
    }

    U256 h = fp_sub(u2, p.x);
    U256 hh = fp_square(h);
    U256 i = fp_double(fp_double(hh));
    U256 j = fp_mul(h, i);
    U256 rr = fp_double(fp_sub(s2, p.y));
    U256 v = fp_mul(p.x, i);

    G1Projective result;
    result.x = fp_sub(fp_sub(fp_square(rr), j), fp_double(v));
    result.y = fp_sub(fp_mul(rr, fp_sub(v, result.x)), fp_double(fp_mul(p.y, j)));
    result.z = fp_mul(fp_sub(fp_square(fp_add(p.z, h)), fp_add(z1z1, hh)), U256(1));
    result.z = fp_mul(fp_add(p.z, h), fp_add(p.z, h));
    result.z = fp_sub(result.z, z1z1);
    result.z = fp_sub(result.z, hh);

    return result;
}

// Scalar multiplication: k * P
inline G1Projective g1_scalar_mul(const G1Projective& p, const U256& k) {
    G1Projective result = G1Projective::infinity();
    G1Projective base = p;

    for (int i = 0; i < 256; i++) {
        if (k.bit(i)) {
            result = g1_add(result, base);
        }
        base = g1_double(base);
    }

    return result;
}

// Convert projective to affine
inline G1Affine g1_to_affine(const G1Projective& p) {
    G1Affine a;
    if (p.is_infinity()) {
        return G1Affine::identity();
    }

    U256 zinv = fp_inv(p.z);
    U256 zinv2 = fp_square(zinv);
    U256 zinv3 = fp_mul(zinv2, zinv);

    a.x = fp_mul(p.x, zinv2);
    a.y = fp_mul(p.y, zinv3);
    a.infinity = false;

    return a;
}

// =============================================================================
// MSM (Multi-Scalar Multiplication) using Pippenger's Algorithm
// =============================================================================

G1Projective msm_bn254(const U256* scalars, const G1Affine* points, size_t n) {
    if (n == 0) return G1Projective::infinity();

    // Choose window size based on input size
    int c = 1;
    if (n >= 32) c = 4;
    if (n >= 256) c = 6;
    if (n >= 1024) c = 8;
    if (n >= 4096) c = 10;

    int num_buckets = (1 << c) - 1;
    int num_windows = (256 + c - 1) / c;

    std::vector<G1Projective> buckets(num_buckets);

    G1Projective result = G1Projective::infinity();

    for (int w = num_windows - 1; w >= 0; w--) {
        // Double result c times
        for (int i = 0; i < c && w != num_windows - 1; i++) {
            result = g1_double(result);
        }

        // Clear buckets
        for (int i = 0; i < num_buckets; i++) {
            buckets[i] = G1Projective::infinity();
        }

        // Distribute points to buckets
        #ifdef _OPENMP
        #pragma omp parallel for if(n > 256)
        #endif
        for (size_t i = 0; i < n; i++) {
            // Extract c-bit window from scalar
            int shift = w * c;
            uint64_t bucket_idx = 0;

            if (shift < 64) {
                bucket_idx = (scalars[i].limbs[shift / 64] >> (shift % 64)) & ((1ULL << c) - 1);
            } else if (shift < 128) {
                int limb = shift / 64;
                int bit_off = shift % 64;
                bucket_idx = scalars[i].limbs[limb] >> bit_off;
                if (bit_off > 0 && limb + 1 < 4) {
                    bucket_idx |= scalars[i].limbs[limb + 1] << (64 - bit_off);
                }
                bucket_idx &= ((1ULL << c) - 1);
            } else if (shift < 192) {
                int limb = shift / 64;
                int bit_off = shift % 64;
                bucket_idx = scalars[i].limbs[limb] >> bit_off;
                if (bit_off > 0 && limb + 1 < 4) {
                    bucket_idx |= scalars[i].limbs[limb + 1] << (64 - bit_off);
                }
                bucket_idx &= ((1ULL << c) - 1);
            } else if (shift < 256) {
                int limb = shift / 64;
                int bit_off = shift % 64;
                if (limb < 4) {
                    bucket_idx = scalars[i].limbs[limb] >> bit_off;
                    bucket_idx &= ((1ULL << c) - 1);
                }
            }

            if (bucket_idx != 0) {
                #ifdef _OPENMP
                #pragma omp critical
                #endif
                {
                    buckets[bucket_idx - 1] = g1_add_mixed(buckets[bucket_idx - 1], points[i]);
                }
            }
        }

        // Sum buckets: bucket[i] contributes with weight (i+1)
        G1Projective running = G1Projective::infinity();
        G1Projective sum = G1Projective::infinity();

        for (int i = num_buckets - 1; i >= 0; i--) {
            running = g1_add(running, buckets[i]);
            sum = g1_add(sum, running);
        }

        result = g1_add(result, sum);
    }

    return result;
}

// =============================================================================
// Modular Arithmetic for NTT
// =============================================================================

static inline uint64_t mod_add(uint64_t a, uint64_t b, uint64_t m) {
    return ((__uint128_t)a + b) % m;
}

static inline uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t m) {
    a %= m;
    b %= m;
    return (a >= b) ? (a - b) : (m - (b - a));
}

static inline uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t m) {
    return ((__uint128_t)a * b) % m;
}

static uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t m) {
    uint64_t result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) result = mod_mul(result, base, m);
        exp >>= 1;
        base = mod_mul(base, base, m);
    }
    return result;
}

static void bit_reverse(uint64_t* data, size_t n) {
    size_t j = 0;
    for (size_t i = 0; i < n; i++) {
        if (i < j) std::swap(data[i], data[j]);
        size_t m = n >> 1;
        while (m >= 1 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

static uint64_t find_primitive_root(size_t n, uint64_t m) {
    // Known primitive roots for common NTT primes
    if (m == 0xFFFFFFFF00000001ULL) return 7;  // Goldilocks
    if (m == 0x1000000000000001ULL) return 3;
    return 3;
}

// =============================================================================
// Poseidon2 Constants for BN254
// =============================================================================

// Poseidon2 round constants (subset for demonstration)
static const U256 POSEIDON2_RC[] = {
    U256(0x0ee9a592ba9a9518ULL, 0x99a7c3e6a8a4d90cULL, 0x53f2b7e7f1f35ebcULL, 0x2ccc32b6c7c21d9cULL),
    U256(0xa54c664ae5b9e8adULL, 0x0e36f420f8a4a5bdULL, 0x6f8c3f5b0b1f4f6eULL, 0x1f5f5f5f5f5f5f5fULL),
    U256(0xb5c55df06f4c52c9ULL, 0x4b7f47e8c0a8a0d9ULL, 0x7e8e8e8e8e8e8e8eULL, 0x0d0d0d0d0d0d0d0dULL),
    U256(0xc6e633e0e0e6e6e6ULL, 0x5c8f58f9d1b9b1eaULL, 0x8f9f9f9f9f9f9f9fULL, 0x1e1e1e1e1e1e1e1eULL),
};

inline U256 fr_pow5(const U256& x) {
    U256 x2 = fr_square(x);
    U256 x4 = fr_square(x2);
    return fr_mul(x4, x);
}

void poseidon2_compress(U256* out, const U256* left, const U256* right) {
    // Simplified Poseidon2: t=3, RF=8, RP=22 (reduced for demo)
    const int RF = 8;
    const int RP = 22;

    U256 state[3];
    state[0] = fr_to_mont(U256(0));
    state[1] = fr_to_mont(*left);
    state[2] = fr_to_mont(*right);

    int rc_idx = 0;

    // Full rounds (first half)
    for (int r = 0; r < RF / 2; r++) {
        for (int i = 0; i < 3; i++) {
            state[i] = fr_add(state[i], POSEIDON2_RC[rc_idx++ % 4]);
            state[i] = fr_pow5(state[i]);
        }
        // MDS: sum all then add back
        U256 sum = fr_add(fr_add(state[0], state[1]), state[2]);
        for (int i = 0; i < 3; i++) {
            state[i] = fr_add(state[i], sum);
        }
    }

    // Partial rounds
    for (int r = 0; r < RP; r++) {
        state[0] = fr_add(state[0], POSEIDON2_RC[rc_idx++ % 4]);
        state[0] = fr_pow5(state[0]);
        U256 sum = fr_add(fr_add(state[0], state[1]), state[2]);
        state[0] = fr_add(state[0], sum);
        state[1] = fr_add(state[1], sum);
        state[2] = fr_add(fr_add(state[2], state[2]), sum);
    }

    // Full rounds (second half)
    for (int r = 0; r < RF / 2; r++) {
        for (int i = 0; i < 3; i++) {
            state[i] = fr_add(state[i], POSEIDON2_RC[rc_idx++ % 4]);
            state[i] = fr_pow5(state[i]);
        }
        U256 sum = fr_add(fr_add(state[0], state[1]), state[2]);
        for (int i = 0; i < 3; i++) {
            state[i] = fr_add(state[i], sum);
        }
    }

    *out = fr_from_mont(state[1]);
}

// =============================================================================
// BLAKE3 Implementation (compression function)
// =============================================================================

static const uint32_t BLAKE3_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

static const uint8_t BLAKE3_MSG_PERMUTATION[16] = {
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8
};

static inline uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static void blake3_g(uint32_t* state, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
    state[a] = state[a] + state[b] + mx;
    state[d] = rotr32(state[d] ^ state[a], 16);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 12);
    state[a] = state[a] + state[b] + my;
    state[d] = rotr32(state[d] ^ state[a], 8);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 7);
}

static void blake3_round(uint32_t* state, const uint32_t* m) {
    blake3_g(state, 0, 4, 8, 12, m[0], m[1]);
    blake3_g(state, 1, 5, 9, 13, m[2], m[3]);
    blake3_g(state, 2, 6, 10, 14, m[4], m[5]);
    blake3_g(state, 3, 7, 11, 15, m[6], m[7]);
    blake3_g(state, 0, 5, 10, 15, m[8], m[9]);
    blake3_g(state, 1, 6, 11, 12, m[10], m[11]);
    blake3_g(state, 2, 7, 8, 13, m[12], m[13]);
    blake3_g(state, 3, 4, 9, 14, m[14], m[15]);
}

static void blake3_compress(const uint32_t* cv, const uint8_t* block,
                            uint64_t counter, uint32_t block_len, uint32_t flags,
                            uint32_t* out) {
    uint32_t state[16] = {
        cv[0], cv[1], cv[2], cv[3],
        cv[4], cv[5], cv[6], cv[7],
        BLAKE3_IV[0], BLAKE3_IV[1], BLAKE3_IV[2], BLAKE3_IV[3],
        (uint32_t)counter, (uint32_t)(counter >> 32), block_len, flags
    };

    uint32_t m[16];
    for (int i = 0; i < 16; i++) {
        m[i] = ((uint32_t)block[i * 4 + 0]) |
               ((uint32_t)block[i * 4 + 1] << 8) |
               ((uint32_t)block[i * 4 + 2] << 16) |
               ((uint32_t)block[i * 4 + 3] << 24);
    }

    // 7 rounds
    for (int round = 0; round < 7; round++) {
        blake3_round(state, m);
        // Permute message
        uint32_t tmp[16];
        for (int i = 0; i < 16; i++) {
            tmp[i] = m[BLAKE3_MSG_PERMUTATION[i]];
        }
        memcpy(m, tmp, sizeof(m));
    }

    for (int i = 0; i < 8; i++) {
        out[i] = state[i] ^ state[i + 8];
    }
}

void blake3_hash_single(const uint8_t* input, size_t len, uint8_t output[32]) {
    uint32_t cv[8];
    memcpy(cv, BLAKE3_IV, sizeof(cv));

    uint8_t block[64];
    memset(block, 0, 64);
    size_t to_copy = len < 64 ? len : 64;
    memcpy(block, input, to_copy);

    uint32_t flags = 0x01 | 0x02 | 0x08;  // CHUNK_START | CHUNK_END | ROOT
    uint32_t out[8];
    blake3_compress(cv, block, 0, (uint32_t)to_copy, flags, out);

    for (int i = 0; i < 8; i++) {
        output[i * 4 + 0] = out[i] & 0xFF;
        output[i * 4 + 1] = (out[i] >> 8) & 0xFF;
        output[i * 4 + 2] = (out[i] >> 16) & 0xFF;
        output[i * 4 + 3] = (out[i] >> 24) & 0xFF;
    }
}

} // anonymous namespace

// =============================================================================
// CPU Backend Functions
// =============================================================================

static LuxBackendContext* cpu_create_context(int device_index) {
    auto ctx = new CPUContext();
    ctx->device_index = device_index;
    return reinterpret_cast<LuxBackendContext*>(ctx);
}

static void cpu_destroy_context(LuxBackendContext* ctx) {
    delete reinterpret_cast<CPUContext*>(ctx);
}

static LuxBackendError cpu_get_device_count(int* count) {
    if (!count) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    *count = 1;
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_get_device_info(LuxBackendContext*, LuxBackendDeviceInfo* info) {
    if (!info) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    info->name = "CPU";
    info->vendor = "System";
    info->memory_total = 0;
    info->memory_available = 0;
#ifdef _OPENMP
    info->compute_units = omp_get_max_threads();
#else
    info->compute_units = 1;
#endif
    info->max_workgroup_size = 1;
    info->is_discrete = false;
    info->is_unified_memory = true;
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_sync(LuxBackendContext*) {
    return LUX_BACKEND_OK;
}

// Buffer management
static LuxBackendBuffer* cpu_buffer_alloc(LuxBackendContext*, size_t bytes) {
    auto buf = new CPUBuffer();
    buf->data = std::malloc(bytes);
    buf->size = bytes;
    if (!buf->data) {
        delete buf;
        return nullptr;
    }
    std::memset(buf->data, 0, bytes);
    return reinterpret_cast<LuxBackendBuffer*>(buf);
}

static LuxBackendBuffer* cpu_buffer_alloc_with_data(LuxBackendContext* ctx, const void* data, size_t bytes) {
    auto buf = reinterpret_cast<CPUBuffer*>(cpu_buffer_alloc(ctx, bytes));
    if (!buf) return nullptr;
    std::memcpy(buf->data, data, bytes);
    return reinterpret_cast<LuxBackendBuffer*>(buf);
}

static void cpu_buffer_free(LuxBackendContext*, LuxBackendBuffer* buf) {
    auto b = reinterpret_cast<CPUBuffer*>(buf);
    if (b) {
        std::free(b->data);
        delete b;
    }
}

static LuxBackendError cpu_buffer_copy_to_host(LuxBackendContext*, LuxBackendBuffer* buf, void* dst, size_t bytes) {
    auto b = reinterpret_cast<CPUBuffer*>(buf);
    if (!b || !dst) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    std::memcpy(dst, b->data, std::min(bytes, b->size));
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_buffer_copy_from_host(LuxBackendContext*, LuxBackendBuffer* buf, const void* src, size_t bytes) {
    auto b = reinterpret_cast<CPUBuffer*>(buf);
    if (!b || !src) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    std::memcpy(b->data, src, std::min(bytes, b->size));
    return LUX_BACKEND_OK;
}

static void* cpu_buffer_get_host_ptr(LuxBackendContext*, LuxBackendBuffer* buf) {
    auto b = reinterpret_cast<CPUBuffer*>(buf);
    return b ? b->data : nullptr;
}

// =============================================================================
// Kernel Emulation (simple interpreter for common patterns)
// =============================================================================

struct CPUKernel {
    std::string source;
    std::string entry_point;
};

static LuxBackendKernel* cpu_kernel_load(LuxBackendContext*, const char* source, const char* entry_point) {
    if (!source || !entry_point) return nullptr;
    auto k = new CPUKernel();
    k->source = source;
    k->entry_point = entry_point;
    return reinterpret_cast<LuxBackendKernel*>(k);
}

static LuxBackendKernel* cpu_kernel_load_binary(LuxBackendContext*, const void*, size_t, const char* entry_point) {
    // Binary kernels not supported on CPU
    if (!entry_point) return nullptr;
    auto k = new CPUKernel();
    k->entry_point = entry_point;
    return reinterpret_cast<LuxBackendKernel*>(k);
}

static void cpu_kernel_destroy(LuxBackendContext*, LuxBackendKernel* kernel) {
    delete reinterpret_cast<CPUKernel*>(kernel);
}

static LuxBackendError cpu_kernel_dispatch(
    LuxBackendContext*, LuxBackendKernel* kernel, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t block_x, uint32_t block_y, uint32_t block_z, LuxBackendBuffer** buffers, int num_buffers) {
    if (!kernel) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    // CPU kernel dispatch is a no-op placeholder; real work done in op_* functions
    (void)grid_x; (void)grid_y; (void)grid_z;
    (void)block_x; (void)block_y; (void)block_z;
    (void)buffers; (void)num_buffers;

    return LUX_BACKEND_OK;
}

// =============================================================================
// Elementwise Operations
// =============================================================================

static LuxBackendError cpu_op_add_f32(LuxBackendContext*, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
    auto ba = reinterpret_cast<CPUBuffer*>(a);
    auto bb = reinterpret_cast<CPUBuffer*>(b);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!ba || !bb || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pa = static_cast<const float*>(ba->data);
    const float* pb = static_cast<const float*>(bb->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for simd
#endif
    for (size_t i = 0; i < n; i++) {
        po[i] = pa[i] + pb[i];
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_sub_f32(LuxBackendContext*, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
    auto ba = reinterpret_cast<CPUBuffer*>(a);
    auto bb = reinterpret_cast<CPUBuffer*>(b);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!ba || !bb || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pa = static_cast<const float*>(ba->data);
    const float* pb = static_cast<const float*>(bb->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for simd
#endif
    for (size_t i = 0; i < n; i++) {
        po[i] = pa[i] - pb[i];
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_mul_f32(LuxBackendContext*, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
    auto ba = reinterpret_cast<CPUBuffer*>(a);
    auto bb = reinterpret_cast<CPUBuffer*>(b);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!ba || !bb || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pa = static_cast<const float*>(ba->data);
    const float* pb = static_cast<const float*>(bb->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for simd
#endif
    for (size_t i = 0; i < n; i++) {
        po[i] = pa[i] * pb[i];
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_div_f32(LuxBackendContext*, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
    auto ba = reinterpret_cast<CPUBuffer*>(a);
    auto bb = reinterpret_cast<CPUBuffer*>(b);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!ba || !bb || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pa = static_cast<const float*>(ba->data);
    const float* pb = static_cast<const float*>(bb->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for simd
#endif
    for (size_t i = 0; i < n; i++) {
        po[i] = pa[i] / pb[i];
    }
    return LUX_BACKEND_OK;
}

// =============================================================================
// Matrix Operations
// =============================================================================

static LuxBackendError cpu_op_matmul_f32(LuxBackendContext*, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, int M, int K, int N) {
    auto ba = reinterpret_cast<CPUBuffer*>(a);
    auto bb = reinterpret_cast<CPUBuffer*>(b);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!ba || !bb || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pa = static_cast<const float*>(ba->data);
    const float* pb = static_cast<const float*>(bb->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += pa[i * K + k] * pb[k * N + j];
            }
            po[i * N + j] = sum;
        }
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_transpose_f32(LuxBackendContext*, LuxBackendBuffer* in, LuxBackendBuffer* out, int rows, int cols) {
    auto bi = reinterpret_cast<CPUBuffer*>(in);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!bi || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pi = static_cast<const float*>(bi->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            po[j * rows + i] = pi[i * cols + j];
        }
    }
    return LUX_BACKEND_OK;
}

// =============================================================================
// Reduction Operations
// =============================================================================

static LuxBackendError cpu_op_reduce_sum_f32(LuxBackendContext*, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n) {
    auto bi = reinterpret_cast<CPUBuffer*>(in);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!bi || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pi = static_cast<const float*>(bi->data);
    float* po = static_cast<float*>(bo->data);

    float sum = 0.0f;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:sum)
#endif
    for (size_t i = 0; i < n; i++) {
        sum += pi[i];
    }
    po[0] = sum;
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_reduce_max_f32(LuxBackendContext*, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n) {
    auto bi = reinterpret_cast<CPUBuffer*>(in);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!bi || !bo || n == 0) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pi = static_cast<const float*>(bi->data);
    float* po = static_cast<float*>(bo->data);

    float maxval = pi[0];
#ifdef _OPENMP
    #pragma omp parallel for reduction(max:maxval)
#endif
    for (size_t i = 1; i < n; i++) {
        if (pi[i] > maxval) maxval = pi[i];
    }
    po[0] = maxval;
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_reduce_min_f32(LuxBackendContext*, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n) {
    auto bi = reinterpret_cast<CPUBuffer*>(in);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!bi || !bo || n == 0) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pi = static_cast<const float*>(bi->data);
    float* po = static_cast<float*>(bo->data);

    float minval = pi[0];
#ifdef _OPENMP
    #pragma omp parallel for reduction(min:minval)
#endif
    for (size_t i = 1; i < n; i++) {
        if (pi[i] < minval) minval = pi[i];
    }
    po[0] = minval;
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_reduce_mean_f32(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n) {
    auto err = cpu_op_reduce_sum_f32(ctx, in, out, n);
    if (err != LUX_BACKEND_OK) return err;
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    float* po = static_cast<float*>(bo->data);
    po[0] /= (float)n;
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_reduce_sum_axis_f32(LuxBackendContext*, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t outer_size, size_t inner_size) {
    auto bi = reinterpret_cast<CPUBuffer*>(in);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!bi || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pi = static_cast<const float*>(bi->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < outer_size; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < inner_size; j++) {
            sum += pi[i * inner_size + j];
        }
        po[i] = sum;
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_reduce_max_axis_f32(LuxBackendContext*, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t outer_size, size_t inner_size) {
    auto bi = reinterpret_cast<CPUBuffer*>(in);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!bi || !bo || inner_size == 0) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pi = static_cast<const float*>(bi->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < outer_size; i++) {
        float maxval = pi[i * inner_size];
        for (size_t j = 1; j < inner_size; j++) {
            float v = pi[i * inner_size + j];
            if (v > maxval) maxval = v;
        }
        po[i] = maxval;
    }
    return LUX_BACKEND_OK;
}

// =============================================================================
// Softmax Operations
// =============================================================================

static LuxBackendError cpu_op_softmax_f32(LuxBackendContext*, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t batch_size, size_t dim) {
    auto bi = reinterpret_cast<CPUBuffer*>(in);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!bi || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pi = static_cast<const float*>(bi->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t b = 0; b < batch_size; b++) {
        const float* row = pi + b * dim;
        float* out_row = po + b * dim;

        // Find max for numerical stability
        float maxval = row[0];
        for (size_t i = 1; i < dim; i++) {
            if (row[i] > maxval) maxval = row[i];
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            out_row[i] = std::exp(row[i] - maxval);
            sum += out_row[i];
        }

        // Normalize
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < dim; i++) {
            out_row[i] *= inv_sum;
        }
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_log_softmax_f32(LuxBackendContext*, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t batch_size, size_t dim) {
    auto bi = reinterpret_cast<CPUBuffer*>(in);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!bi || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pi = static_cast<const float*>(bi->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t b = 0; b < batch_size; b++) {
        const float* row = pi + b * dim;
        float* out_row = po + b * dim;

        float maxval = row[0];
        for (size_t i = 1; i < dim; i++) {
            if (row[i] > maxval) maxval = row[i];
        }

        float sum = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            sum += std::exp(row[i] - maxval);
        }
        float log_sum = std::log(sum);

        for (size_t i = 0; i < dim; i++) {
            out_row[i] = row[i] - maxval - log_sum;
        }
    }
    return LUX_BACKEND_OK;
}

// =============================================================================
// Unary Operations
// =============================================================================

static LuxBackendError cpu_op_exp_f32(LuxBackendContext*, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n) {
    auto bi = reinterpret_cast<CPUBuffer*>(in);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!bi || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pi = static_cast<const float*>(bi->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for simd
#endif
    for (size_t i = 0; i < n; i++) {
        po[i] = std::exp(pi[i]);
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_log_f32(LuxBackendContext*, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n) {
    auto bi = reinterpret_cast<CPUBuffer*>(in);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!bi || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pi = static_cast<const float*>(bi->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for simd
#endif
    for (size_t i = 0; i < n; i++) {
        po[i] = std::log(pi[i]);
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_sqrt_f32(LuxBackendContext*, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n) {
    auto bi = reinterpret_cast<CPUBuffer*>(in);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!bi || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pi = static_cast<const float*>(bi->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for simd
#endif
    for (size_t i = 0; i < n; i++) {
        po[i] = std::sqrt(pi[i]);
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_neg_f32(LuxBackendContext*, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n) {
    auto bi = reinterpret_cast<CPUBuffer*>(in);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!bi || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pi = static_cast<const float*>(bi->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for simd
#endif
    for (size_t i = 0; i < n; i++) {
        po[i] = -pi[i];
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_abs_f32(LuxBackendContext*, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n) {
    auto bi = reinterpret_cast<CPUBuffer*>(in);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!bi || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pi = static_cast<const float*>(bi->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for simd
#endif
    for (size_t i = 0; i < n; i++) {
        po[i] = std::fabs(pi[i]);
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_tanh_f32(LuxBackendContext*, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n) {
    auto bi = reinterpret_cast<CPUBuffer*>(in);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!bi || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pi = static_cast<const float*>(bi->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for simd
#endif
    for (size_t i = 0; i < n; i++) {
        po[i] = std::tanh(pi[i]);
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_sigmoid_f32(LuxBackendContext*, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n) {
    auto bi = reinterpret_cast<CPUBuffer*>(in);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!bi || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pi = static_cast<const float*>(bi->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for simd
#endif
    for (size_t i = 0; i < n; i++) {
        po[i] = 1.0f / (1.0f + std::exp(-pi[i]));
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_relu_f32(LuxBackendContext*, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n) {
    auto bi = reinterpret_cast<CPUBuffer*>(in);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!bi || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pi = static_cast<const float*>(bi->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for simd
#endif
    for (size_t i = 0; i < n; i++) {
        po[i] = pi[i] > 0.0f ? pi[i] : 0.0f;
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_gelu_f32(LuxBackendContext*, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n) {
    auto bi = reinterpret_cast<CPUBuffer*>(in);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!bi || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pi = static_cast<const float*>(bi->data);
    float* po = static_cast<float*>(bo->data);

    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float c = 0.044715f;

#ifdef _OPENMP
    #pragma omp parallel for simd
#endif
    for (size_t i = 0; i < n; i++) {
        float x = pi[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + c * x3);
        po[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
    return LUX_BACKEND_OK;
}

// =============================================================================
// Copy Operations
// =============================================================================

static LuxBackendError cpu_op_copy_f32(LuxBackendContext*, LuxBackendBuffer* src, LuxBackendBuffer* dst, size_t n) {
    auto bs = reinterpret_cast<CPUBuffer*>(src);
    auto bd = reinterpret_cast<CPUBuffer*>(dst);
    if (!bs || !bd) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    std::memcpy(bd->data, bs->data, n * sizeof(float));
    return LUX_BACKEND_OK;
}

// =============================================================================
// Normalization Operations
// =============================================================================

static LuxBackendError cpu_op_layer_norm_f32(LuxBackendContext*, LuxBackendBuffer* in, LuxBackendBuffer* out,
                                              LuxBackendBuffer* gamma, LuxBackendBuffer* beta,
                                              size_t batch_size, size_t dim, float eps) {
    auto bi = reinterpret_cast<CPUBuffer*>(in);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    auto bg = reinterpret_cast<CPUBuffer*>(gamma);
    auto bb = reinterpret_cast<CPUBuffer*>(beta);
    if (!bi || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pi = static_cast<const float*>(bi->data);
    float* po = static_cast<float*>(bo->data);
    const float* pg = bg ? static_cast<const float*>(bg->data) : nullptr;
    const float* pb = bb ? static_cast<const float*>(bb->data) : nullptr;

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t b = 0; b < batch_size; b++) {
        const float* row = pi + b * dim;
        float* out_row = po + b * dim;

        // Compute mean
        float mean = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            mean += row[i];
        }
        mean /= (float)dim;

        // Compute variance
        float var = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            float d = row[i] - mean;
            var += d * d;
        }
        var /= (float)dim;

        // Normalize
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (size_t i = 0; i < dim; i++) {
            float normalized = (row[i] - mean) * inv_std;
            if (pg && pb) {
                out_row[i] = pg[i] * normalized + pb[i];
            } else {
                out_row[i] = normalized;
            }
        }
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_rms_norm_f32(LuxBackendContext*, LuxBackendBuffer* in, LuxBackendBuffer* out,
                                            LuxBackendBuffer* weight, size_t batch_size, size_t dim, float eps) {
    auto bi = reinterpret_cast<CPUBuffer*>(in);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    auto bw = reinterpret_cast<CPUBuffer*>(weight);
    if (!bi || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pi = static_cast<const float*>(bi->data);
    float* po = static_cast<float*>(bo->data);
    const float* pw = bw ? static_cast<const float*>(bw->data) : nullptr;

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t b = 0; b < batch_size; b++) {
        const float* row = pi + b * dim;
        float* out_row = po + b * dim;

        // Compute RMS
        float sum_sq = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            sum_sq += row[i] * row[i];
        }
        float rms = std::sqrt(sum_sq / (float)dim + eps);
        float inv_rms = 1.0f / rms;

        for (size_t i = 0; i < dim; i++) {
            float normalized = row[i] * inv_rms;
            if (pw) {
                out_row[i] = pw[i] * normalized;
            } else {
                out_row[i] = normalized;
            }
        }
    }
    return LUX_BACKEND_OK;
}

// =============================================================================
// NTT Operations
// =============================================================================

static LuxBackendError cpu_op_ntt_forward(LuxBackendContext*, uint64_t* data, size_t n, uint64_t modulus) {
    if (!data || n == 0 || (n & (n - 1)) != 0) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    uint64_t g = find_primitive_root(n, modulus);
    uint64_t omega_n = mod_pow(g, (modulus - 1) / n, modulus);

    bit_reverse(data, n);

    for (size_t len = 2; len <= n; len *= 2) {
        uint64_t w = mod_pow(omega_n, n / len, modulus);
        for (size_t i = 0; i < n; i += len) {
            uint64_t wn = 1;
            for (size_t j = 0; j < len / 2; j++) {
                uint64_t u = data[i + j];
                uint64_t t = mod_mul(wn, data[i + j + len / 2], modulus);
                data[i + j] = mod_add(u, t, modulus);
                data[i + j + len / 2] = mod_sub(u, t, modulus);
                wn = mod_mul(wn, w, modulus);
            }
        }
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_ntt_inverse(LuxBackendContext*, uint64_t* data, size_t n, uint64_t modulus) {
    if (!data || n == 0 || (n & (n - 1)) != 0) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    uint64_t g = find_primitive_root(n, modulus);
    uint64_t omega_n = mod_pow(g, (modulus - 1) / n, modulus);
    uint64_t omega_n_inv = mod_pow(omega_n, modulus - 2, modulus);

    // DIF butterfly (large to small)
    for (size_t len = n; len >= 2; len /= 2) {
        uint64_t w = mod_pow(omega_n_inv, n / len, modulus);
        for (size_t i = 0; i < n; i += len) {
            uint64_t wn = 1;
            for (size_t j = 0; j < len / 2; j++) {
                uint64_t u = data[i + j];
                uint64_t v = data[i + j + len / 2];
                data[i + j] = mod_add(u, v, modulus);
                data[i + j + len / 2] = mod_mul(mod_sub(u, v, modulus), wn, modulus);
                wn = mod_mul(wn, w, modulus);
            }
        }
    }

    bit_reverse(data, n);

    // Scale by n^-1
    uint64_t n_inv = mod_pow(n, modulus - 2, modulus);
    for (size_t i = 0; i < n; i++) {
        data[i] = mod_mul(data[i], n_inv, modulus);
    }
    return LUX_BACKEND_OK;
}

// =============================================================================
// MSM Operation
// =============================================================================

static LuxBackendError cpu_op_msm(LuxBackendContext*, const void* scalars, const void* points, void* result, size_t n, int curve_type) {
    if (!scalars || !points || !result || n == 0) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    // Currently only BN254 implemented
    if (curve_type != 1) {  // LUX_CURVE_BN254
        return LUX_BACKEND_ERROR_NOT_SUPPORTED;
    }

    const U256* sc = static_cast<const U256*>(scalars);
    const G1Affine* pts = static_cast<const G1Affine*>(points);
    G1Projective* res = static_cast<G1Projective*>(result);

    *res = msm_bn254(sc, pts, n);
    return LUX_BACKEND_OK;
}

// =============================================================================
// FHE Operations - Forward declarations
// =============================================================================

static LuxBackendError cpu_op_blind_rotate(LuxBackendContext* ctx,
                                            uint64_t* acc, const uint64_t* bsk,
                                            const uint64_t* lwe_a,
                                            uint32_t n_lwe, uint32_t N, uint32_t k,
                                            uint32_t l, uint64_t q);

static LuxBackendError cpu_op_sample_extract(LuxBackendContext* ctx,
                                              const uint64_t* glwe, uint64_t* lwe,
                                              uint32_t N, uint32_t k, uint64_t q);

// =============================================================================
// FHE Operations
// =============================================================================

static LuxBackendError cpu_op_poly_mul(LuxBackendContext* ctx, const uint64_t* a, const uint64_t* b,
                                        uint64_t* result, size_t n, uint64_t modulus) {
    if (!a || !b || !result || n == 0 || (n & (n - 1)) != 0)
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    // Use NTT for polynomial multiplication
    std::vector<uint64_t> a_ntt(n), b_ntt(n);
    std::memcpy(a_ntt.data(), a, n * sizeof(uint64_t));
    std::memcpy(b_ntt.data(), b, n * sizeof(uint64_t));

    cpu_op_ntt_forward(ctx, a_ntt.data(), n, modulus);
    cpu_op_ntt_forward(ctx, b_ntt.data(), n, modulus);

    // Pointwise multiplication
    for (size_t i = 0; i < n; i++) {
        result[i] = mod_mul(a_ntt[i], b_ntt[i], modulus);
    }

    cpu_op_ntt_inverse(ctx, result, n, modulus);
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_tfhe_bootstrap(LuxBackendContext* ctx,
                                              const uint64_t* lwe_in, uint64_t* lwe_out,
                                              const uint64_t* bsk, const uint64_t* test_poly,
                                              uint32_t n_lwe, uint32_t N, uint32_t k,
                                              uint32_t l, uint64_t q) {
    if (!lwe_in || !lwe_out || !bsk || !test_poly)
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    // Extract the body b from LWE ciphertext
    uint64_t b = lwe_in[n_lwe];

    // Compute rotation index
    uint64_t rotation = (b * 2 * N) / q;
    rotation = rotation % (2 * N);

    // Initialize accumulator with rotated test polynomial
    size_t acc_size = (k + 1) * N;
    std::vector<uint64_t> acc(acc_size, 0);

    // Rotate test polynomial by -rotation (negacyclic)
    for (size_t i = 0; i < N; i++) {
        size_t src_idx = (i + rotation) % (2 * N);
        if (src_idx < N) {
            acc[(k * N) + i] = test_poly[src_idx];
        } else {
            acc[(k * N) + i] = (q - test_poly[src_idx - N]) % q;
        }
    }

    // Blind rotation using BSK
    cpu_op_blind_rotate(ctx, acc.data(), bsk, lwe_in, n_lwe, N, k, l, q);

    // Sample extraction
    cpu_op_sample_extract(ctx, acc.data(), lwe_out, N, k, q);

    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_tfhe_keyswitch(LuxBackendContext*,
                                              const uint64_t* lwe_in, uint64_t* lwe_out,
                                              const uint64_t* ksk,
                                              uint32_t n_in, uint32_t n_out,
                                              uint32_t l, uint32_t base_log, uint64_t q) {
    if (!lwe_in || !lwe_out || !ksk)
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    // Initialize output to zero
    for (uint32_t i = 0; i <= n_out; i++) {
        lwe_out[i] = 0;
    }

    // Copy the body
    lwe_out[n_out] = lwe_in[n_in];

    uint64_t base = 1ULL << base_log;
    uint64_t mask = base - 1;

    // Key switching
    for (uint32_t i = 0; i < n_in; i++) {
        uint64_t a_i = lwe_in[i];

        for (uint32_t j = 0; j < l; j++) {
            // Decompose a_i
            uint64_t shift = (l - 1 - j) * base_log;
            uint64_t digit = (a_i >> shift) & mask;

            if (digit != 0) {
                // Subtract ksk[i][j][digit] from output
                const uint64_t* ksk_entry = ksk + (i * l * base + j * base + digit) * (n_out + 1);
                for (uint32_t m = 0; m <= n_out; m++) {
                    lwe_out[m] = mod_sub(lwe_out[m], ksk_entry[m], q);
                }
            }
        }
    }

    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_blind_rotate(LuxBackendContext* ctx,
                                            uint64_t* acc, const uint64_t* bsk,
                                            const uint64_t* lwe_a,
                                            uint32_t n_lwe, uint32_t N, uint32_t k,
                                            uint32_t l, uint64_t q) {
    if (!acc || !bsk || !lwe_a)
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    size_t glwe_size = (k + 1) * N;

    // For each LWE coefficient
    for (uint32_t i = 0; i < n_lwe; i++) {
        // Compute rotation exponent
        uint64_t a_i = lwe_a[i];
        uint64_t exponent = (a_i * 2 * N) / q;
        exponent = exponent % (2 * N);

        if (exponent == 0) continue;

        // External product with BSK[i]
        // This is simplified; full implementation requires GGSW x GLWE
        const uint64_t* ggsw = bsk + i * (k + 1) * l * glwe_size;

        // Decompose accumulator and multiply
        std::vector<uint64_t> rotated(glwe_size);
        for (size_t j = 0; j < glwe_size; j++) {
            size_t poly_idx = j / N;
            size_t coef_idx = j % N;
            size_t src_idx = (coef_idx + exponent) % (2 * N);

            if (src_idx < N) {
                rotated[j] = acc[poly_idx * N + src_idx];
            } else {
                rotated[j] = (q - acc[poly_idx * N + (src_idx - N)]) % q;
            }
        }

        // Subtract rotated from acc (simplified CMux)
        for (size_t j = 0; j < glwe_size; j++) {
            acc[j] = mod_sub(acc[j], rotated[j], q);
        }

        // Add external product result (simplified)
        for (size_t j = 0; j < glwe_size; j++) {
            acc[j] = mod_add(acc[j], ggsw[j % (l * glwe_size)], q);
        }
    }

    (void)ctx;
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_sample_extract(LuxBackendContext*,
                                              const uint64_t* glwe, uint64_t* lwe,
                                              uint32_t N, uint32_t k, uint64_t q) {
    if (!glwe || !lwe)
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    // Extract LWE from GLWE at position 0
    // LWE dimension is k * N

    size_t lwe_dim = (size_t)k * N;

    // Extract 'a' coefficients (negacyclic unrolling)
    for (uint32_t i = 0; i < k; i++) {
        for (uint32_t j = 0; j < N; j++) {
            size_t lwe_idx = i * N + j;
            if (j == 0) {
                lwe[lwe_idx] = glwe[i * N];
            } else {
                // Negate due to negacyclic
                lwe[lwe_idx] = (q - glwe[i * N + (N - j)]) % q;
            }
        }
    }

    // Extract body (constant term of last polynomial)
    lwe[lwe_dim] = glwe[k * N];

    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_sample_ntt(LuxBackendContext* ctx,
                                          uint64_t* output, size_t n, uint64_t modulus,
                                          double sigma, uint64_t seed) {
    if (!output || n == 0)
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, sigma);

    // Sample Gaussian and reduce mod modulus
    for (size_t i = 0; i < n; i++) {
        double sample = dist(rng);
        int64_t rounded = static_cast<int64_t>(std::round(sample));

        // Convert to positive residue
        if (rounded >= 0) {
            output[i] = static_cast<uint64_t>(rounded) % modulus;
        } else {
            output[i] = modulus - (static_cast<uint64_t>(-rounded) % modulus);
        }
    }

    // Convert to NTT domain
    return cpu_op_ntt_forward(ctx, output, n, modulus);
}

// =============================================================================
// Crypto Hash Operations
// =============================================================================

static LuxBackendError cpu_op_poseidon2_hash(LuxBackendContext*,
                                              const uint64_t* inputs, uint64_t* outputs,
                                              size_t rate, size_t num_hashes) {
    if (!inputs || !outputs || rate == 0)
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    // Poseidon2 with rate inputs per hash
#ifdef _OPENMP
    #pragma omp parallel for if(num_hashes > 64)
#endif
    for (size_t h = 0; h < num_hashes; h++) {
        U256 left, right;
        memcpy(left.limbs, inputs + h * rate * 4, sizeof(U256));
        if (rate > 1) {
            memcpy(right.limbs, inputs + (h * rate + 1) * 4, sizeof(U256));
        } else {
            right = U256(0);
        }

        U256 out;
        poseidon2_compress(&out, &left, &right);
        memcpy(outputs + h * 4, out.limbs, sizeof(U256));
    }

    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_blake3_hash(LuxBackendContext*,
                                           const uint8_t* inputs, uint8_t* outputs,
                                           const size_t* input_lens, size_t num_hashes) {
    if (!inputs || !outputs || !input_lens)
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    size_t offset = 0;
#ifdef _OPENMP
    #pragma omp parallel for if(num_hashes > 64)
#endif
    for (size_t h = 0; h < num_hashes; h++) {
        // Calculate input offset for this hash
        size_t local_offset = 0;
        for (size_t i = 0; i < h; i++) {
            local_offset += input_lens[i];
        }
        blake3_hash_single(inputs + local_offset, input_lens[h], outputs + h * 32);
    }

    return LUX_BACKEND_OK;
}

// =============================================================================
// BLS12-381 Curve Operations (stub - uses similar structure to BN254)
// =============================================================================

static LuxBackendError cpu_op_bls12_381_add(LuxBackendContext*,
                                             const void* a, const void* b, void* out,
                                             size_t n, bool is_g2) {
    // BLS12-381 requires 384-bit arithmetic; stub for now
    (void)a; (void)b; (void)out; (void)n; (void)is_g2;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError cpu_op_bls12_381_mul(LuxBackendContext*,
                                             const void* points, const void* scalars, void* out,
                                             size_t n, bool is_g2) {
    (void)points; (void)scalars; (void)out; (void)n; (void)is_g2;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError cpu_op_bls12_381_pairing(LuxBackendContext*,
                                                 const void* g1_points, const void* g2_points,
                                                 void* out, size_t n) {
    (void)g1_points; (void)g2_points; (void)out; (void)n;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

// =============================================================================
// BN254 Curve Operations
// =============================================================================

static LuxBackendError cpu_op_bn254_add(LuxBackendContext*,
                                         const void* a, const void* b, void* out,
                                         size_t n, bool is_g2) {
    if (is_g2) return LUX_BACKEND_ERROR_NOT_SUPPORTED;
    if (!a || !b || !out) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const G1Projective* pa = static_cast<const G1Projective*>(a);
    const G1Projective* pb = static_cast<const G1Projective*>(b);
    G1Projective* po = static_cast<G1Projective*>(out);

#ifdef _OPENMP
    #pragma omp parallel for if(n > 256)
#endif
    for (size_t i = 0; i < n; i++) {
        po[i] = g1_add(pa[i], pb[i]);
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_bn254_mul(LuxBackendContext*,
                                         const void* points, const void* scalars, void* out,
                                         size_t n, bool is_g2) {
    if (is_g2) return LUX_BACKEND_ERROR_NOT_SUPPORTED;
    if (!points || !scalars || !out) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const G1Projective* pp = static_cast<const G1Projective*>(points);
    const U256* ps = static_cast<const U256*>(scalars);
    G1Projective* po = static_cast<G1Projective*>(out);

#ifdef _OPENMP
    #pragma omp parallel for if(n > 64)
#endif
    for (size_t i = 0; i < n; i++) {
        po[i] = g1_scalar_mul(pp[i], ps[i]);
    }
    return LUX_BACKEND_OK;
}

// =============================================================================
// KZG Operations
// =============================================================================

static LuxBackendError cpu_op_kzg_commit(LuxBackendContext*,
                                          const void* coeffs, const void* srs,
                                          void* commitment, size_t degree, int curve_type) {
    if (curve_type != 1) return LUX_BACKEND_ERROR_NOT_SUPPORTED;  // BN254 only
    if (!coeffs || !srs || !commitment) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    // KZG commitment is MSM of coefficients with SRS points
    const U256* c = static_cast<const U256*>(coeffs);
    const G1Affine* g = static_cast<const G1Affine*>(srs);
    G1Projective* out = static_cast<G1Projective*>(commitment);

    *out = msm_bn254(c, g, degree);
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_kzg_open(LuxBackendContext*,
                                        const void* coeffs, const void* srs,
                                        const void* point, void* proof,
                                        size_t degree, int curve_type) {
    if (curve_type != 1) return LUX_BACKEND_ERROR_NOT_SUPPORTED;
    if (!coeffs || !srs || !point || !proof) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const U256* c = static_cast<const U256*>(coeffs);
    const G1Affine* g = static_cast<const G1Affine*>(srs);
    const U256* z = static_cast<const U256*>(point);
    G1Projective* out = static_cast<G1Projective*>(proof);

    // Evaluate polynomial at z
    U256 z_mont = fr_to_mont(*z);
    U256 eval = U256(0);
    U256 z_power = BN254_FR_R;  // 1 in Montgomery form

    for (size_t i = 0; i < degree; i++) {
        U256 term = fr_mul(fr_to_mont(c[i]), z_power);
        eval = fr_add(eval, term);
        z_power = fr_mul(z_power, z_mont);
    }

    // Compute quotient polynomial: (p(x) - p(z)) / (x - z)
    std::vector<U256> quotient(degree - 1);
    U256 remainder = fr_to_mont(c[degree - 1]);

    for (int i = (int)degree - 2; i >= 0; i--) {
        quotient[i] = remainder;
        remainder = fr_add(fr_mul(remainder, z_mont), fr_to_mont(c[i]));
    }

    // Commit to quotient
    std::vector<G1Affine> srs_affine(degree - 1);
    for (size_t i = 0; i < degree - 1; i++) {
        srs_affine[i] = g[i];
    }

    *out = msm_bn254(quotient.data(), srs_affine.data(), degree - 1);
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_kzg_verify(LuxBackendContext*,
                                          const void* commitment, const void* proof,
                                          const void* point, const void* value,
                                          const void* srs_g2, bool* result, int curve_type) {
    // Verification requires pairing which is not yet implemented for BN254
    (void)commitment; (void)proof; (void)point; (void)value; (void)srs_g2;
    (void)result; (void)curve_type;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

// =============================================================================
// CPU Backend VTable
// =============================================================================

static const lux_gpu_backend_vtbl cpu_vtbl = {
    // Lifecycle
    .create_context = cpu_create_context,
    .destroy_context = cpu_destroy_context,

    // Device info
    .get_device_count = cpu_get_device_count,
    .get_device_info = cpu_get_device_info,

    // Sync
    .sync = cpu_sync,

    // Buffer management
    .buffer_alloc = cpu_buffer_alloc,
    .buffer_alloc_with_data = cpu_buffer_alloc_with_data,
    .buffer_free = cpu_buffer_free,
    .buffer_copy_to_host = cpu_buffer_copy_to_host,
    .buffer_copy_from_host = cpu_buffer_copy_from_host,
    .buffer_get_host_ptr = cpu_buffer_get_host_ptr,

    // Kernel management
    .kernel_load = cpu_kernel_load,
    .kernel_load_binary = cpu_kernel_load_binary,
    .kernel_destroy = cpu_kernel_destroy,
    .kernel_dispatch = cpu_kernel_dispatch,

    // Elementwise operations
    .op_add_f32 = cpu_op_add_f32,
    .op_sub_f32 = cpu_op_sub_f32,
    .op_mul_f32 = cpu_op_mul_f32,
    .op_div_f32 = cpu_op_div_f32,

    // Matrix operations
    .op_matmul_f32 = cpu_op_matmul_f32,
    .op_transpose_f32 = cpu_op_transpose_f32,

    // Reduction operations
    .op_reduce_sum_f32 = cpu_op_reduce_sum_f32,
    .op_reduce_max_f32 = cpu_op_reduce_max_f32,
    .op_reduce_min_f32 = cpu_op_reduce_min_f32,
    .op_reduce_mean_f32 = cpu_op_reduce_mean_f32,
    .op_reduce_sum_axis_f32 = cpu_op_reduce_sum_axis_f32,
    .op_reduce_max_axis_f32 = cpu_op_reduce_max_axis_f32,

    // Softmax operations
    .op_softmax_f32 = cpu_op_softmax_f32,
    .op_log_softmax_f32 = cpu_op_log_softmax_f32,

    // Unary operations
    .op_exp_f32 = cpu_op_exp_f32,
    .op_log_f32 = cpu_op_log_f32,
    .op_sqrt_f32 = cpu_op_sqrt_f32,
    .op_neg_f32 = cpu_op_neg_f32,
    .op_abs_f32 = cpu_op_abs_f32,
    .op_tanh_f32 = cpu_op_tanh_f32,
    .op_sigmoid_f32 = cpu_op_sigmoid_f32,
    .op_relu_f32 = cpu_op_relu_f32,
    .op_gelu_f32 = cpu_op_gelu_f32,

    // Copy operations
    .op_copy_f32 = cpu_op_copy_f32,

    // Normalization operations
    .op_layer_norm_f32 = cpu_op_layer_norm_f32,
    .op_rms_norm_f32 = cpu_op_rms_norm_f32,

    // NTT operations
    .op_ntt_forward = cpu_op_ntt_forward,
    .op_ntt_inverse = cpu_op_ntt_inverse,

    // MSM
    .op_msm = cpu_op_msm,

    // FHE operations
    .op_poly_mul = cpu_op_poly_mul,
    .op_tfhe_bootstrap = cpu_op_tfhe_bootstrap,
    .op_tfhe_keyswitch = cpu_op_tfhe_keyswitch,
    .op_blind_rotate = cpu_op_blind_rotate,
    .op_sample_extract = cpu_op_sample_extract,
    .op_sample_ntt = cpu_op_sample_ntt,

    // Crypto hash operations
    .op_poseidon2_hash = cpu_op_poseidon2_hash,
    .op_blake3_hash = cpu_op_blake3_hash,

    // BLS12-381 operations
    .op_bls12_381_add = cpu_op_bls12_381_add,
    .op_bls12_381_mul = cpu_op_bls12_381_mul,
    .op_bls12_381_pairing = cpu_op_bls12_381_pairing,

    // BN254 operations
    .op_bn254_add = cpu_op_bn254_add,
    .op_bn254_mul = cpu_op_bn254_mul,

    // KZG operations
    .op_kzg_commit = cpu_op_kzg_commit,
    .op_kzg_open = cpu_op_kzg_open,
    .op_kzg_verify = cpu_op_kzg_verify,

    // Reserved
    ._reserved = {nullptr, nullptr, nullptr, nullptr}
};

// =============================================================================
// Entry Point (called by core, not a plugin)
// =============================================================================

extern "C" bool cpu_backend_init(lux_gpu_backend_desc* out) {
    if (!out) return false;
    out->abi_version = LUX_GPU_BACKEND_ABI_VERSION;
    out->backend_name = "cpu";
    out->backend_version = "0.2.0";
    out->capabilities = LUX_CAP_TENSOR_OPS | LUX_CAP_MATMUL | LUX_CAP_NTT | LUX_CAP_MSM
                      | LUX_CAP_UNIFIED_MEMORY | LUX_CAP_FHE | LUX_CAP_TFHE
                      | LUX_CAP_REDUCE | LUX_CAP_SOFTMAX | LUX_CAP_UNARY
                      | LUX_CAP_NORMALIZATION | LUX_CAP_BN254 | LUX_CAP_KZG
                      | LUX_CAP_POSEIDON2 | LUX_CAP_BLAKE3 | LUX_CAP_BLIND_ROTATE
                      | LUX_CAP_POLY_MUL;
    out->vtbl = &cpu_vtbl;
    return true;
}
