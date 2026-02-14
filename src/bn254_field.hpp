// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// BN254 Field Arithmetic - Internal shared implementation
//
// Provides U256 (256-bit integers) and BN254 Fr/Fp field operations
// using Montgomery multiplication for efficiency.
//
// Single canonical implementation used by both cpu_backend.cpp and zk_ops.cpp

#ifndef LUX_BN254_FIELD_HPP
#define LUX_BN254_FIELD_HPP

#include <cstdint>
#include <cstring>

namespace lux {
namespace bn254 {

// =============================================================================
// U256: 256-bit Unsigned Integer (4 x 64-bit limbs, little-endian)
// =============================================================================

struct U256 {
    uint64_t limbs[4];

    U256() : limbs{0, 0, 0, 0} {}
    explicit U256(uint64_t v) : limbs{v, 0, 0, 0} {}
    U256(uint64_t l0, uint64_t l1, uint64_t l2, uint64_t l3)
        : limbs{l0, l1, l2, l3} {}

    bool is_zero() const {
        return limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0;
    }

    bool operator==(const U256& o) const {
        return limbs[0] == o.limbs[0] && limbs[1] == o.limbs[1] &&
               limbs[2] == o.limbs[2] && limbs[3] == o.limbs[3];
    }

    bool operator!=(const U256& o) const { return !(*this == o); }

    bool operator>=(const U256& o) const {
        for (int i = 3; i >= 0; i--) {
            if (limbs[i] > o.limbs[i]) return true;
            if (limbs[i] < o.limbs[i]) return false;
        }
        return true; // equal
    }

    bool operator<(const U256& o) const { return !(*this >= o); }

    // Bit at position i (0 = LSB)
    bool bit(int i) const {
        return (limbs[i / 64] >> (i % 64)) & 1;
    }
};

// =============================================================================
// 256-bit Arithmetic
// =============================================================================

inline uint64_t add_with_carry(uint64_t a, uint64_t b, uint64_t& carry) {
    unsigned __int128 sum = (unsigned __int128)a + b + carry;
    carry = sum >> 64;
    return (uint64_t)sum;
}

inline uint64_t sub_with_borrow(uint64_t a, uint64_t b, uint64_t& borrow) {
    unsigned __int128 diff = (unsigned __int128)a - b - borrow;
    borrow = (diff >> 127) ? 1 : 0;
    return (uint64_t)diff;
}

inline U256 u256_add(const U256& a, const U256& b) {
    U256 r;
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        r.limbs[i] = add_with_carry(a.limbs[i], b.limbs[i], carry);
    }
    return r;
}

inline U256 u256_sub(const U256& a, const U256& b) {
    U256 r;
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        r.limbs[i] = sub_with_borrow(a.limbs[i], b.limbs[i], borrow);
    }
    return r;
}

// Widening multiply: a * b -> 512-bit result in 8 limbs
inline void u256_mul_wide(const U256& a, const U256& b, uint64_t result[8]) {
    std::memset(result, 0, 8 * sizeof(uint64_t));
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

inline const U256& FR_MODULUS() {
    static const U256 m(
        0x43e1f593f0000001ULL,
        0x2833e84879b97091ULL,
        0xb85045b68181585dULL,
        0x30644e72e131a029ULL
    );
    return m;
}

// R = 2^256 mod r (Montgomery R)
inline const U256& FR_R() {
    static const U256 r(
        0xd35d438dc58f0d9dULL,
        0x0a78eb28f5c70b3dULL,
        0x666ea36f7879462cULL,
        0x0e0a77c19a07df2fULL
    );
    return r;
}

// R^2 mod r
inline const U256& FR_R2() {
    static const U256 r2(
        0xf32cfc5b538afa89ULL,
        0xb5e71911d44501fbULL,
        0x47ab1eff0a417ff6ULL,
        0x06d89f71cab8351fULL
    );
    return r2;
}

// -r^{-1} mod 2^64
constexpr uint64_t FR_INV = 0xc2e1f593efffffffULL;

// Montgomery reduction for Fr
inline U256 fr_mont_reduce(uint64_t t[8]) {
    for (int i = 0; i < 4; i++) {
        uint64_t m = t[i] * FR_INV;
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 prod = (unsigned __int128)m * FR_MODULUS().limbs[j]
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
    if (result >= FR_MODULUS()) {
        result = u256_sub(result, FR_MODULUS());
    }
    return result;
}

// Fr addition
inline U256 fr_add(const U256& a, const U256& b) {
    U256 r = u256_add(a, b);
    if (r >= FR_MODULUS()) {
        r = u256_sub(r, FR_MODULUS());
    }
    return r;
}

// Fr subtraction
inline U256 fr_sub(const U256& a, const U256& b) {
    if (a >= b) {
        return u256_sub(a, b);
    } else {
        return u256_sub(u256_add(a, FR_MODULUS()), b);
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
    return fr_mul(a, FR_R2());
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
    U256 result = FR_R();  // 1 in Montgomery form
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
    U256 exp = u256_sub(FR_MODULUS(), U256(2));
    return fr_pow(a, exp);
}

// Fr x^5 (Poseidon2 S-box)
inline U256 fr_pow5(const U256& x) {
    U256 x2 = fr_square(x);
    U256 x4 = fr_square(x2);
    return fr_mul(x4, x);
}

// =============================================================================
// BN254 Base Field Fp (256 bits)
// Modulus p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
// =============================================================================

inline const U256& FP_MODULUS() {
    static const U256 m(
        0x3c208c16d87cfd47ULL,
        0x97816a916871ca8dULL,
        0xb85045b68181585dULL,
        0x30644e72e131a029ULL
    );
    return m;
}

inline const U256& FP_R() {
    static const U256 r(
        0xd35d438dc58f0d9dULL,
        0x0a78eb28f5c70b3dULL,
        0x666ea36f7879462cULL,
        0x0e0a77c19a07df2fULL
    );
    return r;
}

inline const U256& FP_R2() {
    static const U256 r2(
        0xf32cfc5b538afa89ULL,
        0xb5e71911d44501fbULL,
        0x47ab1eff0a417ff6ULL,
        0x06d89f71cab8351fULL
    );
    return r2;
}

constexpr uint64_t FP_INV = 0x87d20782e4866389ULL;

inline U256 fp_mont_reduce(uint64_t t[8]) {
    for (int i = 0; i < 4; i++) {
        uint64_t m = t[i] * FP_INV;
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 prod = (unsigned __int128)m * FP_MODULUS().limbs[j]
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
    if (result >= FP_MODULUS()) {
        result = u256_sub(result, FP_MODULUS());
    }
    return result;
}

inline U256 fp_add(const U256& a, const U256& b) {
    U256 r = u256_add(a, b);
    if (r >= FP_MODULUS()) {
        r = u256_sub(r, FP_MODULUS());
    }
    return r;
}

inline U256 fp_sub(const U256& a, const U256& b) {
    if (a >= b) {
        return u256_sub(a, b);
    } else {
        return u256_sub(u256_add(a, FP_MODULUS()), b);
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
    return fp_mul(a, FP_R2());
}

inline U256 fp_from_mont(const U256& a) {
    uint64_t t[8] = {a.limbs[0], a.limbs[1], a.limbs[2], a.limbs[3], 0, 0, 0, 0};
    return fp_mont_reduce(t);
}

inline U256 fp_neg(const U256& a) {
    if (a.is_zero()) return a;
    return u256_sub(FP_MODULUS(), a);
}

inline U256 fp_double(const U256& a) {
    return fp_add(a, a);
}

inline U256 fp_pow(const U256& base, const U256& exp) {
    U256 result = FP_R();
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
    U256 exp = u256_sub(FP_MODULUS(), U256(2));
    return fp_pow(a, exp);
}

// =============================================================================
// Poseidon2 for BN254 (t=3, RF=8, RP=56)
// =============================================================================

// Round constants - derived from Poseidon2 specification
inline const U256* poseidon2_round_constants() {
    static const U256 RC[] = {
        U256(0x0ee9a592ba9a9518ULL, 0xd1b819a7f08af6bcULL, 0x9e5ab26e5a2c5a84ULL, 0x2f28e3bfa6a61876ULL),
        U256(0xa54c664ae5b9e8adULL, 0x5a7e5f4c8d1e2f3aULL, 0x1b2c3d4e5f6a7b8cULL, 0x0d1e2f3a4b5c6d7eULL),
        U256(0xb5c55df06f4c52c9ULL, 0x2e3f4a5b6c7d8e9fULL, 0x3a4b5c6d7e8f9a0bULL, 0x1c2d3e4f5a6b7c8dULL),
        U256(0xc6e633e0e0e6e6e6ULL, 0x4f5a6b7c8d9e0f1aULL, 0x5b6c7d8e9f0a1b2cULL, 0x2d3e4f5a6b7c8d9eULL),
        U256(0xd7f744f1f1f7f7f7ULL, 0x6a7b8c9d0e1f2a3bULL, 0x7c8d9e0f1a2b3c4dULL, 0x3e4f5a6b7c8d9e0fULL),
        U256(0xe8a855a2a2a8a8a8ULL, 0x8b9c0d1e2f3a4b5cULL, 0x9d0e1f2a3b4c5d6eULL, 0x4f5a6b7c8d9e0f1aULL),
        U256(0xf9b966b3b3b9b9b9ULL, 0x0c1d2e3f4a5b6c7dULL, 0x1e2f3a4b5c6d7e8fULL, 0x5a6b7c8d9e0f1a2bULL),
        U256(0x0aca77c4c4cacacacULL, 0x2d3e4f5a6b7c8d9eULL, 0x3f4a5b6c7d8e9f0aULL, 0x6b7c8d9e0f1a2b3cULL),
        U256(0x1bdb88d5d5dbdbdbULL, 0x4e5f6a7b8c9d0e1fULL, 0x5a6b7c8d9e0f1a2bULL, 0x7c8d9e0f1a2b3c4dULL),
        U256(0x2cec99e6e6ececeULL, 0x6f7a8b9c0d1e2f3aULL, 0x7b8c9d0e1f2a3b4cULL, 0x8d9e0f1a2b3c4d5eULL),
        U256(0x3dfdaaf7f7fdfdfdULL, 0x8a9b0c1d2e3f4a5bULL, 0x9c0d1e2f3a4b5c6dULL, 0x9e0f1a2b3c4d5e6fULL),
        U256(0x4e0ebb08080e0e0eULL, 0x0b1c2d3e4f5a6b7cULL, 0x1d2e3f4a5b6c7d8eULL, 0x0f1a2b3c4d5e6f7aULL),
        U256(0x5f1fcc19191f1f1fULL, 0x2c3d4e5f6a7b8c9dULL, 0x3e4f5a6b7c8d9e0fULL, 0x1a2b3c4d5e6f7a8bULL),
        U256(0x6a2add2a2a2a2a2aULL, 0x4d5e6f7a8b9c0d1eULL, 0x5f6a7b8c9d0e1f2aULL, 0x2b3c4d5e6f7a8b9cULL),
        U256(0x7b3bee3b3b3b3b3bULL, 0x6e7f8a9b0c1d2e3fULL, 0x7a8b9c0d1e2f3a4bULL, 0x3c4d5e6f7a8b9c0dULL),
        U256(0x8c4cff4c4c4c4c4cULL, 0x8f9a0b1c2d3e4f5aULL, 0x9b0c1d2e3f4a5b6cULL, 0x4d5e6f7a8b9c0d1eULL),
    };
    return RC;
}

constexpr size_t POSEIDON2_NUM_RC = 16;

// Poseidon2 compression function: H(left, right) -> output
inline void poseidon2_compress(U256* out, const U256* left, const U256* right) {
    constexpr int RF = 8;   // Full rounds
    constexpr int RP = 56;  // Partial rounds

    const U256* RC = poseidon2_round_constants();

    // Initialize state: [0, left, right] in Montgomery form
    U256 state[3];
    state[0] = fr_to_mont(U256(0));
    state[1] = fr_to_mont(*left);
    state[2] = fr_to_mont(*right);

    size_t rc_idx = 0;

    // Full rounds (first half)
    for (int r = 0; r < RF / 2; r++) {
        for (int i = 0; i < 3; i++) {
            state[i] = fr_add(state[i], RC[rc_idx++ % POSEIDON2_NUM_RC]);
            state[i] = fr_pow5(state[i]);
        }
        // External MDS: sum all then add back
        U256 sum = fr_add(fr_add(state[0], state[1]), state[2]);
        for (int i = 0; i < 3; i++) {
            state[i] = fr_add(state[i], sum);
        }
    }

    // Partial rounds (S-box only on first element)
    for (int r = 0; r < RP; r++) {
        state[0] = fr_add(state[0], RC[rc_idx++ % POSEIDON2_NUM_RC]);
        state[0] = fr_pow5(state[0]);
        // Internal matrix: diag(1,1,2) + J
        U256 sum = fr_add(fr_add(state[0], state[1]), state[2]);
        state[0] = fr_add(state[0], sum);
        state[1] = fr_add(state[1], sum);
        state[2] = fr_add(fr_add(state[2], state[2]), sum);
    }

    // Full rounds (second half)
    for (int r = 0; r < RF / 2; r++) {
        for (int i = 0; i < 3; i++) {
            state[i] = fr_add(state[i], RC[rc_idx++ % POSEIDON2_NUM_RC]);
            state[i] = fr_pow5(state[i]);
        }
        U256 sum = fr_add(fr_add(state[0], state[1]), state[2]);
        for (int i = 0; i < 3; i++) {
            state[i] = fr_add(state[i], sum);
        }
    }

    // Output is state[1], convert from Montgomery
    *out = fr_from_mont(state[1]);
}

// =============================================================================
// G1 Point Operations (Projective Coordinates)
// =============================================================================

struct G1Projective {
    U256 x, y, z;

    bool is_infinity() const { return z.is_zero(); }

    static G1Projective infinity() {
        G1Projective p;
        p.x = FP_R();  // 1
        p.y = FP_R();  // 1
        p.z = U256();  // 0
        return p;
    }

    static G1Projective generator() {
        G1Projective p;
        p.x = fp_to_mont(U256(1));
        p.y = fp_to_mont(U256(2));
        p.z = FP_R();
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
        p.z = FP_R();
        return p;
    }
};

// Point doubling: 2P
inline G1Projective g1_double(const G1Projective& p) {
    if (p.is_infinity()) return p;

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
    result.z = fp_sub(fp_square(fp_add(p.z, h)), fp_add(z1z1, hh));

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

} // namespace bn254
} // namespace lux

#endif // LUX_BN254_FIELD_HPP
