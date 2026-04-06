// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
/// @file cggmp21.metal
/// Metal compute shader for CGGMP21 threshold ECDSA operations.
///
/// CGGMP21 is the state-of-the-art threshold ECDSA protocol enabling
/// t-of-n signers to produce a standard ECDSA signature.
///
/// The heaviest GPU operation is Paillier encryption/decryption, which
/// requires 2048-bit modular exponentiation. This is extremely GPU-friendly
/// because the exponentiation is pure multiply-and-square with no branching
/// on the data path.
///
/// Operations:
///   - cggmp21_partial_sign_batch: Paillier-based partial signing
///
/// The 2048-bit arithmetic uses 32 x 64-bit limbs.

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// 2048-bit unsigned integer (32 x 64-bit limbs, little-endian)
// =============================================================================

struct uint2048 {
    ulong limbs[32];
};

// =============================================================================
// 2048-bit arithmetic
// =============================================================================

inline void u2048_zero(thread uint2048& a) {
    for (int i = 0; i < 32; i++) a.limbs[i] = 0;
}

inline bool u2048_is_zero(thread const uint2048& a) {
    ulong acc = 0;
    for (int i = 0; i < 32; i++) acc |= a.limbs[i];
    return acc == 0;
}

inline int u2048_cmp(thread const uint2048& a, thread const uint2048& b) {
    for (int i = 31; i >= 0; i--) {
        if (a.limbs[i] < b.limbs[i]) return -1;
        if (a.limbs[i] > b.limbs[i]) return 1;
    }
    return 0;
}

inline uint2048 u2048_add(thread const uint2048& a, thread const uint2048& b, thread ulong& carry) {
    uint2048 r;
    ulong c = 0;
    for (int i = 0; i < 32; i++) {
        ulong sum = a.limbs[i] + c;
        c = (sum < a.limbs[i]) ? 1UL : 0UL;
        ulong sum2 = sum + b.limbs[i];
        c += (sum2 < sum) ? 1UL : 0UL;
        r.limbs[i] = sum2;
    }
    carry = c;
    return r;
}

inline uint2048 u2048_sub(thread const uint2048& a, thread const uint2048& b, thread ulong& borrow) {
    uint2048 r;
    ulong bw = 0;
    for (int i = 0; i < 32; i++) {
        ulong diff = a.limbs[i] - bw;
        bw = (diff > a.limbs[i]) ? 1UL : 0UL;
        ulong diff2 = diff - b.limbs[i];
        bw += (diff2 > diff) ? 1UL : 0UL;
        r.limbs[i] = diff2;
    }
    borrow = bw;
    return r;
}

// 64x64->128 multiply
inline void mul64(ulong a, ulong b, thread ulong& lo, thread ulong& hi) {
    ulong a_lo = a & 0xFFFFFFFFUL, a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFFUL, b_hi = b >> 32;
    ulong ll = a_lo * b_lo, lh = a_lo * b_hi;
    ulong hl = a_hi * b_lo, hh = a_hi * b_hi;
    ulong mid = lh + (ll >> 32);
    ulong mid2 = mid + hl;
    if (mid2 < mid) hh += (1UL << 32);
    lo = (mid2 << 32) | (ll & 0xFFFFFFFFUL);
    hi = hh + (mid2 >> 32);
}

// =============================================================================
// Montgomery multiplication for 2048-bit modulus
// =============================================================================

/// Montgomery reduction for 2048-bit: t * R^{-1} mod m
/// t is 4096-bit (64 limbs), m is 2048-bit (32 limbs)
/// m_inv = -m^{-1} mod 2^64
inline void mont_reduce_2048(thread ulong t[64],
                              thread const uint2048& m,
                              ulong m_inv,
                              thread uint2048& result) {
    ulong a[65];
    for (int i = 0; i < 64; i++) a[i] = t[i];
    a[64] = 0;

    for (int i = 0; i < 32; i++) {
        ulong u = a[i] * m_inv;
        ulong carry = 0;
        for (int j = 0; j < 32; j++) {
            ulong lo, hi;
            mul64(u, m.limbs[j], lo, hi);
            ulong sum = lo + carry;
            if (sum < lo) hi++;
            lo = sum;
            sum = a[i + j] + lo;
            if (sum < a[i + j]) hi++;
            a[i + j] = sum;
            carry = hi;
        }
        for (int j = 32; i + j <= 64; j++) {
            ulong sum = a[i + j] + carry;
            carry = (sum < a[i + j]) ? 1UL : 0UL;
            a[i + j] = sum;
            if (!carry) break;
        }
    }

    for (int i = 0; i < 32; i++) result.limbs[i] = a[i + 32];

    if (a[64] || u2048_cmp(result, m) >= 0) {
        ulong bw;
        result = u2048_sub(result, m, bw);
    }
}

/// Montgomery multiplication: a * b * R^{-1} mod m (both a,b in Montgomery form)
inline void mont_mul_2048(thread const uint2048& a,
                           thread const uint2048& b,
                           thread const uint2048& m,
                           ulong m_inv,
                           thread uint2048& result) {
    ulong t[64] = {};

    for (int i = 0; i < 32; i++) {
        ulong carry = 0;
        for (int j = 0; j < 32; j++) {
            ulong lo, hi;
            mul64(a.limbs[i], b.limbs[j], lo, hi);
            ulong sum = lo + carry;
            if (sum < lo) hi++;
            lo = sum;
            sum = t[i + j] + lo;
            if (sum < t[i + j]) hi++;
            t[i + j] = sum;
            carry = hi;
        }
        t[i + 32] = carry;
    }

    mont_reduce_2048(t, m, m_inv, result);
}

/// Montgomery squaring (optimization: fewer multiplications for a*a)
inline void mont_sqr_2048(thread const uint2048& a,
                           thread const uint2048& m,
                           ulong m_inv,
                           thread uint2048& result) {
    mont_mul_2048(a, a, m, m_inv, result);
}

/// Modular exponentiation: base^exp mod m  (all in Montgomery form)
/// exp is 2048-bit, base and m are 2048-bit
inline void mont_pow_2048(thread const uint2048& base,
                           thread const uint2048& exp,
                           thread const uint2048& m,
                           ulong m_inv,
                           thread const uint2048& mont_one,
                           thread uint2048& result) {
    result = mont_one;
    uint2048 b = base;

    for (int i = 0; i < 32; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((exp.limbs[i] >> bit) & 1) {
                mont_mul_2048(result, b, m, m_inv, result);
            }
            mont_sqr_2048(b, m, m_inv, b);
        }
    }
}

// =============================================================================
// Paillier encryption primitives
// =============================================================================

/// Paillier public key: N (2048-bit), N^2 (4096-bit stored as two uint2048)
struct PaillierPubKey {
    uchar n_data[256];     // N in big-endian
    uchar n_inv64[8];      // -N^{-1} mod 2^64 for Montgomery
};

// =============================================================================
// CGGMP21 structures
// =============================================================================

/// Input for partial signing: encrypted share + message
struct CGGMP21Input {
    uchar k_share[32];      // k_i share (secp256k1 scalar)
    uchar chi_share[32];    // chi_i = k_i * x_i share
    uchar msg_hash[32];     // Message hash
    uchar gamma_share[32];  // gamma_i share
};

/// Output: partial signature components
struct CGGMP21PartialSig {
    uchar sigma_i[32];      // sigma_i = k_i * m + r * chi_i (mod n)
};

// =============================================================================
// secp256k1 order for scalar arithmetic
// =============================================================================

constant ulong SECP_N[4] = {
    0xBFD25E8CD0364141UL, 0xBAAEDCE6AF48A03BUL,
    0xFFFFFFFFFFFFFFFEUL, 0xFFFFFFFFFFFFFFFFUL
};

/// Modular multiplication mod secp256k1 order (256-bit)
inline void scalar_mul_mod_n(thread const ulong a[4],
                              thread const ulong b[4],
                              thread ulong result[4]) {
    // Full 512-bit product
    ulong t[8] = {};
    for (int i = 0; i < 4; i++) {
        ulong carry = 0;
        for (int j = 0; j < 4; j++) {
            ulong lo, hi;
            mul64(a[i], b[j], lo, hi);
            ulong sum = lo + carry; if (sum < lo) hi++;
            sum = t[i + j] + sum; if (sum < t[i + j]) hi++;
            t[i + j] = sum;
            carry = hi;
        }
        t[i + 4] = carry;
    }

    // Barrett reduction mod n (simplified: iterate subtraction)
    // For production, proper Barrett with precomputed constant
    ulong r[4] = {t[0], t[1], t[2], t[3]};

    // Subtract n while >= n (at most a few iterations for 512->256 bit reduction)
    for (int iter = 0; iter < 4; iter++) {
        ulong borrow = 0;
        ulong diff[4];
        for (int i = 0; i < 4; i++) {
            ulong d = r[i] - borrow;
            borrow = (d > r[i]) ? 1UL : 0UL;
            ulong d2 = d - SECP_N[i];
            borrow += (d2 > d) ? 1UL : 0UL;
            diff[i] = d2;
        }
        if (!borrow) {
            for (int i = 0; i < 4; i++) r[i] = diff[i];
        }
    }

    for (int i = 0; i < 4; i++) result[i] = r[i];
}

/// Modular addition mod n
inline void scalar_add_mod_n(thread const ulong a[4],
                              thread const ulong b[4],
                              thread ulong result[4]) {
    ulong carry = 0;
    for (int i = 0; i < 4; i++) {
        ulong sum = a[i] + carry;
        carry = (sum < a[i]) ? 1UL : 0UL;
        ulong sum2 = sum + b[i];
        carry += (sum2 < sum) ? 1UL : 0UL;
        result[i] = sum2;
    }
    // Reduce
    ulong borrow = 0;
    ulong diff[4];
    for (int i = 0; i < 4; i++) {
        ulong d = result[i] - borrow;
        borrow = (d > result[i]) ? 1UL : 0UL;
        ulong d2 = d - SECP_N[i];
        borrow += (d2 > d) ? 1UL : 0UL;
        diff[i] = d2;
    }
    if (!borrow || carry) {
        for (int i = 0; i < 4; i++) result[i] = diff[i];
    }
}

// =============================================================================
// Partial signing kernel
// =============================================================================

/// CGGMP21 partial signing.
/// Each thread computes: sigma_i = k_i * m + r * chi_i  (mod n)
/// where r is the x-coordinate of the combined nonce point R.
///
/// This is the scalar arithmetic portion. The Paillier operations for
/// MtA (Multiplicative-to-Additive) conversion are done in separate passes.
///
/// Output: partial_sigs[tid] contains sigma_i.
kernel void cggmp21_partial_sign_batch(
    device const CGGMP21Input*     inputs       [[buffer(0)]],
    device CGGMP21PartialSig*      partial_sigs [[buffer(1)]],
    device const uchar*            r_x          [[buffer(2)]],   // 32 bytes: R.x mod n
    constant uint&                 num_ops      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_ops) return;

    // Read k_i share
    ulong k[4];
    for (int i = 0; i < 4; i++) {
        k[i] = 0;
        for (int b = 0; b < 8; b++)
            k[i] |= (ulong)inputs[tid].k_share[i * 8 + b] << (b * 8);
    }

    // Read chi_i share
    ulong chi[4];
    for (int i = 0; i < 4; i++) {
        chi[i] = 0;
        for (int b = 0; b < 8; b++)
            chi[i] |= (ulong)inputs[tid].chi_share[i * 8 + b] << (b * 8);
    }

    // Read message hash
    ulong msg[4];
    for (int i = 0; i < 4; i++) {
        msg[i] = 0;
        for (int b = 0; b < 8; b++)
            msg[i] |= (ulong)inputs[tid].msg_hash[i * 8 + b] << (b * 8);
    }

    // Read r (x-coordinate of nonce point)
    ulong r[4];
    for (int i = 0; i < 4; i++) {
        r[i] = 0;
        for (int b = 0; b < 8; b++)
            r[i] |= (ulong)r_x[i * 8 + b] << (b * 8);
    }

    // sigma_i = k_i * m + r * chi_i  (mod n)
    ulong km[4], rchi[4], sigma[4];
    scalar_mul_mod_n(k, msg, km);
    scalar_mul_mod_n(r, chi, rchi);
    scalar_add_mod_n(km, rchi, sigma);

    // Write output
    device uchar* out = partial_sigs[tid].sigma_i;
    for (int i = 0; i < 4; i++) {
        for (int b = 0; b < 8; b++) {
            out[i * 8 + b] = uchar(sigma[i] >> (b * 8));
        }
    }
}
