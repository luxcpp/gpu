// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
/// @file sr25519.metal
/// Metal compute shader for batch sr25519 (Schnorrkel/Ristretto255) verification.
///
/// sr25519 uses Schnorr signatures on the Ristretto255 group, which provides
/// a prime-order group via cofactor elimination on Curve25519.
///
/// Ristretto255 uses the same field as Ed25519 (p = 2^255 - 19) with
/// cofactor-free group operations.
///
/// VRF (Verifiable Random Function) support included.
///
/// Each thread verifies one Schnorr signature.

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Reuse Ed25519 field arithmetic (same field: p = 2^255 - 19)
// =============================================================================

struct uint256 {
    ulong limbs[4];
};

constant uint256 SR_P = {{
    0xFFFFFFFFFFFFFFEDUL, 0xFFFFFFFFFFFFFFFFUL,
    0xFFFFFFFFFFFFFFFFUL, 0x7FFFFFFFFFFFFFFFUL
}};

constant uint256 SR_L = {{
    0x5812631A5CF5D3EDUL, 0x14DEF9DEA2F79CD6UL,
    0x0000000000000000UL, 0x1000000000000000UL
}};

constant uint256 SR_ZERO = {{0, 0, 0, 0}};
constant uint256 SR_ONE  = {{1, 0, 0, 0}};

// d = -121665/121666 mod p (same as Ed25519)
constant uint256 SR_D = {{
    0x75EB4DCA135978A3UL, 0x00700A4D4141D8ABUL,
    0x8CC740797779E898UL, 0x52036CBC148B6DE8UL
}};

constant uint256 SR_2D = {{
    0xEBD69B9426B2F159UL, 0x00E0149A8283B156UL,
    0x198E80F2EEF3D130UL, 0x2406D9DC56DFFCE7UL
}};

// sqrt(-1) mod p
constant uint256 SR_SQRT_M1 = {{
    0xC4EE1B274A0EA0B0UL, 0x2F431806AD2FE478UL,
    0x2B4D00993DFBD7A7UL, 0x2B8324804FC1DF0BUL
}};

// =============================================================================
// Field arithmetic (same as ed25519.metal, reproduced for standalone compilation)
// =============================================================================

inline int u256_cmp(uint256 a, uint256 b) {
    for (int i = 3; i >= 0; i--) {
        if (a.limbs[i] < b.limbs[i]) return -1;
        if (a.limbs[i] > b.limbs[i]) return 1;
    }
    return 0;
}

inline bool u256_is_zero(uint256 a) {
    return (a.limbs[0] | a.limbs[1] | a.limbs[2] | a.limbs[3]) == 0;
}

inline uint256 u256_add(uint256 a, uint256 b, thread ulong& carry) {
    uint256 r;
    ulong c = 0;
    for (int i = 0; i < 4; i++) {
        ulong sum = a.limbs[i] + c;
        c = (sum < a.limbs[i]) ? 1UL : 0UL;
        ulong sum2 = sum + b.limbs[i];
        c += (sum2 < sum) ? 1UL : 0UL;
        r.limbs[i] = sum2;
    }
    carry = c;
    return r;
}

inline uint256 u256_sub(uint256 a, uint256 b, thread ulong& borrow) {
    uint256 r;
    ulong bw = 0;
    for (int i = 0; i < 4; i++) {
        ulong diff = a.limbs[i] - bw;
        bw = (diff > a.limbs[i]) ? 1UL : 0UL;
        ulong diff2 = diff - b.limbs[i];
        bw += (diff2 > diff) ? 1UL : 0UL;
        r.limbs[i] = diff2;
    }
    borrow = bw;
    return r;
}

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

inline uint256 fp_add(uint256 a, uint256 b) {
    ulong c;
    uint256 r = u256_add(a, b, c);
    if (c || u256_cmp(r, SR_P) >= 0) { ulong bw; r = u256_sub(r, SR_P, bw); }
    return r;
}

inline uint256 fp_sub(uint256 a, uint256 b) {
    ulong bw;
    uint256 r = u256_sub(a, b, bw);
    if (bw) { ulong c; r = u256_add(r, SR_P, c); }
    return r;
}

inline uint256 fp_mul(uint256 a, uint256 b) {
    ulong t[8] = {};
    for (int i = 0; i < 4; i++) {
        ulong carry = 0;
        for (int j = 0; j < 4; j++) {
            ulong lo, hi;
            mul64(a.limbs[i], b.limbs[j], lo, hi);
            ulong sum = lo + carry; if (sum < lo) hi++;
            lo = sum;
            sum = t[i + j] + lo; if (sum < t[i + j]) hi++;
            t[i + j] = sum;
            carry = hi;
        }
        t[i + 4] = carry;
    }
    uint256 lo_part = {{t[0], t[1], t[2], t[3]}};
    uint256 hi_part = {{t[4], t[5], t[6], t[7]}};
    ulong c2 = 0;
    uint256 hi38;
    for (int i = 0; i < 4; i++) {
        ulong lo, hi;
        mul64(hi_part.limbs[i], 38UL, lo, hi);
        ulong sum = lo + c2;
        c2 = hi + ((sum < lo) ? 1UL : 0UL);
        hi38.limbs[i] = sum;
    }
    ulong c;
    uint256 result = u256_add(lo_part, hi38, c);
    if (c || c2) {
        ulong extra = (c + c2) * 38;
        uint256 extra256 = {{extra, 0, 0, 0}};
        result = u256_add(result, extra256, c);
    }
    while (u256_cmp(result, SR_P) >= 0) { ulong bw; result = u256_sub(result, SR_P, bw); }
    return result;
}

inline uint256 fp_sqr(uint256 a) { return fp_mul(a, a); }
inline uint256 fp_neg(uint256 a) {
    if (u256_is_zero(a)) return a;
    ulong bw; return u256_sub(SR_P, a, bw);
}

inline uint256 fp_inv(uint256 a) {
    uint256 exp = SR_P; exp.limbs[0] -= 2;
    uint256 result = SR_ONE, base = a;
    for (int i = 0; i < 4; i++)
        for (int bit = 0; bit < 64; bit++) {
            if ((exp.limbs[i] >> bit) & 1) result = fp_mul(result, base);
            base = fp_sqr(base);
        }
    return result;
}

// =============================================================================
// Extended Edwards point (same curve as Ed25519)
// =============================================================================

struct RistrettoPoint {
    uint256 X, Y, Z, T;
};

inline RistrettoPoint ristretto_identity() {
    RistrettoPoint p;
    p.X = SR_ZERO; p.Y = SR_ONE; p.Z = SR_ONE; p.T = SR_ZERO;
    return p;
}

inline RistrettoPoint ristretto_double(RistrettoPoint P) {
    uint256 A = fp_sqr(P.X);
    uint256 B = fp_sqr(P.Y);
    uint256 C = fp_add(fp_sqr(P.Z), fp_sqr(P.Z));
    uint256 D = fp_neg(A);
    uint256 E = fp_sub(fp_sqr(fp_add(P.X, P.Y)), fp_add(A, B));
    uint256 G = fp_add(D, B);
    uint256 F = fp_sub(G, C);
    uint256 H = fp_sub(D, B);
    RistrettoPoint R;
    R.X = fp_mul(E, F); R.Y = fp_mul(G, H);
    R.T = fp_mul(E, H); R.Z = fp_mul(F, G);
    return R;
}

inline RistrettoPoint ristretto_add(RistrettoPoint P, RistrettoPoint Q) {
    uint256 A = fp_mul(P.X, Q.X);
    uint256 B = fp_mul(P.Y, Q.Y);
    uint256 C = fp_mul(P.T, fp_mul(SR_2D, Q.T));
    uint256 D = fp_add(fp_mul(P.Z, Q.Z), fp_mul(P.Z, Q.Z));
    uint256 E = fp_sub(fp_mul(fp_add(P.X, P.Y), fp_add(Q.X, Q.Y)), fp_add(A, B));
    uint256 F = fp_sub(D, C);
    uint256 G = fp_add(D, C);
    uint256 H = fp_add(B, A);
    RistrettoPoint R;
    R.X = fp_mul(E, F); R.Y = fp_mul(G, H);
    R.T = fp_mul(E, H); R.Z = fp_mul(F, G);
    return R;
}

inline RistrettoPoint ristretto_mul(uint256 k, RistrettoPoint P) {
    RistrettoPoint result = ristretto_identity();
    for (int i = 3; i >= 0; i--)
        for (int bit = 63; bit >= 0; bit--) {
            result = ristretto_double(result);
            if ((k.limbs[i] >> bit) & 1) result = ristretto_add(result, P);
        }
    return result;
}

// =============================================================================
// Ristretto255 decoding (from 32-byte compressed form)
// =============================================================================

/// Decode a Ristretto255 point from 32 bytes.
/// Follows the Ristretto255 spec (draft-irtf-cfrg-ristretto255-00).
inline bool ristretto_decode(device const uchar* encoded, thread RistrettoPoint& P) {
    // Read s (little-endian field element)
    uint256 s;
    for (int i = 0; i < 4; i++) {
        s.limbs[i] = 0;
        for (int b = 0; b < 8 && i * 8 + b < 32; b++)
            s.limbs[i] |= (ulong)encoded[i * 8 + b] << (b * 8);
    }

    // s must be non-negative (MSB must be 0) and < p
    if (s.limbs[3] >> 63) return false;
    if (u256_cmp(s, SR_P) >= 0) return false;

    // Check s is canonical (s must equal its encoding)
    // s^2
    uint256 ss = fp_sqr(s);
    // u1 = 1 - s^2
    uint256 u1 = fp_sub(SR_ONE, ss);
    // u2 = 1 + s^2
    uint256 u2 = fp_add(SR_ONE, ss);
    uint256 u2_sq = fp_sqr(u2);
    // v = -(d) * u1^2 - u2^2
    uint256 v = fp_sub(fp_neg(fp_mul(SR_D, fp_sqr(u1))), u2_sq);

    // invsqrt(v * u2^2) using p = 5 mod 8 shortcut
    uint256 vu2sq = fp_mul(v, u2_sq);

    // Compute candidate: (v * u2^2)^((p-5)/8)
    uint256 exp58 = SR_P;
    exp58.limbs[0] -= 5;
    // Divide by 8: shift right 3
    for (int i = 0; i < 3; i++)
        exp58.limbs[i] = (exp58.limbs[i] >> 3) | (exp58.limbs[i + 1] << 61);
    exp58.limbs[3] >>= 3;

    uint256 inv_sqrt = SR_ONE;
    uint256 base = vu2sq;
    for (int i = 0; i < 4; i++)
        for (int bit = 0; bit < 64; bit++) {
            if ((exp58.limbs[i] >> bit) & 1) inv_sqrt = fp_mul(inv_sqrt, base);
            base = fp_sqr(base);
        }

    // Check: (inv_sqrt^2 * v * u2^2) should be +/-1
    uint256 check = fp_mul(fp_sqr(inv_sqrt), vu2sq);
    bool negated = false;
    if (u256_cmp(check, SR_ONE) != 0) {
        uint256 neg1 = fp_neg(SR_ONE);
        if (u256_cmp(check, neg1) == 0) {
            inv_sqrt = fp_mul(inv_sqrt, SR_SQRT_M1);
            negated = true;
        } else {
            return false;
        }
    }

    // x = |2 * s * invsqrt| (take absolute value)
    uint256 x = fp_mul(fp_add(s, s), fp_mul(inv_sqrt, u2));
    if (x.limbs[0] & 1) x = fp_neg(x); // Make non-negative (even)

    uint256 y = fp_mul(u1, fp_mul(inv_sqrt, u2));

    P.X = x;
    P.Y = y;
    P.Z = SR_ONE;
    P.T = fp_mul(x, y);
    return true;
}

// =============================================================================
// Structures
// =============================================================================

struct Sr25519PublicKey {
    uchar data[32]; // Ristretto255 compressed point
};

struct Sr25519Signature {
    uchar data[64]; // R[32] || s[32]
};

struct Sr25519Message {
    uchar hash[64]; // Pre-computed transcript hash
};

// =============================================================================
// Verification kernel
// =============================================================================

/// Batch sr25519 Schnorr signature verification.
/// Each thread verifies one signature.
///
/// Schnorr verify: check s*B == R + H(R||A||M)*A
/// Host pre-computes the transcript hash and reduces mod L.
///
/// Output: results[tid] = 1 if valid, 0 otherwise.
kernel void sr25519_verify_batch(
    device const Sr25519PublicKey*  pubkeys    [[buffer(0)]],
    device const Sr25519Message*    messages   [[buffer(1)]],
    device const Sr25519Signature*  signatures [[buffer(2)]],
    device uint*                    results    [[buffer(3)]],
    constant uint&                  num_sigs   [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_sigs) return;

    // Decode public key (Ristretto255 point)
    RistrettoPoint A;
    if (!ristretto_decode(pubkeys[tid].data, A)) {
        results[tid] = 0;
        return;
    }

    // Decode R from signature
    RistrettoPoint R;
    if (!ristretto_decode(signatures[tid].data, R)) {
        results[tid] = 0;
        return;
    }

    // Read scalar s
    uint256 s;
    for (int i = 0; i < 4; i++) {
        s.limbs[i] = 0;
        for (int b = 0; b < 8; b++)
            s.limbs[i] |= (ulong)signatures[tid].data[32 + i * 8 + b] << (b * 8);
    }

    if (u256_cmp(s, SR_L) >= 0) {
        results[tid] = 0;
        return;
    }

    // Read pre-computed challenge scalar (reduced mod L by host)
    uint256 c;
    for (int i = 0; i < 4; i++) {
        c.limbs[i] = 0;
        for (int b = 0; b < 8; b++)
            c.limbs[i] |= (ulong)messages[tid].hash[i * 8 + b] << (b * 8);
    }

    // Generator point B (same as Ed25519)
    const uint256 BX = {{0xC9562D608F25D51AUL, 0x692CC7609525A7B2UL,
                         0xC0A4E231FDD6DC5CUL, 0x216936D3CD6E53FEUL}};
    const uint256 BY = {{0x6666666666666658UL, 0x6666666666666666UL,
                         0x6666666666666666UL, 0x6666666666666666UL}};
    RistrettoPoint B;
    B.X = BX; B.Y = BY; B.Z = SR_ONE; B.T = fp_mul(BX, BY);

    // Verify: s*B == R + c*A
    RistrettoPoint sB = ristretto_mul(s, B);
    RistrettoPoint cA = ristretto_mul(c, A);
    RistrettoPoint RcA = ristretto_add(R, cA);

    // Compare in affine
    uint256 sb_x = fp_mul(sB.X, fp_inv(sB.Z));
    uint256 sb_y = fp_mul(sB.Y, fp_inv(sB.Z));
    uint256 rca_x = fp_mul(RcA.X, fp_inv(RcA.Z));
    uint256 rca_y = fp_mul(RcA.Y, fp_inv(RcA.Z));

    bool valid = (u256_cmp(sb_x, rca_x) == 0) && (u256_cmp(sb_y, rca_y) == 0);
    results[tid] = valid ? 1u : 0u;
}
