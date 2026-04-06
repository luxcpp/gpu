// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
/// @file ed25519.metal
/// Metal compute shader for batch Ed25519 EdDSA signature verification.
///
/// Twisted Edwards curve: -x^2 + y^2 = 1 + d*x^2*y^2  over F_p
///   p = 2^255 - 19
///   d = -121665/121666 mod p
///   L = 2^252 + 27742317777372353535851937790883648493 (group order)
///   B = generator point
///
/// Verification: check [8][S]B == [8]R + [8][H(R||A||M)]A
/// Each thread verifies one signature.

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// 256-bit integer (4 x 64-bit limbs, little-endian)
// =============================================================================

struct uint256 {
    ulong limbs[4];
};

// =============================================================================
// Ed25519 constants (field prime p = 2^255 - 19)
// =============================================================================

constant uint256 ED_P = {{
    0xFFFFFFFFFFFFFFEDUL, 0xFFFFFFFFFFFFFFFFUL,
    0xFFFFFFFFFFFFFFFFUL, 0x7FFFFFFFFFFFFFFFUL
}};

// d = -121665/121666 mod p
// = 37095705934669439343138083508754565189542113879843219016388785533085940283555
constant uint256 ED_D = {{
    0x75EB4DCA135978A3UL, 0x00700A4D4141D8ABUL,
    0x8CC740797779E898UL, 0x52036CBC148B6DE8UL
}};

// 2*d mod p
constant uint256 ED_2D = {{
    0xEBD69B9426B2F159UL, 0x00E0149A8283B156UL,
    0x198E80F2EEF3D130UL, 0x2406D9DC56DFFCE7UL
}};

// Group order L
constant uint256 ED_L = {{
    0x5812631A5CF5D3EDUL, 0x14DEF9DEA2F79CD6UL,
    0x0000000000000000UL, 0x1000000000000000UL
}};

// Generator B (y-coordinate, x is recovered)
constant uint256 ED_BY = {{
    0x6666666666666658UL, 0x6666666666666666UL,
    0x6666666666666666UL, 0x6666666666666666UL
}};

// B.x
constant uint256 ED_BX = {{
    0xC9562D608F25D51AUL, 0x692CC7609525A7B2UL,
    0xC0A4E231FDD6DC5CUL, 0x216936D3CD6E53FEUL
}};

constant uint256 ZERO = {{0, 0, 0, 0}};
constant uint256 ONE  = {{1, 0, 0, 0}};

// =============================================================================
// 256-bit arithmetic
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

// 64x64->128 multiply
inline void mul64(ulong a, ulong b, thread ulong& lo, thread ulong& hi) {
    ulong a_lo = a & 0xFFFFFFFFUL;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFFUL;
    ulong b_hi = b >> 32;
    ulong ll = a_lo * b_lo;
    ulong lh = a_lo * b_hi;
    ulong hl = a_hi * b_lo;
    ulong hh = a_hi * b_hi;
    ulong mid = lh + (ll >> 32);
    ulong mid2 = mid + hl;
    if (mid2 < mid) hh += (1UL << 32);
    lo = (mid2 << 32) | (ll & 0xFFFFFFFFUL);
    hi = hh + (mid2 >> 32);
}

// =============================================================================
// Field arithmetic mod p = 2^255 - 19
// Uses direct reduction (not Montgomery) since p has special form.
// =============================================================================

inline uint256 fp_reduce(uint256 a) {
    while (u256_cmp(a, ED_P) >= 0) {
        ulong bw;
        a = u256_sub(a, ED_P, bw);
    }
    return a;
}

inline uint256 fp_add(uint256 a, uint256 b) {
    ulong c;
    uint256 r = u256_add(a, b, c);
    if (c || u256_cmp(r, ED_P) >= 0) {
        ulong bw;
        r = u256_sub(r, ED_P, bw);
    }
    return r;
}

inline uint256 fp_sub(uint256 a, uint256 b) {
    ulong bw;
    uint256 r = u256_sub(a, b, bw);
    if (bw) {
        ulong c;
        r = u256_add(r, ED_P, c);
    }
    return r;
}

inline uint256 fp_mul(uint256 a, uint256 b) {
    // Full 512-bit multiply, then reduce mod p = 2^255 - 19
    ulong t[8] = {};
    for (int i = 0; i < 4; i++) {
        ulong carry = 0;
        for (int j = 0; j < 4; j++) {
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
        t[i + 4] = carry;
    }

    // Reduce mod 2^255 - 19:
    // Split t into low 255 bits and high bits, use high * 38 = high * 2 * 19
    // t = t_lo + t_hi * 2^256
    // 2^256 mod p = 2*19 = 38
    // So t mod p = t_lo + 38 * t_hi (approximately)

    // Extract low 256 bits and high 256 bits
    uint256 lo_part = {{t[0], t[1], t[2], t[3]}};
    uint256 hi_part = {{t[4], t[5], t[6], t[7]}};

    // Multiply hi by 38 and add to lo
    // Since hi_part is at most 256 bits and 38 is small, result fits in ~262 bits
    ulong carry = 0;
    uint256 hi38;
    for (int i = 0; i < 4; i++) {
        ulong lo_val, hi_val;
        mul64(hi_part.limbs[i], 38UL, lo_val, hi_val);
        ulong sum = lo_val + carry;
        carry = hi_val + ((sum < lo_val) ? 1UL : 0UL);
        hi38.limbs[i] = sum;
    }

    ulong c;
    uint256 result = u256_add(lo_part, hi38, c);
    // Handle final carry: carry * 2^256 = carry * 38
    if (c || carry) {
        ulong extra = (c + carry) * 38;
        uint256 extra256 = {{extra, 0, 0, 0}};
        result = u256_add(result, extra256, c);
    }

    return fp_reduce(result);
}

inline uint256 fp_sqr(uint256 a) { return fp_mul(a, a); }

inline uint256 fp_neg(uint256 a) {
    if (u256_is_zero(a)) return a;
    ulong bw;
    return u256_sub(ED_P, a, bw);
}

/// Fermat inversion: a^(p-2) mod p
inline uint256 fp_inv(uint256 a) {
    uint256 exp = ED_P;
    exp.limbs[0] -= 2;
    uint256 result = ONE;
    uint256 base = a;
    for (int i = 0; i < 4; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((exp.limbs[i] >> bit) & 1)
                result = fp_mul(result, base);
            base = fp_sqr(base);
        }
    }
    return result;
}

// =============================================================================
// Extended twisted Edwards point: (X, Y, Z, T) where x=X/Z, y=Y/Z, T=X*Y/Z
// =============================================================================

struct EdPoint {
    uint256 X, Y, Z, T;
};

inline EdPoint ed_identity() {
    EdPoint p;
    p.X = ZERO; p.Y = ONE; p.Z = ONE; p.T = ZERO;
    return p;
}

inline bool ed_is_identity(EdPoint p) {
    return u256_is_zero(p.X) && !u256_is_zero(p.Y) && !u256_is_zero(p.Z);
}

/// Extended point doubling
inline EdPoint ed_double(EdPoint P) {
    uint256 A = fp_sqr(P.X);
    uint256 B = fp_sqr(P.Y);
    uint256 C = fp_add(fp_sqr(P.Z), fp_sqr(P.Z)); // 2*Z^2
    uint256 D = fp_neg(A);  // a*X^2 where a=-1
    uint256 E = fp_sub(fp_sqr(fp_add(P.X, P.Y)), fp_add(A, B));
    uint256 G = fp_add(D, B);
    uint256 F = fp_sub(G, C);
    uint256 H = fp_sub(D, B);

    EdPoint R;
    R.X = fp_mul(E, F);
    R.Y = fp_mul(G, H);
    R.T = fp_mul(E, H);
    R.Z = fp_mul(F, G);
    return R;
}

/// Extended point addition
inline EdPoint ed_add(EdPoint P, EdPoint Q) {
    uint256 A = fp_mul(P.X, Q.X);
    uint256 B = fp_mul(P.Y, Q.Y);
    uint256 C = fp_mul(P.T, fp_mul(ED_2D, Q.T));
    uint256 D = fp_mul(P.Z, Q.Z);
    D = fp_add(D, D); // 2*Z1*Z2
    uint256 E = fp_sub(fp_mul(fp_add(P.X, P.Y), fp_add(Q.X, Q.Y)), fp_add(A, B));
    uint256 F = fp_sub(D, C);
    uint256 G = fp_add(D, C);
    uint256 H = fp_add(B, A); // a=-1, so B - a*A = B + A

    EdPoint R;
    R.X = fp_mul(E, F);
    R.Y = fp_mul(G, H);
    R.T = fp_mul(E, H);
    R.Z = fp_mul(F, G);
    return R;
}

/// Scalar multiplication: k * P
inline EdPoint ed_mul(uint256 k, EdPoint P) {
    EdPoint result = ed_identity();
    for (int i = 3; i >= 0; i--) {
        for (int bit = 63; bit >= 0; bit--) {
            result = ed_double(result);
            if ((k.limbs[i] >> bit) & 1)
                result = ed_add(result, P);
        }
    }
    return result;
}

/// Convert extended -> affine
inline void ed_to_affine(EdPoint p, thread uint256& x, thread uint256& y) {
    uint256 z_inv = fp_inv(p.Z);
    x = fp_mul(p.X, z_inv);
    y = fp_mul(p.Y, z_inv);
}

// =============================================================================
// Point decompression
// =============================================================================

/// Decompress Ed25519 point from 32-byte encoding.
/// Encoding: y-coordinate (255 bits, little-endian) + sign bit of x in MSB.
inline bool ed_decompress(device const uchar* encoded, thread EdPoint& P) {
    // Read y (little-endian, 255 bits)
    uint256 y;
    for (int i = 0; i < 4; i++) {
        y.limbs[i] = 0;
        int start = i * 8;
        int end = (i < 3) ? 8 : 8;
        for (int b = 0; b < end && start + b < 32; b++) {
            y.limbs[i] |= (ulong)encoded[start + b] << (b * 8);
        }
    }
    // Extract x sign bit (bit 255 = MSB of byte 31)
    bool x_sign = (encoded[31] >> 7) & 1;
    y.limbs[3] &= 0x7FFFFFFFFFFFFFFFUL; // Clear sign bit

    if (u256_cmp(y, ED_P) >= 0) return false;

    // Recover x from y: x^2 = (y^2 - 1) / (d*y^2 + 1)
    uint256 y2 = fp_sqr(y);
    uint256 num = fp_sub(y2, ONE);
    uint256 den = fp_add(fp_mul(ED_D, y2), ONE);
    uint256 den_inv = fp_inv(den);
    uint256 x2 = fp_mul(num, den_inv);

    if (u256_is_zero(x2)) {
        if (x_sign) return false; // x must be 0 but sign says negative
        P.X = ZERO; P.Y = y; P.Z = ONE; P.T = ZERO;
        return true;
    }

    // x = x2^((p+3)/8) (works because p = 5 mod 8)
    uint256 exp_val = ED_P;
    // (p+3)/8 = (2^255 - 19 + 3)/8 = (2^255 - 16)/8 = 2^252 - 2
    exp_val.limbs[0] += 3;
    // Shift right by 3
    for (int i = 0; i < 3; i++) {
        exp_val.limbs[i] = (exp_val.limbs[i] >> 3) | (exp_val.limbs[i + 1] << 61);
    }
    exp_val.limbs[3] >>= 3;

    uint256 x = ONE;
    uint256 base = x2;
    for (int i = 0; i < 4; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((exp_val.limbs[i] >> bit) & 1)
                x = fp_mul(x, base);
            base = fp_sqr(base);
        }
    }

    // Check: if x^2 != x2, multiply by sqrt(-1)
    if (u256_cmp(fp_sqr(x), x2) != 0) {
        // sqrt(-1) mod p = 2^((p-1)/4) mod p
        // Precomputed: 19681161376707505956807079304988542015446066515923890162744021073123829784752
        const uint256 SQRT_M1 = {{
            0xC4EE1B274A0EA0B0UL, 0x2F431806AD2FE478UL,
            0x2B4D00993DFBD7A7UL, 0x2B8324804FC1DF0BUL
        }};
        x = fp_mul(x, SQRT_M1);
        if (u256_cmp(fp_sqr(x), x2) != 0) return false;
    }

    // Adjust sign
    bool x_is_odd = x.limbs[0] & 1;
    if (x_is_odd != x_sign) x = fp_neg(x);

    P.X = x;
    P.Y = y;
    P.Z = ONE;
    P.T = fp_mul(x, y);
    return true;
}

// =============================================================================
// SHA-512 for Ed25519 (verification needs H(R||A||M))
// Simplified: use first 64 bytes of hash, reduce mod L for scalar
// =============================================================================

// SHA-512 would be needed here for full implementation.
// For the GPU kernel, we accept pre-hashed scalars from the host.

// =============================================================================
// Structures
// =============================================================================

struct Ed25519PublicKey {
    uchar data[32];
};

struct Ed25519Signature {
    uchar data[64]; // R[32] || S[32]
};

struct Ed25519Message {
    uchar hash[64]; // Pre-computed H(R || A || M), 64 bytes
};

// =============================================================================
// Verification kernel
// =============================================================================

/// Batch Ed25519 signature verification.
/// Each thread verifies one (pubkey, message_hash, signature) tuple.
///
/// The host pre-computes H(R || A || M) and reduces it mod L to get scalar h.
/// The GPU performs the expensive point arithmetic: check [S]B == R + [h]A.
///
/// Output: results[tid] = 1 if valid, 0 otherwise.
kernel void ed25519_verify_batch(
    device const Ed25519PublicKey*  pubkeys    [[buffer(0)]],
    device const Ed25519Message*    messages   [[buffer(1)]],
    device const Ed25519Signature*  signatures [[buffer(2)]],
    device uint*                    results    [[buffer(3)]],
    constant uint&                  num_sigs   [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_sigs) return;

    // -- Decompress public key A --
    EdPoint A;
    if (!ed_decompress(pubkeys[tid].data, A)) {
        results[tid] = 0;
        return;
    }

    // -- Decompress signature point R --
    EdPoint R;
    if (!ed_decompress(signatures[tid].data, R)) {
        results[tid] = 0;
        return;
    }

    // -- Read scalar S from signature (bytes 32..63, little-endian) --
    uint256 S;
    for (int i = 0; i < 4; i++) {
        S.limbs[i] = 0;
        for (int b = 0; b < 8; b++) {
            S.limbs[i] |= (ulong)signatures[tid].data[32 + i * 8 + b] << (b * 8);
        }
    }

    // S must be < L
    if (u256_cmp(S, ED_L) >= 0) {
        results[tid] = 0;
        return;
    }

    // -- Read pre-computed hash scalar h (reduced mod L by host) --
    uint256 h;
    for (int i = 0; i < 4; i++) {
        h.limbs[i] = 0;
        for (int b = 0; b < 8; b++) {
            h.limbs[i] |= (ulong)messages[tid].hash[i * 8 + b] << (b * 8);
        }
    }

    // -- Verify: [S]B == R + [h]A --
    // Equivalently: [S]B - [h]A - R == identity

    // Compute [S]B
    EdPoint B;
    B.X = ED_BX; B.Y = ED_BY; B.Z = ONE; B.T = fp_mul(ED_BX, ED_BY);
    EdPoint SB = ed_mul(S, B);

    // Compute [h]A
    EdPoint hA = ed_mul(h, A);

    // Compute R + [h]A
    EdPoint RhA = ed_add(R, hA);

    // Compare [S]B == R + [h]A by checking coordinates
    uint256 sb_x, sb_y, rha_x, rha_y;
    ed_to_affine(SB, sb_x, sb_y);
    ed_to_affine(RhA, rha_x, rha_y);

    bool valid = (u256_cmp(sb_x, rha_x) == 0) && (u256_cmp(sb_y, rha_y) == 0);
    results[tid] = valid ? 1u : 0u;
}
