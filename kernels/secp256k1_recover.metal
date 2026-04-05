// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
/// @file secp256k1_recover.metal
/// Metal compute shader for batch secp256k1 ECDSA public key recovery.
///
/// Each GPU thread recovers one (r, s, v, msg_hash) tuple into an Ethereum
/// address (20 bytes). This is the critical EVM operation that dominates block
/// processing time.
///
/// Algorithm per thread:
///   1. Decompress r → point R using recovery flag v
///   2. Compute s_inv = s^(-1) mod n  (Fermat's little theorem)
///   3. Compute Q = s_inv * (s*R - hash*G) = s_inv*s*R - s_inv*hash*G
///      Equivalently: Q = r_inv * (s*R - hash*G)  -- but we use the standard form:
///        Q = s_inv * (s * R - hash * G)  which simplifies to s_inv*s*R - s_inv*hash*G
///      Actually the standard ecrecover formula is:
///        Q = r^(-1) * (s * R - e * G)
///      where e = msg_hash, R = decompressed point from r with parity v.
///   4. Serialize Q as uncompressed (x || y), 64 bytes
///   5. Keccak-256(Q.x || Q.y), take last 20 bytes = Ethereum address
///
/// secp256k1 curve:  y^2 = x^3 + 7  over F_p
///   p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
///   n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
///   G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
///        0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// 256-bit unsigned integer (4 x 64-bit limbs, little-endian)
// =============================================================================

struct uint256 {
    ulong limbs[4]; // limbs[0] = least significant
};

// =============================================================================
// secp256k1 constants
// =============================================================================

// Field prime p = 2^256 - 2^32 - 977
constant uint256 SECP256K1_P = {{
    0xFFFFFFFEFFFFFC2FUL, 0xFFFFFFFFFFFFFFFFUL,
    0xFFFFFFFFFFFFFFFFUL, 0xFFFFFFFFFFFFFFFFUL
}};

// Curve order n
constant uint256 SECP256K1_N = {{
    0xBFD25E8CD0364141UL, 0xBAAEDCE6AF48A03BUL,
    0xFFFFFFFFFFFFFFFEUL, 0xFFFFFFFFFFFFFFFFUL
}};

// Generator point G (affine coordinates)
constant uint256 GX = {{
    0x59F2815B16F81798UL, 0x029BFCDB2DCE28D9UL,
    0x55A06295CE870B07UL, 0x79BE667EF9DCBBACUL
}};
constant uint256 GY = {{
    0x9C47D08FFB10D4B8UL, 0xFD17B448A6855419UL,
    0x5DA4FBFC0E1108A8UL, 0x483ADA7726A3C465UL
}};

// Montgomery constants for field p:
//   R = 2^256 mod p
//   R^2 mod p  (for Montgomery encoding)
//   p_inv = -p^(-1) mod 2^64
constant uint256 MONT_R_P = {{
    0x00000001000003D1UL, 0x0000000000000000UL,
    0x0000000000000000UL, 0x0000000000000000UL
}};
constant uint256 MONT_R2_P = {{
    0x000007A2000E90A1UL, 0x0000000000000001UL,
    0x0000000000000000UL, 0x0000000000000000UL
}};
constant ulong P_INV = 0xD838091DD2253531UL; // -p^(-1) mod 2^64

// Montgomery constants for order n:
//   R^2 mod n
//   n_inv = -n^(-1) mod 2^64
constant uint256 MONT_R2_N = {{
    0x896CF21467D7D140UL, 0x741496C20E7CF878UL,
    0xE697F5E45BCD07C6UL, 0x9D671CD581C69BC5UL
}};
constant ulong N_INV = 0x4B0DFF665588B13FUL; // -n^(-1) mod 2^64

// Zero and one
constant uint256 ZERO256 = {{0, 0, 0, 0}};
constant uint256 ONE256 = {{1, 0, 0, 0}};

// =============================================================================
// 256-bit arithmetic helpers
// =============================================================================

// Compare: returns -1 if a < b, 0 if a == b, 1 if a > b
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

// a + b, returns carry
inline uint256 u256_add(uint256 a, uint256 b, thread ulong &carry) {
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

// a - b, returns borrow
inline uint256 u256_sub(uint256 a, uint256 b, thread ulong &borrow) {
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

// =============================================================================
// Modular arithmetic over F_p (Montgomery form)
// =============================================================================

// Montgomery reduction: given T (up to 512-bit), compute T * R^(-1) mod m
// where m is either p or n, inv is the corresponding -m^(-1) mod 2^64
inline uint256 mont_reduce(ulong t[8], uint256 m, ulong inv) {
    // CIOS (Coarsely Integrated Operand Scanning) reduction
    ulong a[8];
    for (int i = 0; i < 8; i++) a[i] = t[i];

    for (int i = 0; i < 4; i++) {
        ulong u = a[i] * inv;

        // a += u * m * 2^(64*i)  (but we just process limb by limb)
        ulong carry = 0;
        for (int j = 0; j < 4; j++) {
            // a[i+j] += u * m.limbs[j] + carry
            // Use 128-bit multiplication via two 64x64->128 bit ops
            ulong hi, lo;

            // u * m.limbs[j]
            // Metal doesn't have native 128-bit multiply, so we split:
            ulong u_lo = u & 0xFFFFFFFFUL;
            ulong u_hi = u >> 32;
            ulong m_lo = m.limbs[j] & 0xFFFFFFFFUL;
            ulong m_hi = m.limbs[j] >> 32;

            ulong ll = u_lo * m_lo;
            ulong lh = u_lo * m_hi;
            ulong hl = u_hi * m_lo;
            ulong hh = u_hi * m_hi;

            ulong mid = lh + (ll >> 32);
            ulong mid2 = mid + hl;
            if (mid2 < mid) hh += (1UL << 32);

            lo = (mid2 << 32) | (ll & 0xFFFFFFFFUL);
            hi = hh + (mid2 >> 32);

            // Add carry
            ulong sum = lo + carry;
            if (sum < lo) hi++;
            lo = sum;

            // Add to a[i+j]
            sum = a[i + j] + lo;
            if (sum < a[i + j]) hi++;
            a[i + j] = sum;
            carry = hi;
        }
        // Propagate carry
        for (int j = 4; i + j < 8; j++) {
            ulong sum = a[i + j] + carry;
            carry = (sum < a[i + j]) ? 1UL : 0UL;
            a[i + j] = sum;
            if (carry == 0) break;
        }
    }

    // Result is in a[4..7]
    uint256 r;
    r.limbs[0] = a[4];
    r.limbs[1] = a[5];
    r.limbs[2] = a[6];
    r.limbs[3] = a[7];

    // Final subtraction if r >= m
    if (u256_cmp(r, m) >= 0) {
        ulong bw;
        r = u256_sub(r, m, bw);
    }
    return r;
}

// Montgomery multiplication: a * b * R^(-1) mod m
inline uint256 mont_mul(uint256 a, uint256 b, uint256 m, ulong inv) {
    ulong t[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < 4; i++) {
        ulong carry = 0;
        for (int j = 0; j < 4; j++) {
            // t[i+j] += a.limbs[i] * b.limbs[j] + carry
            ulong a_lo = a.limbs[i] & 0xFFFFFFFFUL;
            ulong a_hi = a.limbs[i] >> 32;
            ulong b_lo = b.limbs[j] & 0xFFFFFFFFUL;
            ulong b_hi = b.limbs[j] >> 32;

            ulong ll = a_lo * b_lo;
            ulong lh = a_lo * b_hi;
            ulong hl = a_hi * b_lo;
            ulong hh = a_hi * b_hi;

            ulong mid = lh + (ll >> 32);
            ulong mid2 = mid + hl;
            if (mid2 < mid) hh += (1UL << 32);

            ulong lo = (mid2 << 32) | (ll & 0xFFFFFFFFUL);
            ulong hi = hh + (mid2 >> 32);

            // Add carry
            ulong sum = lo + carry;
            if (sum < lo) hi++;
            lo = sum;

            // Add to t[i+j]
            sum = t[i + j] + lo;
            if (sum < t[i + j]) hi++;
            t[i + j] = sum;
            carry = hi;
        }
        // Propagate carry into higher limbs
        for (int j = 4; i + j < 8; j++) {
            ulong sum = t[i + j] + carry;
            carry = (sum < t[i + j]) ? 1UL : 0UL;
            t[i + j] = sum;
            if (carry == 0) break;
        }
    }

    return mont_reduce(t, m, inv);
}

// Convert to Montgomery form: a * R mod m
inline uint256 to_mont(uint256 a, uint256 r2, uint256 m, ulong inv) {
    return mont_mul(a, r2, m, inv);
}

// Convert from Montgomery form: aR * R^(-1) mod m = a
inline uint256 from_mont(uint256 a, uint256 m, ulong inv) {
    ulong t[8] = {a.limbs[0], a.limbs[1], a.limbs[2], a.limbs[3], 0, 0, 0, 0};
    return mont_reduce(t, m, inv);
}

// Field operations over p (in Montgomery form)
inline uint256 fp_add(uint256 a, uint256 b) {
    ulong carry;
    uint256 r = u256_add(a, b, carry);
    if (carry || u256_cmp(r, SECP256K1_P) >= 0) {
        ulong bw;
        r = u256_sub(r, SECP256K1_P, bw);
    }
    return r;
}

inline uint256 fp_sub(uint256 a, uint256 b) {
    ulong bw;
    uint256 r = u256_sub(a, b, bw);
    if (bw) {
        ulong c;
        r = u256_add(r, SECP256K1_P, c);
    }
    return r;
}

inline uint256 fp_mul(uint256 a, uint256 b) {
    return mont_mul(a, b, SECP256K1_P, P_INV);
}

inline uint256 fp_sqr(uint256 a) {
    return fp_mul(a, a);
}

// Modular operations over n (scalar field, in Montgomery form)
inline uint256 fn_mul(uint256 a, uint256 b) {
    return mont_mul(a, b, SECP256K1_N, N_INV);
}

// Fermat's little theorem: a^(m-2) mod m
// For field p: a^(p-2) mod p
inline uint256 fp_inv(uint256 a) {
    // p - 2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    // Use square-and-multiply with a chain optimized for secp256k1 p
    // We use a simple binary method over the bits of p-2
    uint256 result = to_mont(ONE256, MONT_R2_P, SECP256K1_P, P_INV);
    uint256 base = a;

    // p-2 in limbs (little-endian)
    ulong exp[4] = {
        0xFFFFFFFEFFFFFC2DUL, 0xFFFFFFFFFFFFFFFFUL,
        0xFFFFFFFFFFFFFFFFUL, 0xFFFFFFFFFFFFFFFFUL
    };

    for (int i = 0; i < 4; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((exp[i] >> bit) & 1) {
                result = fp_mul(result, base);
            }
            base = fp_sqr(base);
        }
    }
    return result;
}

// Scalar inversion: a^(n-2) mod n  (for r_inv)
inline uint256 fn_inv(uint256 a) {
    // n-2 in limbs (little-endian)
    ulong exp[4] = {
        0xBFD25E8CD036413FUL, 0xBAAEDCE6AF48A03BUL,
        0xFFFFFFFFFFFFFFFEUL, 0xFFFFFFFFFFFFFFFFUL
    };

    uint256 result = to_mont(ONE256, MONT_R2_N, SECP256K1_N, N_INV);
    uint256 base = a;

    for (int i = 0; i < 4; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((exp[i] >> bit) & 1) {
                result = fn_mul(result, base);
            }
            base = fn_mul(base, base);
        }
    }
    return result;
}

// =============================================================================
// secp256k1 elliptic curve point operations (Jacobian coordinates)
// All coordinates in Montgomery form over F_p
// =============================================================================

struct ECPoint {
    uint256 x, y, z; // Jacobian: (X, Y, Z), affine = (X/Z^2, Y/Z^3)
};

inline ECPoint ec_identity() {
    ECPoint p;
    p.x = to_mont(ONE256, MONT_R2_P, SECP256K1_P, P_INV);
    p.y = to_mont(ONE256, MONT_R2_P, SECP256K1_P, P_INV);
    p.z = ZERO256; // Z=0 indicates point at infinity
    return p;
}

inline bool ec_is_infinity(ECPoint p) {
    return u256_is_zero(p.z);
}

// Point doubling in Jacobian coordinates
// Uses the formula for a = 0 (secp256k1: y^2 = x^3 + 7)
inline ECPoint ec_double(ECPoint p) {
    if (ec_is_infinity(p)) return p;

    // Using optimized formulas for a=0:
    // A = Y^2
    uint256 A = fp_sqr(p.y);
    // B = X * A
    uint256 B = fp_mul(p.x, A);
    // C = A^2
    uint256 C = fp_sqr(A);
    // D = 2 * ((X + A)^2 - B - C)  ... actually just 4*X*Y^2
    // Simpler: S = 4*B
    uint256 S = fp_add(B, B);
    S = fp_add(S, S);
    // M = 3*X^2  (since a=0 for secp256k1)
    uint256 X2 = fp_sqr(p.x);
    uint256 M = fp_add(X2, fp_add(X2, X2));
    // X3 = M^2 - 2*S
    uint256 X3 = fp_sub(fp_sqr(M), fp_add(S, S));
    // Y3 = M * (S - X3) - 8*C
    uint256 C8 = fp_add(C, C); // 2C
    C8 = fp_add(C8, C8);       // 4C
    C8 = fp_add(C8, C8);       // 8C
    uint256 Y3 = fp_sub(fp_mul(M, fp_sub(S, X3)), C8);
    // Z3 = 2*Y*Z
    uint256 Z3 = fp_mul(p.y, p.z);
    Z3 = fp_add(Z3, Z3);

    ECPoint r;
    r.x = X3;
    r.y = Y3;
    r.z = Z3;
    return r;
}

// Point addition (mixed: Q is affine, P is Jacobian)
// P in Jacobian, Q in affine (Qz = 1 implicitly)
inline ECPoint ec_add_mixed(ECPoint P, uint256 Qx, uint256 Qy) {
    if (ec_is_infinity(P)) {
        ECPoint r;
        r.x = Qx;
        r.y = Qy;
        r.z = to_mont(ONE256, MONT_R2_P, SECP256K1_P, P_INV);
        return r;
    }

    // U1 = P.X, U2 = Qx * P.Z^2
    uint256 Z2 = fp_sqr(P.z);
    uint256 U2 = fp_mul(Qx, Z2);
    // S1 = P.Y, S2 = Qy * P.Z^3
    uint256 Z3 = fp_mul(Z2, P.z);
    uint256 S2 = fp_mul(Qy, Z3);

    uint256 H = fp_sub(U2, P.x);
    uint256 R = fp_sub(S2, P.y);

    // If H == 0 and R == 0, points are equal -> double
    if (u256_is_zero(H)) {
        if (u256_is_zero(R)) {
            return ec_double(P);
        }
        // Points are inverses -> return identity
        return ec_identity();
    }

    uint256 H2 = fp_sqr(H);
    uint256 H3 = fp_mul(H, H2);
    uint256 U1H2 = fp_mul(P.x, H2);

    // X3 = R^2 - H^3 - 2*U1*H^2
    uint256 X3 = fp_sub(fp_sub(fp_sqr(R), H3), fp_add(U1H2, U1H2));
    // Y3 = R*(U1*H^2 - X3) - S1*H^3
    uint256 Y3 = fp_sub(fp_mul(R, fp_sub(U1H2, X3)), fp_mul(P.y, H3));
    // Z3 = H * P.Z
    uint256 Zr = fp_mul(H, P.z);

    ECPoint res;
    res.x = X3;
    res.y = Y3;
    res.z = Zr;
    return res;
}

// Convert Jacobian to affine coordinates
inline void ec_to_affine(ECPoint p, thread uint256 &ax, thread uint256 &ay) {
    if (ec_is_infinity(p)) {
        ax = ZERO256;
        ay = ZERO256;
        return;
    }
    uint256 z_inv = fp_inv(p.z);
    uint256 z_inv2 = fp_sqr(z_inv);
    uint256 z_inv3 = fp_mul(z_inv2, z_inv);
    ax = fp_mul(p.x, z_inv2);
    ay = fp_mul(p.y, z_inv3);
}

// Scalar multiplication: k * P (double-and-add, constant time not needed for ecrecover)
// k is a regular (non-Montgomery) 256-bit integer
inline ECPoint ec_mul(uint256 k, ECPoint P) {
    ECPoint result = ec_identity();

    for (int i = 3; i >= 0; i--) {
        for (int bit = 63; bit >= 0; bit--) {
            result = ec_double(result);
            if ((k.limbs[i] >> bit) & 1) {
                // Add P (we need it in affine for mixed add)
                // For simplicity, convert P to affine once and use mixed add
                // But P is already Jacobian... use full Jacobian add instead
                // Actually for the generator table, we pre-convert to affine
                // For general case, use full add:
                if (ec_is_infinity(result)) {
                    result = P;
                } else if (ec_is_infinity(P)) {
                    // nothing
                } else {
                    // Full Jacobian addition
                    uint256 U1 = fp_mul(result.x, fp_sqr(P.z));
                    uint256 U2 = fp_mul(P.x, fp_sqr(result.z));
                    uint256 S1 = fp_mul(result.y, fp_mul(fp_sqr(P.z), P.z));
                    uint256 S2 = fp_mul(P.y, fp_mul(fp_sqr(result.z), result.z));

                    uint256 H = fp_sub(U2, U1);
                    uint256 R = fp_sub(S2, S1);

                    if (u256_is_zero(H)) {
                        if (u256_is_zero(R)) {
                            result = ec_double(result);
                        } else {
                            result = ec_identity();
                        }
                    } else {
                        uint256 H2 = fp_sqr(H);
                        uint256 H3 = fp_mul(H, H2);
                        uint256 U1H2 = fp_mul(U1, H2);

                        uint256 X3 = fp_sub(fp_sub(fp_sqr(R), H3), fp_add(U1H2, U1H2));
                        uint256 Y3 = fp_sub(fp_mul(R, fp_sub(U1H2, X3)), fp_mul(S1, H3));
                        uint256 Z3 = fp_mul(fp_mul(H, result.z), P.z);

                        result.x = X3;
                        result.y = Y3;
                        result.z = Z3;
                    }
                }
            }
        }
    }
    return result;
}

// Scalar multiplication with affine base point (more efficient for generator)
inline ECPoint ec_mul_affine(uint256 k, uint256 Px, uint256 Py) {
    ECPoint result = ec_identity();

    for (int i = 3; i >= 0; i--) {
        for (int bit = 63; bit >= 0; bit--) {
            result = ec_double(result);
            if ((k.limbs[i] >> bit) & 1) {
                result = ec_add_mixed(result, Px, Py);
            }
        }
    }
    return result;
}

// =============================================================================
// Keccak-256 (inline, for address derivation)
// =============================================================================

constant ulong KECCAK_RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL,
    0x800000000000808AUL, 0x8000000080008000UL,
    0x000000000000808BUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008AUL, 0x0000000000000088UL,
    0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL,
    0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800AUL, 0x800000008000000AUL,
    0x8000000080008081UL, 0x8000000000008080UL,
    0x0000000080000001UL, 0x8000000080008008UL,
};

constant int KECCAK_PI_LANE[24] = {
    10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
    15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
};

constant int KECCAK_RHO[24] = {
     1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
};

inline ulong keccak_rotl64(ulong x, int n) {
    return (x << n) | (x >> (64 - n));
}

void keccak_f1600(thread ulong st[25]) {
    for (int round = 0; round < 24; ++round) {
        ulong C[5];
        for (int x = 0; x < 5; ++x)
            C[x] = st[x] ^ st[x + 5] ^ st[x + 10] ^ st[x + 15] ^ st[x + 20];
        for (int x = 0; x < 5; ++x) {
            ulong d = C[(x + 4) % 5] ^ keccak_rotl64(C[(x + 1) % 5], 1);
            for (int y = 0; y < 5; ++y)
                st[x + 5 * y] ^= d;
        }
        ulong t = st[1];
        for (int i = 0; i < 24; ++i) {
            ulong tmp = st[KECCAK_PI_LANE[i]];
            st[KECCAK_PI_LANE[i]] = keccak_rotl64(t, KECCAK_RHO[i]);
            t = tmp;
        }
        for (int y = 0; y < 5; ++y) {
            ulong row[5];
            for (int x = 0; x < 5; ++x) row[x] = st[x + 5 * y];
            for (int x = 0; x < 5; ++x)
                st[x + 5 * y] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
        }
        st[0] ^= KECCAK_RC[round];
    }
}

// Keccak-256 of exactly 64 bytes (uncompressed public key without 0x04 prefix)
inline void keccak256_64(thread const uchar data[64], thread uchar out[32]) {
    ulong state[25] = {};

    // Absorb 64 bytes into first 8 lanes (rate = 136 bytes = 17 lanes)
    for (uint w = 0; w < 8; ++w) {
        ulong lane = 0;
        for (uint b = 0; b < 8; ++b)
            lane |= ulong(data[w * 8 + b]) << (b * 8);
        state[w] ^= lane;
    }

    // Pad: byte 64 gets 0x01, byte 135 gets 0x80
    // Since input is 64 bytes and rate is 136, remaining = 72 bytes of padding
    // padded[0] = 0x01 at position 64 (lane 8, byte 0)
    state[8] ^= 0x01UL;
    // padded[135-64] = 0x80 at position 135 (lane 16, byte 7)
    state[16] ^= 0x80UL << 56;

    keccak_f1600(state);

    // Extract 32 bytes
    for (uint w = 0; w < 4; ++w) {
        ulong lane = state[w];
        for (uint b = 0; b < 8; ++b)
            out[w * 8 + b] = uchar(lane >> (b * 8));
    }
}

// =============================================================================
// Input/Output structures
// =============================================================================

// Per-signature input: packed as (r[32] || s[32] || v[1] || msg_hash[32]) = 97 bytes
// Padded to 128 bytes for alignment
struct EcrecoverInput {
    uchar r[32];        // offset 0
    uchar s[32];        // offset 32
    uchar v;            // offset 64: recovery id (0 or 1)
    uchar _pad[3];      // offset 65: alignment padding
    uchar msg_hash[32]; // offset 68
    uchar _pad2[28];    // pad to 128 bytes total
};

// Per-signature output: 20-byte Ethereum address, padded to 32 bytes
struct EcrecoverOutput {
    uchar address[20]; // offset 0
    uchar valid;       // offset 20: 1 if recovery succeeded, 0 otherwise
    uchar _pad[11];    // pad to 32 bytes
};

// =============================================================================
// Helper: load 32-byte big-endian into uint256 (little-endian limbs)
// =============================================================================

inline uint256 load_be32(thread const uchar bytes[32]) {
    uint256 r;
    for (int limb = 0; limb < 4; limb++) {
        ulong v = 0;
        // Limb 0 = bytes[24..31], limb 3 = bytes[0..7]
        int base = (3 - limb) * 8;
        for (int b = 0; b < 8; b++) {
            v = (v << 8) | ulong(bytes[base + b]);
        }
        r.limbs[limb] = v;
    }
    return r;
}

inline uint256 load_be32_device(device const uchar* bytes) {
    uint256 r;
    for (int limb = 0; limb < 4; limb++) {
        ulong v = 0;
        int base = (3 - limb) * 8;
        for (int b = 0; b < 8; b++) {
            v = (v << 8) | ulong(bytes[base + b]);
        }
        r.limbs[limb] = v;
    }
    return r;
}

// Store uint256 (little-endian limbs) as 32-byte big-endian
inline void store_be32(uint256 val, thread uchar bytes[32]) {
    for (int limb = 0; limb < 4; limb++) {
        int base = (3 - limb) * 8;
        ulong v = val.limbs[limb];
        for (int b = 7; b >= 0; b--) {
            bytes[base + b] = uchar(v & 0xFF);
            v >>= 8;
        }
    }
}

// =============================================================================
// Main kernel: batch secp256k1 ecrecover
// =============================================================================

kernel void secp256k1_ecrecover_batch(
    device const EcrecoverInput*  inputs  [[buffer(0)]],
    device EcrecoverOutput*       outputs [[buffer(1)]],
    uint tid                              [[thread_position_in_grid]])
{
    device const EcrecoverInput& inp = inputs[tid];
    device EcrecoverOutput& out = outputs[tid];

    // Clear output
    for (int i = 0; i < 20; i++) out.address[i] = 0;
    out.valid = 0;
    for (int i = 0; i < 11; i++) out._pad[i] = 0;

    // Load r, s, v, hash from device memory
    uint256 r = load_be32_device(inp.r);
    uint256 s = load_be32_device(inp.s);
    uint256 e = load_be32_device(inp.msg_hash);
    uint v = uint(inp.v);

    // Validate: r and s must be in [1, n-1]
    if (u256_is_zero(r) || u256_cmp(r, SECP256K1_N) >= 0) return;
    if (u256_is_zero(s) || u256_cmp(s, SECP256K1_N) >= 0) return;
    if (v > 1) return;

    // Step 1: Decompress r to point R = (r, y) on secp256k1
    // Compute y^2 = x^3 + 7 mod p
    uint256 r_mont = to_mont(r, MONT_R2_P, SECP256K1_P, P_INV);
    uint256 r2 = fp_sqr(r_mont);
    uint256 r3 = fp_mul(r2, r_mont);
    uint256 seven_mont = to_mont(uint256{{7, 0, 0, 0}}, MONT_R2_P, SECP256K1_P, P_INV);
    uint256 y2 = fp_add(r3, seven_mont);

    // Compute y = sqrt(y2) via Tonelli-Shanks
    // For secp256k1, p ≡ 3 mod 4, so sqrt(a) = a^((p+1)/4)
    // (p+1)/4 = 0x3FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBFFFFF0C
    uint256 y_mont;
    {
        ulong exp[4] = {
            0xFFFFFFFFBFFFFF0CUL, 0xFFFFFFFFFFFFFFFFUL,
            0xFFFFFFFFFFFFFFFFUL, 0x3FFFFFFFFFFFFFFFUL
        };
        uint256 result = to_mont(ONE256, MONT_R2_P, SECP256K1_P, P_INV);
        uint256 base = y2;
        for (int i = 0; i < 4; i++) {
            for (int bit = 0; bit < 64; bit++) {
                if ((exp[i] >> bit) & 1) {
                    result = fp_mul(result, base);
                }
                base = fp_sqr(base);
            }
        }
        y_mont = result;
    }

    // Verify: y^2 == y2 (sqrt exists)
    if (u256_cmp(fp_sqr(y_mont), y2) != 0) return;

    // Select correct y parity based on v
    uint256 y_normal = from_mont(y_mont, SECP256K1_P, P_INV);
    bool y_is_odd = (y_normal.limbs[0] & 1) != 0;
    if ((v == 0 && y_is_odd) || (v == 1 && !y_is_odd)) {
        // Negate y: y = p - y
        y_mont = fp_sub(ZERO256, y_mont);
        // Recalculate -- fp_sub(0, y_mont) needs special handling since 0 in mont form
        // is just 0. So: p_mont - y_mont... actually 0 - y_mont in field = p - y_mont
        // fp_sub handles the borrow correctly, producing p - y in Montgomery form
    }

    // R = (r, y) in Montgomery affine
    uint256 Rx_mont = r_mont;
    uint256 Ry_mont = y_mont;

    // Step 2: Compute r_inv = r^(-1) mod n
    uint256 r_n_mont = to_mont(r, MONT_R2_N, SECP256K1_N, N_INV);
    uint256 r_inv_mont = fn_inv(r_n_mont);

    // Step 3: Compute Q = r^(-1) * (s * R - e * G)
    // In scalar field: u1 = -e * r^(-1) mod n,  u2 = s * r^(-1) mod n
    // Then Q = u1 * G + u2 * R

    uint256 e_n_mont = to_mont(e, MONT_R2_N, SECP256K1_N, N_INV);
    uint256 s_n_mont = to_mont(s, MONT_R2_N, SECP256K1_N, N_INV);

    // u1 = -(e * r_inv) mod n
    uint256 u1_mont = fn_mul(e_n_mont, r_inv_mont);
    // Negate in scalar field: n - u1
    uint256 u1 = from_mont(u1_mont, SECP256K1_N, N_INV);
    if (!u256_is_zero(u1)) {
        ulong bw;
        u1 = u256_sub(SECP256K1_N, u1, bw);
    }

    // u2 = s * r_inv mod n
    uint256 u2 = from_mont(fn_mul(s_n_mont, r_inv_mont), SECP256K1_N, N_INV);

    // Step 4: Multi-scalar multiply Q = u1*G + u2*R
    // Generator G in Montgomery form
    uint256 Gx_mont = to_mont(GX, MONT_R2_P, SECP256K1_P, P_INV);
    uint256 Gy_mont = to_mont(GY, MONT_R2_P, SECP256K1_P, P_INV);

    ECPoint Q1 = ec_mul_affine(u1, Gx_mont, Gy_mont);
    ECPoint Q2 = ec_mul_affine(u2, Rx_mont, Ry_mont);

    // Add Q1 + Q2
    ECPoint Q;
    if (ec_is_infinity(Q1)) {
        Q = Q2;
    } else if (ec_is_infinity(Q2)) {
        Q = Q1;
    } else {
        // Convert Q2 to affine for mixed addition
        uint256 Q2x_aff, Q2y_aff;
        ec_to_affine(Q2, Q2x_aff, Q2y_aff);
        Q = ec_add_mixed(Q1, Q2x_aff, Q2y_aff);
    }

    if (ec_is_infinity(Q)) return;

    // Step 5: Convert Q to affine, serialize as big-endian bytes
    uint256 Qx_aff, Qy_aff;
    ec_to_affine(Q, Qx_aff, Qy_aff);

    // Convert from Montgomery
    uint256 Qx_norm = from_mont(Qx_aff, SECP256K1_P, P_INV);
    uint256 Qy_norm = from_mont(Qy_aff, SECP256K1_P, P_INV);

    // Serialize Q.x || Q.y as 64 bytes big-endian
    uchar pubkey[64];
    store_be32(Qx_norm, pubkey);
    store_be32(Qy_norm, pubkey + 32);

    // Step 6: address = keccak256(pubkey)[12:]
    uchar hash[32];
    keccak256_64(pubkey, hash);

    for (int i = 0; i < 20; i++) {
        out.address[i] = hash[12 + i];
    }
    out.valid = 1;
}
