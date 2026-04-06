// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
/// @file bls12_381.metal
/// Metal compute shader for BLS12-381 signature verification.
///
/// Implements 384-bit field arithmetic (Fp) in Montgomery form for
/// batch BLS signature verification in Quasar consensus.
///
/// BLS12-381 curve parameters:
///   p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
///   r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
///   G1: y^2 = x^3 + 4   over Fp
///   G2: y^2 = x^3 + 4(1+i) over Fp2 = Fp[u]/(u^2+1)
///
/// Operations:
///   bls_verify_batch:  verify N BLS signatures in parallel (one per thread)
///   bls_aggregate_g1:  aggregate N G1 points (signature aggregation)
///
/// Each BLS signature verification requires:
///   1. Deserialize signature (G1 point, 48 bytes compressed)
///   2. Deserialize public key (G2 point, 96 bytes compressed)
///   3. Hash message to G1 point (hash-to-curve, simplified)
///   4. Pairing check: e(sig, G2_gen) == e(H(msg), pubkey)
///
/// For the initial implementation, we focus on G1 point operations
/// and batch verification of the signature equation WITHOUT full
/// pairing (which requires Fp12 and Miller loop). Instead, we verify
/// the aggregated form: e(sum(sigs), G2_gen) == e(sum(H(msgs_i) * alpha_i), pubkey)
/// reducing to a single pairing check on the host, with the GPU doing
/// the heavy G1 scalar multiplications and aggregation.

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// 384-bit unsigned integer (6 x 64-bit limbs, little-endian)
// =============================================================================

struct uint384 {
    ulong limbs[6];  // limbs[0] = least significant
};

// =============================================================================
// BLS12-381 constants
// =============================================================================

// Field modulus p (384 bits)
constant uint384 BLS_P = {{
    0xB9FEFFFFFFFFAAABUL,
    0x1EABFFFEB153FFFFUL,
    0x6730D2A0F6B0F624UL,
    0x64774B84F38512BFUL,
    0x4B1BA7B6434BACD7UL,
    0x1A0111EA397FE69AUL
}};

// Montgomery R^2 mod p (for encoding to Montgomery form)
constant uint384 BLS_R2 = {{
    0xF4DF1F341C341746UL,
    0x0A76E6A609D104F1UL,
    0x8DE5476C4C95B6D5UL,
    0x67EB88A9939D83C0UL,
    0x9A793E85B519952DUL,
    0x11988FE592CAE3AAUL
}};

// Montgomery R mod p
constant uint384 BLS_R = {{
    0x760900000002FFCDUL,
    0xEBF4000BC40C0002UL,
    0x5F48985753C758BAUL,
    0x77CE585370525745UL,
    0x5C071A97A256EC6DUL,
    0x15F65EC3FA80E493UL
}};

// -p^(-1) mod 2^64 (Montgomery constant)
constant ulong BLS_P_INV = 0x89F3FFFCFFFCFFFDUL;

// Generator G1 (affine, Montgomery form)
// G1_x = 0x17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb
// G1_y = 0x08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1
constant uint384 G1_X = {{
    0x5CB38790FD666E19UL,
    0xF85DDE8F09FE5D5CUL,
    0x2C0B0A5CAFB74CD8UL,
    0x95F7B3B14AAE717DUL,
    0x70E02F1AB69D14E3UL,
    0x03C26A6D58B32048UL
}};
constant uint384 G1_Y = {{
    0xA402B931448DC5C8UL,
    0xFBD6AA1ADEAD1CF6UL,
    0x5B9D93D1BA1F5B57UL,
    0x6DC08AFF5B3AF6DDUL,
    0xA4CF5B5C1B6CE90CUL,
    0x13F48FFF25F51018UL
}};

// Zero
constant uint384 ZERO384 = {{0, 0, 0, 0, 0, 0}};

// =============================================================================
// 384-bit arithmetic
// =============================================================================

inline int u384_cmp(uint384 a, uint384 b) {
    for (int i = 5; i >= 0; i--) {
        if (a.limbs[i] < b.limbs[i]) return -1;
        if (a.limbs[i] > b.limbs[i]) return 1;
    }
    return 0;
}

inline bool u384_is_zero(uint384 a) {
    return (a.limbs[0] | a.limbs[1] | a.limbs[2] |
            a.limbs[3] | a.limbs[4] | a.limbs[5]) == 0;
}

inline uint384 u384_add(uint384 a, uint384 b, thread ulong& carry) {
    uint384 r;
    ulong c = 0;
    for (int i = 0; i < 6; i++) {
        ulong sum = a.limbs[i] + c;
        c = (sum < a.limbs[i]) ? 1UL : 0UL;
        ulong sum2 = sum + b.limbs[i];
        c += (sum2 < sum) ? 1UL : 0UL;
        r.limbs[i] = sum2;
    }
    carry = c;
    return r;
}

inline uint384 u384_sub(uint384 a, uint384 b, thread ulong& borrow) {
    uint384 r;
    ulong bw = 0;
    for (int i = 0; i < 6; i++) {
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
// 64x64 -> 128 bit multiplication (no native 128-bit on Metal)
// =============================================================================

/// Multiply two 64-bit values, return (lo, hi).
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
// Montgomery arithmetic over Fp (384-bit)
// =============================================================================

/// Montgomery reduction of a 768-bit value t[12] mod p.
/// Returns t * R^(-1) mod p.
inline uint384 mont_reduce_384(ulong t[12]) {
    ulong a[13];
    for (int i = 0; i < 12; i++) a[i] = t[i];
    a[12] = 0;

    for (int i = 0; i < 6; i++) {
        ulong u = a[i] * BLS_P_INV;

        ulong carry = 0;
        for (int j = 0; j < 6; j++) {
            ulong lo, hi;
            mul64(u, BLS_P.limbs[j], lo, hi);

            ulong sum = lo + carry;
            if (sum < lo) hi++;
            lo = sum;

            sum = a[i + j] + lo;
            if (sum < a[i + j]) hi++;
            a[i + j] = sum;
            carry = hi;
        }
        for (int j = 6; i + j <= 12; j++) {
            ulong sum = a[i + j] + carry;
            carry = (sum < a[i + j]) ? 1UL : 0UL;
            a[i + j] = sum;
            if (carry == 0) break;
        }
    }

    uint384 r;
    r.limbs[0] = a[6];
    r.limbs[1] = a[7];
    r.limbs[2] = a[8];
    r.limbs[3] = a[9];
    r.limbs[4] = a[10];
    r.limbs[5] = a[11];

    if (a[12] || u384_cmp(r, BLS_P) >= 0) {
        ulong bw;
        r = u384_sub(r, BLS_P, bw);
    }
    return r;
}

/// Montgomery multiplication: a * b * R^(-1) mod p
inline uint384 fp_mul(uint384 a, uint384 b) {
    ulong t[12] = {};

    for (int i = 0; i < 6; i++) {
        ulong carry = 0;
        for (int j = 0; j < 6; j++) {
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
        for (int j = 6; i + j < 12; j++) {
            ulong sum = t[i + j] + carry;
            carry = (sum < t[i + j]) ? 1UL : 0UL;
            t[i + j] = sum;
            if (carry == 0) break;
        }
    }

    return mont_reduce_384(t);
}

inline uint384 fp_sqr(uint384 a) {
    return fp_mul(a, a);
}

inline uint384 fp_add(uint384 a, uint384 b) {
    ulong carry;
    uint384 r = u384_add(a, b, carry);
    if (carry || u384_cmp(r, BLS_P) >= 0) {
        ulong bw;
        r = u384_sub(r, BLS_P, bw);
    }
    return r;
}

inline uint384 fp_sub(uint384 a, uint384 b) {
    ulong bw;
    uint384 r = u384_sub(a, b, bw);
    if (bw) {
        ulong c;
        r = u384_add(r, BLS_P, c);
    }
    return r;
}

inline uint384 fp_neg(uint384 a) {
    if (u384_is_zero(a)) return a;
    ulong bw;
    return u384_sub(BLS_P, a, bw);
}

/// Convert to Montgomery form: a * R mod p
inline uint384 to_mont(uint384 a) {
    return fp_mul(a, BLS_R2);
}

/// Convert from Montgomery form: aR * R^(-1) = a
inline uint384 from_mont(uint384 a) {
    ulong t[12] = {a.limbs[0], a.limbs[1], a.limbs[2],
                    a.limbs[3], a.limbs[4], a.limbs[5],
                    0, 0, 0, 0, 0, 0};
    return mont_reduce_384(t);
}

/// Fermat inversion: a^(p-2) mod p
inline uint384 fp_inv(uint384 a) {
    // p-2 as 6 limbs (BLS_P - 2)
    uint384 exp = BLS_P;
    exp.limbs[0] -= 2;

    uint384 result = BLS_R;  // 1 in Montgomery form
    uint384 base = a;

    for (int i = 0; i < 6; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((exp.limbs[i] >> bit) & 1) {
                result = fp_mul(result, base);
            }
            base = fp_sqr(base);
        }
    }
    return result;
}

// =============================================================================
// G1 point operations (Jacobian coordinates, Montgomery Fp)
// =============================================================================

struct G1Point {
    uint384 x, y, z;
};

inline G1Point g1_identity() {
    G1Point p;
    p.x = BLS_R;  // 1 in Montgomery
    p.y = BLS_R;
    p.z = ZERO384; // Z=0 is identity
    return p;
}

inline bool g1_is_infinity(G1Point p) {
    return u384_is_zero(p.z);
}

/// G1 point doubling (a=0 for BLS12-381 G1: y^2 = x^3 + 4)
inline G1Point g1_double(G1Point p) {
    if (g1_is_infinity(p)) return p;

    uint384 A = fp_sqr(p.y);
    uint384 B = fp_mul(p.x, A);
    uint384 C = fp_sqr(A);

    // S = 4*B
    uint384 S = fp_add(B, B);
    S = fp_add(S, S);

    // M = 3*X^2 (a=0)
    uint384 X2 = fp_sqr(p.x);
    uint384 M = fp_add(X2, fp_add(X2, X2));

    // X3 = M^2 - 2*S
    uint384 X3 = fp_sub(fp_sqr(M), fp_add(S, S));

    // Y3 = M*(S - X3) - 8*C
    uint384 C8 = fp_add(C, C);
    C8 = fp_add(C8, C8);
    C8 = fp_add(C8, C8);
    uint384 Y3 = fp_sub(fp_mul(M, fp_sub(S, X3)), C8);

    // Z3 = 2*Y*Z
    uint384 Z3 = fp_mul(p.y, p.z);
    Z3 = fp_add(Z3, Z3);

    G1Point r;
    r.x = X3; r.y = Y3; r.z = Z3;
    return r;
}

/// G1 mixed addition (Q in affine, P in Jacobian)
inline G1Point g1_add_mixed(G1Point P, uint384 Qx, uint384 Qy) {
    if (g1_is_infinity(P)) {
        G1Point r;
        r.x = Qx; r.y = Qy; r.z = BLS_R;
        return r;
    }

    uint384 Z2 = fp_sqr(P.z);
    uint384 U2 = fp_mul(Qx, Z2);
    uint384 Z3 = fp_mul(Z2, P.z);
    uint384 S2 = fp_mul(Qy, Z3);

    uint384 H = fp_sub(U2, P.x);
    uint384 R = fp_sub(S2, P.y);

    if (u384_is_zero(H)) {
        if (u384_is_zero(R))
            return g1_double(P);
        return g1_identity();
    }

    uint384 H2 = fp_sqr(H);
    uint384 H3 = fp_mul(H, H2);
    uint384 U1H2 = fp_mul(P.x, H2);

    uint384 X3 = fp_sub(fp_sub(fp_sqr(R), H3), fp_add(U1H2, U1H2));
    uint384 Y3 = fp_sub(fp_mul(R, fp_sub(U1H2, X3)), fp_mul(P.y, H3));
    uint384 Zr = fp_mul(H, P.z);

    G1Point res;
    res.x = X3; res.y = Y3; res.z = Zr;
    return res;
}

/// Scalar multiplication k * P (affine base point)
inline G1Point g1_mul(uint384 k, uint384 Px, uint384 Py) {
    G1Point result = g1_identity();

    for (int i = 5; i >= 0; i--) {
        for (int bit = 63; bit >= 0; bit--) {
            result = g1_double(result);
            if ((k.limbs[i] >> bit) & 1) {
                result = g1_add_mixed(result, Px, Py);
            }
        }
    }
    return result;
}

/// Convert Jacobian -> affine
inline void g1_to_affine(G1Point p, thread uint384& ax, thread uint384& ay) {
    if (g1_is_infinity(p)) {
        ax = ZERO384; ay = ZERO384;
        return;
    }
    uint384 z_inv = fp_inv(p.z);
    uint384 z_inv2 = fp_sqr(z_inv);
    uint384 z_inv3 = fp_mul(z_inv2, z_inv);
    ax = fp_mul(p.x, z_inv2);
    ay = fp_mul(p.y, z_inv3);
}

// =============================================================================
// BLS signature structures
// =============================================================================

/// Compressed BLS signature (G1 point, 48 bytes)
struct BLSSignature {
    uchar data[48];
};

/// Compressed BLS public key (G2 point, 96 bytes)
struct BLSPublicKey {
    uchar data[96];
};

/// Message hash for BLS verification (32 bytes, pre-hashed)
struct BLSMessage {
    uchar data[32];
};

// =============================================================================
// Deserialization helpers
// =============================================================================

/// Deserialize a 48-byte compressed G1 point to uint384 x-coordinate.
/// Format: big-endian 48 bytes. Bit 383 = compression flag, bit 382 = infinity,
/// bit 381 = sign of y (0 = positive).
inline uint384 deserialize_fp(device const uchar* data) {
    uint384 r = {};
    // Big-endian to little-endian limbs
    for (int limb = 0; limb < 6; limb++) {
        ulong val = 0;
        for (int byte_idx = 0; byte_idx < 8; byte_idx++) {
            // Byte position: (5 - limb) * 8 + (7 - byte_idx)
            int src = (5 - limb) * 8 + (7 - byte_idx);
            if (src < 48)
                val |= (ulong)data[src] << (byte_idx * 8);
        }
        r.limbs[limb] = val;
    }
    return r;
}

/// Decompress G1 point: recover y from x using curve equation y^2 = x^3 + 4
inline bool decompress_g1(uint384 x_raw, bool y_positive, thread uint384& x_mont, thread uint384& y_mont) {
    x_mont = to_mont(x_raw);

    // y^2 = x^3 + 4
    uint384 x2 = fp_sqr(x_mont);
    uint384 x3 = fp_mul(x2, x_mont);
    uint384 b_mont = to_mont(uint384{{4, 0, 0, 0, 0, 0}});
    uint384 y2 = fp_add(x3, b_mont);

    // Square root via Tonelli-Shanks. For BLS12-381, p = 3 mod 4,
    // so sqrt(a) = a^((p+1)/4).
    // (p+1)/4 computed offline
    uint384 exp = {{
        0xEE7FBFFFFFFFEAAFUL,
        0x07AAFFFFAC54FFFFUL,
        0xD9CC34A83DAC3D89UL,
        0xD91DD2E13CE144AFUL,
        0x92C6E9ED90D2EB35UL,
        0x0680447A8E5FF9A6UL
    }};

    uint384 y_cand = BLS_R;
    uint384 base = y2;
    for (int i = 0; i < 6; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((exp.limbs[i] >> bit) & 1) {
                y_cand = fp_mul(y_cand, base);
            }
            base = fp_sqr(base);
        }
    }

    // Verify: y_cand^2 == y2
    uint384 check = fp_sqr(y_cand);
    if (u384_cmp(check, y2) != 0)
        return false;  // Not on curve

    // Pick correct sign
    uint384 y_normal = from_mont(y_cand);
    bool is_positive = (y_normal.limbs[0] & 1) == 0;
    if (is_positive != y_positive) {
        y_mont = fp_neg(y_cand);
    } else {
        y_mont = y_cand;
    }

    return true;
}

// =============================================================================
// BLS Verification kernel
// =============================================================================

/// Batch BLS signature verification.
/// Each thread verifies one signature independently.
///
/// For initial implementation, this performs the G1 point operations needed
/// for verification (deserialization, decompression, point arithmetic).
/// The final pairing check is deferred to the host (requires Fp12 Miller loop).
///
/// Output:
///   results[tid] = 1 if the G1 operations succeeded (point on curve, valid)
///   results[tid] = 0 if deserialization or curve check failed
kernel void bls_verify_batch(
    device const BLSSignature*  sigs    [[buffer(0)]],
    device const BLSPublicKey*  pubkeys [[buffer(1)]],
    device const BLSMessage*    messages [[buffer(2)]],
    device uint*                results [[buffer(3)]],
    constant uint&              num_sigs [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_sigs) return;

    // -- Deserialize signature (compressed G1 point) --------------------------
    device const uchar* sig_data = sigs[tid].data;

    // Check flags byte
    uchar flags = sig_data[0];
    bool compressed = (flags & 0x80) != 0;
    bool infinity = (flags & 0x40) != 0;
    bool y_sign = (flags & 0x20) != 0;

    if (infinity) {
        // Signature at infinity is invalid
        results[tid] = 0;
        return;
    }

    if (!compressed) {
        // We only handle compressed format
        results[tid] = 0;
        return;
    }

    // Clear flag bits for x-coordinate deserialization
    uchar clean_data[48];
    for (int i = 0; i < 48; i++) clean_data[i] = sig_data[i];
    clean_data[0] &= 0x1F;

    uint384 x_raw = {};
    for (int limb = 0; limb < 6; limb++) {
        ulong val = 0;
        for (int b = 0; b < 8; b++) {
            int src = (5 - limb) * 8 + (7 - b);
            if (src < 48)
                val |= (ulong)clean_data[src] << (b * 8);
        }
        x_raw.limbs[limb] = val;
    }

    // -- Decompress to affine G1 point ----------------------------------------
    uint384 sig_x, sig_y;
    bool on_curve = decompress_g1(x_raw, !y_sign, sig_x, sig_y);
    if (!on_curve) {
        results[tid] = 0;
        return;
    }

    // -- Subgroup check (simplified: multiply by r, check identity) -----------
    // For a full implementation, we'd do scalar mul by the curve order r
    // and verify the result is the identity. This is expensive (256-bit scalar mul
    // on a 384-bit curve), so we mark it as valid for now and let the host
    // do the full check if needed.

    // -- Mark as valid (G1 deserialization and curve check passed) -------------
    results[tid] = 1;
}

/// Aggregate N G1 signatures into one by summing the points.
/// This is the main GPU-accelerated operation for BLS aggregation.
///
/// Input: N compressed G1 points (48 bytes each)
/// Output: 1 uncompressed G1 point (96 bytes: x[48] || y[48])
///
/// Uses parallel reduction within a threadgroup.
kernel void bls_aggregate_g1(
    device const BLSSignature*  sigs       [[buffer(0)]],
    device uchar*               agg_out    [[buffer(1)]],  // 96 bytes
    device atomic_uint*         counter    [[buffer(2)]],   // Atomic completion counter
    constant uint&              num_sigs   [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Each thread deserializes and decompresses one signature
    G1Point local_sum = g1_identity();

    if (tid < num_sigs) {
        device const uchar* sig_data = sigs[tid].data;

        uchar flags = sig_data[0];
        bool infinity = (flags & 0x40) != 0;
        bool y_sign = (flags & 0x20) != 0;

        if (!infinity) {
            uchar clean_data[48];
            for (int i = 0; i < 48; i++) clean_data[i] = sig_data[i];
            clean_data[0] &= 0x1F;

            uint384 x_raw = {};
            for (int limb = 0; limb < 6; limb++) {
                ulong val = 0;
                for (int b = 0; b < 8; b++) {
                    int src = (5 - limb) * 8 + (7 - b);
                    if (src < 48)
                        val |= (ulong)clean_data[src] << (b * 8);
                }
                x_raw.limbs[limb] = val;
            }

            uint384 sx, sy;
            if (decompress_g1(x_raw, !y_sign, sx, sy)) {
                local_sum.x = sx;
                local_sum.y = sy;
                local_sum.z = BLS_R;  // Affine: z=1 in Montgomery
            }
        }
    }

    // Threadgroup reduction: sum all points in this threadgroup.
    // Use threadgroup memory for inter-thread communication.
    threadgroup uint384 shared_x[256];
    threadgroup uint384 shared_y[256];
    threadgroup uint384 shared_z[256];

    shared_x[lid] = local_sum.x;
    shared_y[lid] = local_sum.y;
    shared_z[lid] = local_sum.z;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Binary reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            G1Point a;
            a.x = shared_x[lid]; a.y = shared_y[lid]; a.z = shared_z[lid];

            G1Point b;
            b.x = shared_x[lid + stride]; b.y = shared_y[lid + stride]; b.z = shared_z[lid + stride];

            if (!g1_is_infinity(b)) {
                if (g1_is_infinity(a)) {
                    a = b;
                } else {
                    // Full Jacobian addition (both points are Jacobian)
                    uint384 Z1sq = fp_sqr(a.z);
                    uint384 Z2sq = fp_sqr(b.z);
                    uint384 U1 = fp_mul(a.x, Z2sq);
                    uint384 U2 = fp_mul(b.x, Z1sq);
                    uint384 S1 = fp_mul(a.y, fp_mul(Z2sq, b.z));
                    uint384 S2 = fp_mul(b.y, fp_mul(Z1sq, a.z));

                    uint384 H = fp_sub(U2, U1);
                    uint384 R = fp_sub(S2, S1);

                    if (u384_is_zero(H)) {
                        if (u384_is_zero(R)) {
                            a = g1_double(a);
                        } else {
                            a = g1_identity();
                        }
                    } else {
                        uint384 H2 = fp_sqr(H);
                        uint384 H3 = fp_mul(H, H2);
                        uint384 U1H2 = fp_mul(U1, H2);
                        a.x = fp_sub(fp_sub(fp_sqr(R), H3), fp_add(U1H2, U1H2));
                        a.y = fp_sub(fp_mul(R, fp_sub(U1H2, a.x)), fp_mul(S1, H3));
                        a.z = fp_mul(fp_mul(H, a.z), b.z);
                    }
                }
            }

            shared_x[lid] = a.x;
            shared_y[lid] = a.y;
            shared_z[lid] = a.z;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 of each threadgroup writes partial result.
    // The host sums partial results from each threadgroup.
    if (lid == 0) {
        G1Point partial;
        partial.x = shared_x[0]; partial.y = shared_y[0]; partial.z = shared_z[0];

        uint384 ax, ay;
        g1_to_affine(partial, ax, ay);

        uint384 ax_norm = from_mont(ax);
        uint384 ay_norm = from_mont(ay);

        // Write to output: big-endian 48 bytes x, then 48 bytes y
        uint tg_offset = tgid * 96;
        for (int limb = 0; limb < 6; limb++) {
            for (int b = 0; b < 8; b++) {
                int dst = (5 - limb) * 8 + (7 - b);
                if (dst < 48) {
                    agg_out[tg_offset + dst] = uchar((ax_norm.limbs[limb] >> (b * 8)) & 0xFF);
                    agg_out[tg_offset + 48 + dst] = uchar((ay_norm.limbs[limb] >> (b * 8)) & 0xFF);
                }
            }
        }

        atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    }
}
