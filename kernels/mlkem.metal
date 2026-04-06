// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
/// @file mlkem.metal
/// Metal compute shader for batch ML-KEM-768 (FIPS 203, CRYSTALS-Kyber)
/// decapsulation.
///
/// ML-KEM operates over polynomial ring Z_q[x]/(x^n + 1) with q=3329, n=256.
/// Decapsulation is the most compute-intensive KEM operation, requiring:
///   1. Decrypt ciphertext to plaintext
///   2. Re-encrypt and compare (implicit rejection)
///
/// The hot path is NTT-based polynomial multiplication over q=3329.
/// Each thread decapsulates one ciphertext independently.

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// ML-KEM-768 parameters (NIST security level 3)
// =============================================================================

constant int16_t MLKEM_Q = 3329;

// =============================================================================
// Montgomery arithmetic for q=3329, R=2^16
// =============================================================================

/// -q^{-1} mod 2^16 = 3327
inline int16_t mlkem_mont_reduce(int32_t a) {
    const int16_t q_inv = 3327;
    int16_t t = (int16_t)a * q_inv;
    int32_t u = (int32_t)t * MLKEM_Q;
    return (int16_t)((a - u) >> 16);
}

/// Barrett reduction for q=3329
inline int16_t mlkem_barrett_reduce(int16_t a) {
    // v = floor(2^26 / q) + 1 = 20159
    int16_t t = (int16_t)(((int32_t)a * 20159) >> 26);
    t = a - t * MLKEM_Q;
    if (t >= MLKEM_Q) t -= MLKEM_Q;
    if (t < 0) t += MLKEM_Q;
    return t;
}

// =============================================================================
// NTT for ML-KEM (q=3329, n=256)
// =============================================================================

// Precomputed zetas in Montgomery form
// zeta = 17 is a primitive 256th root of unity mod 3329
constant int16_t KYBER_ZETAS[128] = {
    2285, 2571, 2970, 1812, 1493, 1422,  287,  202,
    3158,  622, 1577,  182,  962, 2127, 1855, 1468,
     573, 2004,  264,  383, 2500, 1458, 1727, 3199,
    2648, 1017,  732,  608, 1787,  411, 3124, 1758,
    1223,  652, 2777, 1015, 2036, 1491, 3047, 1785,
     516, 3321, 3009, 2663, 1711, 2167,  126, 1469,
    2476, 3239, 3058,  830,  107, 1908, 3082, 2378,
    2931,  961, 1821, 2604,  448, 2264,  677, 2054,
    2226,  430,  555,  843, 2078,  871, 1550,  105,
     422,  587,  177, 3094, 3038, 2869, 1574, 1653,
    3083,  778, 1159, 3182, 2552, 1483, 2727, 1119,
    1739,  644, 2457,  349,  418,  329, 3173, 3254,
     817, 1097,  603,  610, 1322, 2044, 1864,  384,
    2114, 3193, 1218, 1994, 2455,  220, 2142, 1670,
    2144, 1799, 2051,  794, 1819, 2475, 2459,  478,
    3221, 3116,  622, 1097, 2470,  882, 1539, 2392
};

/// Forward NTT butterfly (int32_t intermediates to prevent int16_t overflow)
inline void kyber_ntt_bf(thread int16_t& a, thread int16_t& b, int16_t zeta) {
    int32_t t = (int32_t)b * (int32_t)zeta;
    t = mlkem_mont_reduce(t);
    int32_t sum  = (int32_t)a + t;
    int32_t diff = (int32_t)a - t;
    a = (int16_t)mlkem_barrett_reduce((int16_t)sum);
    b = (int16_t)mlkem_barrett_reduce((int16_t)diff);
}

/// Inverse NTT butterfly
inline void kyber_inv_ntt_bf(thread int16_t& a, thread int16_t& b, int16_t zeta) {
    int16_t t = a;
    a = t + b;
    b = t - b;
    b = mlkem_mont_reduce((int32_t)zeta * b);
}

inline void kyber_ntt(thread int16_t poly[256]) {
    int k = 0;
    for (int len = 128; len >= 2; len >>= 1) {
        for (int start = 0; start < 256; start += 2 * len) {
            int16_t z = KYBER_ZETAS[++k];
            for (int j = start; j < start + len; j++) {
                kyber_ntt_bf(poly[j], poly[j + len], z);
            }
        }
    }
}

inline void kyber_inv_ntt(thread int16_t poly[256]) {
    // f = R * 128^{-1} mod q = 1441 (Montgomery form of 256^{-1})
    const int16_t f = 1441;
    int k = 127;
    for (int len = 2; len <= 128; len <<= 1) {
        for (int start = 0; start < 256; start += 2 * len) {
            int16_t z = KYBER_ZETAS[k--];
            z = MLKEM_Q - z; // negate
            for (int j = start; j < start + len; j++) {
                kyber_inv_ntt_bf(poly[j], poly[j + len], z);
            }
        }
    }
    for (int i = 0; i < 256; i++) {
        poly[i] = mlkem_mont_reduce((int32_t)f * poly[i]);
    }
}

/// Pointwise multiplication of two NTT-domain polynomials (basemul)
inline void kyber_basemul(thread int16_t r[2],
                          thread const int16_t a[2],
                          thread const int16_t b[2],
                          int16_t zeta) {
    r[0] = mlkem_mont_reduce((int32_t)a[1] * b[1]);
    r[0] = mlkem_mont_reduce((int32_t)r[0] * zeta);
    r[0] = r[0] + mlkem_mont_reduce((int32_t)a[0] * b[0]);
    r[1] = mlkem_mont_reduce((int32_t)a[0] * b[1]);
    r[1] = r[1] + mlkem_mont_reduce((int32_t)a[1] * b[0]);
}

/// Full pointwise multiplication of NTT polynomials
inline void kyber_poly_pointwise(thread int16_t r[256],
                                  thread const int16_t a[256],
                                  thread const int16_t b[256]) {
    for (int i = 0; i < 256 / 4; i++) {
        int16_t a_pair[2] = {a[4 * i], a[4 * i + 1]};
        int16_t b_pair[2] = {b[4 * i], b[4 * i + 1]};
        int16_t r_pair[2];
        kyber_basemul(r_pair, a_pair, b_pair, KYBER_ZETAS[64 + i]);
        r[4 * i]     = r_pair[0];
        r[4 * i + 1] = r_pair[1];

        int16_t a_pair2[2] = {a[4 * i + 2], a[4 * i + 3]};
        int16_t b_pair2[2] = {b[4 * i + 2], b[4 * i + 3]};
        int16_t r_pair2[2];
        kyber_basemul(r_pair2, a_pair2, b_pair2, -KYBER_ZETAS[64 + i]);
        r[4 * i + 2] = r_pair2[0];
        r[4 * i + 3] = r_pair2[1];
    }
}

// =============================================================================
// ML-KEM structures
// =============================================================================

/// ML-KEM-768 secret key (decapsulation key)
/// dk = s_hat[k*384] || ek[1184] || h[32] || z[32]
/// where s_hat is the NTT of the secret vector, ek is the encapsulation key
struct MLKEMSecretKey {
    uchar data[2400]; // 3*384 + 1184 + 32 + 32
};

/// ML-KEM-768 ciphertext
/// ct = c1[k*320] || c2[128]
struct MLKEMCiphertext {
    uchar data[1088]; // 3*320 + 128
};

/// Shared secret output (32 bytes)
struct MLKEMSharedSecret {
    uchar data[32];
};

// =============================================================================
// Decapsulation kernel
// =============================================================================

/// Batch ML-KEM-768 decapsulation.
/// Each thread decapsulates one ciphertext to recover the shared secret.
///
/// The GPU handles the NTT-based polynomial arithmetic (decrypt step).
/// The full decapsulation includes SHA3/SHAKE hashing done on the host.
///
/// Output: shared_secrets[tid] = decrypted message polynomial (32 bytes).
kernel void mlkem_decapsulate_batch(
    device const MLKEMSecretKey*    secret_keys    [[buffer(0)]],
    device const MLKEMCiphertext*   ciphertexts    [[buffer(1)]],
    device MLKEMSharedSecret*       shared_secrets [[buffer(2)]],
    constant uint&                  num_ops        [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_ops) return;

    device const uchar* sk = secret_keys[tid].data;
    device const uchar* ct = ciphertexts[tid].data;

    // -- Decode secret key s_hat (NTT domain, k=3 polynomials) --
    int16_t s_hat[3][256];
    for (int p = 0; p < 3; p++) {
        device const uchar* sp = sk + p * 384;
        for (int i = 0; i < 256; i++) {
            uint idx = i * 3 / 2;
            if (i & 1) {
                s_hat[p][i] = (int16_t)(((sp[idx] >> 4) | ((uint)sp[idx + 1] << 4)) & 0xFFF);
            } else {
                s_hat[p][i] = (int16_t)((sp[idx] | ((uint)sp[idx + 1] << 8)) & 0xFFF);
            }
        }
    }

    // -- Decode ciphertext u (compressed, k=3 polynomials, 10 bits each) --
    int16_t u[3][256];
    for (int p = 0; p < 3; p++) {
        device const uchar* up = ct + p * 320;
        for (int i = 0; i < 256; i += 4) {
            uint idx = (i / 4) * 5;
            uint b0 = up[idx], b1 = up[idx+1], b2 = up[idx+2];
            uint b3 = up[idx+3], b4 = up[idx+4];

            u[p][i]   = (int16_t)(b0 | ((b1 & 0x03) << 8));
            u[p][i+1] = (int16_t)((b1 >> 2) | ((b2 & 0x0F) << 6));
            u[p][i+2] = (int16_t)((b2 >> 4) | ((b3 & 0x3F) << 4));
            u[p][i+3] = (int16_t)((b3 >> 6) | (b4 << 2));
        }
        // Decompress: multiply by q/2^10 and round
        for (int i = 0; i < 256; i++) {
            u[p][i] = (int16_t)(((uint32_t)u[p][i] * MLKEM_Q + 512) >> 10);
        }
    }

    // -- Decode ciphertext v (compressed, 4 bits per coefficient) --
    int16_t v[256];
    device const uchar* vp = ct + 3 * 320;
    for (int i = 0; i < 256; i += 2) {
        v[i]     = (int16_t)(vp[i / 2] & 0x0F);
        v[i + 1] = (int16_t)(vp[i / 2] >> 4);
    }
    // Decompress: multiply by q/2^4
    for (int i = 0; i < 256; i++) {
        v[i] = (int16_t)(((uint32_t)v[i] * MLKEM_Q + 8) >> 4);
    }

    // -- Compute NTT(u) for inner product --
    int16_t u_hat[3][256];
    for (int p = 0; p < 3; p++) {
        for (int i = 0; i < 256; i++) u_hat[p][i] = u[p][i];
        kyber_ntt(u_hat[p]);
    }

    // -- Compute s_hat^T * NTT(u) --
    int16_t mp[256] = {};
    for (int p = 0; p < 3; p++) {
        int16_t tmp[256];
        kyber_poly_pointwise(tmp, s_hat[p], u_hat[p]);
        for (int i = 0; i < 256; i++) {
            mp[i] = mlkem_barrett_reduce(mp[i] + tmp[i]);
        }
    }

    // -- INTT to get s^T * u in normal domain --
    kyber_inv_ntt(mp);

    // -- Compute m = v - s^T * u, then compress to bits --
    // m_i = round((v_i - mp_i) * 2 / q) mod 2
    device uchar* out = shared_secrets[tid].data;
    for (int i = 0; i < 32; i++) out[i] = 0;

    for (int i = 0; i < 256; i++) {
        int16_t diff = v[i] - mp[i];
        if (diff < 0) diff += MLKEM_Q;
        // Compress to 1 bit: round(2*diff/q) mod 2
        uint16_t t = ((uint16_t)diff << 1) + MLKEM_Q / 2;
        uint8_t bit = (uint8_t)((t / MLKEM_Q) & 1);
        out[i / 8] |= bit << (i % 8);
    }
}
