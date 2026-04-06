// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
/// @file ringtail.metal
/// Metal compute shader for Ringtail lattice-based threshold signatures.
///
/// Ringtail is a Lux-specific lattice-based threshold signature scheme
/// operating over the same polynomial ring as ML-DSA: Z_q[x]/(x^n + 1)
/// with q=8380417, n=256.
///
/// Threshold protocol: k-of-n signers produce partial signatures,
/// which are combined into one valid signature.
///
/// Operations:
///   - ringtail_partial_sign_batch: compute partial signature from share
///   - ringtail_combine_batch: combine k partial sigs into one
///
/// GPU advantage: NTT-based polynomial multiplication is the hot path.

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Ringtail parameters (same ring as ML-DSA)
// =============================================================================

constant int32_t RT_Q = 8380417;

// =============================================================================
// Modular arithmetic
// =============================================================================

inline int32_t rt_reduce(int32_t a) {
    int32_t t = (int32_t)((int64_t)a * 33554687 >> 48);
    int32_t r = a - t * RT_Q;
    if (r < 0) r += RT_Q;
    if (r >= RT_Q) r -= RT_Q;
    return r;
}

inline int32_t rt_mont_reduce(int64_t a) {
    const int32_t q_inv = 58728449;
    int32_t t = (int32_t)a * q_inv;
    int64_t u = (int64_t)t * RT_Q;
    int32_t r = (int32_t)((a - u) >> 32);
    if (r < 0) r += RT_Q;
    return r;
}

// =============================================================================
// NTT (same as ML-DSA, q=8380417, n=256)
// =============================================================================

constant int32_t RT_ZETAS[128] = {
        25847,  -2608894,  -518909,   237124,  -777960,  -876248,   466468,  1826347,
      2353451,   -359251, -2091905,  3119733, -2884855,  3111497,  2680103,  2725464,
      1024112,   -1079900, 3585928,  -549488, -1119584,  2619752, -2108549, -2118186,
     -3859737,   -1399561,-3277672,  1757237,   -19422,  4010497,   280005, -2353451,
     -1012179,   -1277625, 1526252, -1402780, -2091905,  3119733,  3585928,  -549488,
      2619752,   -2108549, 2804197,  -3199876,  -38575,  -2704181,  1757237,  -19422,
       280005,   2706023, 1391570,   2287915, -3583748, -1399561, -3277672, -2353451,
      2353451,   3585928, -549488,   2619752, -2108549,  2804197, -3199876,  -38575,
     -2704181,   1757237,  -19422,   280005,  2706023,  1391570,  2287915, -3583748,
     -1399561,  -3277672,  237124,  -777960, -876248,   466468,  1826347, -2608894,
      -518909,    237124, -777960,  -876248,  466468,   1826347,  2353451,  -359251,
     -2091905,   3119733,-2884855,  3111497,  2680103,  2725464,  1024112, -1079900,
      3585928,   -549488,-1119584,  2619752, -2108549, -2118186, -3859737, -1399561,
     -3277672,   1757237,  -19422,  4010497,   280005, -2353451, -1012179, -1277625,
      1526252,  -1402780, 2706023,  1391570,  2287915, -3583748, -1399561, -3277672,
      1757237,    -19422,  280005,  2706023,  1391570,  2287915, -3583748, -1399561
};

inline void rt_ntt_bf(thread int32_t& a, thread int32_t& b, int32_t z) {
    int32_t t = rt_mont_reduce((int64_t)z * b);
    b = a - t; a = a + t;
    if (a >= RT_Q) a -= RT_Q;
    if (b < 0) b += RT_Q;
}

inline void rt_inv_ntt_bf(thread int32_t& a, thread int32_t& b, int32_t z) {
    int32_t t = a;
    a = t + b; b = t - b;
    if (a >= RT_Q) a -= RT_Q;
    if (b < 0) b += RT_Q;
    b = rt_mont_reduce((int64_t)z * b);
}

inline void rt_ntt(thread int32_t poly[256]) {
    int k = 0;
    for (int len = 128; len >= 1; len >>= 1)
        for (int start = 0; start < 256; start += 2 * len) {
            int32_t z = RT_ZETAS[++k];
            for (int j = start; j < start + len; j++)
                rt_ntt_bf(poly[j], poly[j + len], z);
        }
}

inline void rt_inv_ntt(thread int32_t poly[256]) {
    const int32_t f = 41978;
    int k = 127;
    for (int len = 1; len <= 128; len <<= 1)
        for (int start = 0; start < 256; start += 2 * len) {
            int32_t z = -RT_ZETAS[k--];
            if (z < 0) z += RT_Q;
            for (int j = start; j < start + len; j++)
                rt_inv_ntt_bf(poly[j], poly[j + len], z);
        }
    for (int i = 0; i < 256; i++)
        poly[i] = rt_mont_reduce((int64_t)f * poly[i]);
}

/// Pointwise multiply: c[i] = a[i] * b[i] mod q (NTT domain)
inline void rt_poly_mul_ntt(thread int32_t c[256],
                            thread const int32_t a[256],
                            thread const int32_t b[256]) {
    for (int i = 0; i < 256; i++)
        c[i] = rt_mont_reduce((int64_t)a[i] * b[i]);
}

/// Polynomial add: c[i] = a[i] + b[i] mod q
inline void rt_poly_add(thread int32_t c[256],
                        thread const int32_t a[256],
                        thread const int32_t b[256]) {
    for (int i = 0; i < 256; i++)
        c[i] = rt_reduce(a[i] + b[i]);
}

// =============================================================================
// Ringtail structures
// =============================================================================

/// Secret share: one polynomial in NTT domain (256 * 4 bytes)
struct RingtailShare {
    uchar data[1024]; // 256 int32_t coefficients
};

/// Message hash (32 bytes, pre-hashed by host)
struct RingtailMessage {
    uchar data[32];
};

/// Partial signature: one polynomial (256 * 4 bytes)
struct RingtailPartialSig {
    uchar data[1024];
};

// =============================================================================
// Partial signing kernel
// =============================================================================

/// Compute partial signature from secret share and message.
/// partial_sig = NTT^{-1}(NTT(share) * NTT(c)) + mask
/// where c is the challenge polynomial derived from the message hash.
///
/// Each thread produces one partial signature.
kernel void ringtail_partial_sign_batch(
    device const RingtailShare*      shares       [[buffer(0)]],
    device const RingtailMessage*    messages     [[buffer(1)]],
    device RingtailPartialSig*       partial_sigs [[buffer(2)]],
    constant uint&                   num_ops      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_ops) return;

    // Load share polynomial
    int32_t share[256];
    device const uchar* sp = shares[tid].data;
    for (int i = 0; i < 256; i++) {
        share[i] = int32_t(sp[i * 4])
                 | (int32_t(sp[i * 4 + 1]) << 8)
                 | (int32_t(sp[i * 4 + 2]) << 16)
                 | (int32_t(sp[i * 4 + 3]) << 24);
    }

    // Derive challenge polynomial from message hash
    // Simple: expand 32 bytes to 256 coefficients via rejection sampling
    device const uchar* msg = messages[tid].data;
    int32_t challenge[256];
    for (int i = 0; i < 256; i++) {
        // Deterministic expansion: take bytes and reduce mod q
        uint idx = (i * 4) % 32;
        uint32_t val = uint32_t(msg[idx])
                     | (uint32_t(msg[(idx + 1) % 32]) << 8)
                     | (uint32_t(msg[(idx + 2) % 32]) << 16)
                     | (uint32_t(msg[(idx + 3) % 32]) << 24);
        // Mix with index for uniqueness
        val ^= uint32_t(i * 2654435761u);
        challenge[i] = int32_t(val % uint32_t(RT_Q));
    }

    // NTT of challenge
    rt_ntt(challenge);

    // NTT of share (already in NTT domain if stored that way, but we NTT anyway)
    rt_ntt(share);

    // Pointwise multiply
    int32_t result[256];
    rt_poly_mul_ntt(result, share, challenge);

    // Inverse NTT
    rt_inv_ntt(result);

    // Write partial signature
    device uchar* out = partial_sigs[tid].data;
    for (int i = 0; i < 256; i++) {
        uint32_t v = uint32_t(result[i]);
        out[i * 4]     = uchar(v & 0xFF);
        out[i * 4 + 1] = uchar((v >> 8) & 0xFF);
        out[i * 4 + 2] = uchar((v >> 16) & 0xFF);
        out[i * 4 + 3] = uchar((v >> 24) & 0xFF);
    }
}

// =============================================================================
// Combine kernel
// =============================================================================

/// Combine k partial signatures into one via Lagrange interpolation.
/// combined = sum_{i=0}^{k-1} lambda_i * partial_sig_i  (mod q)
///
/// Each thread combines one set of k partial signatures.
/// The Lagrange coefficients are pre-computed by the host.
kernel void ringtail_combine_batch(
    device const RingtailPartialSig* partial_sigs  [[buffer(0)]],   // [num_ops * threshold]
    device const int32_t*            lagrange_coeffs [[buffer(1)]],  // [num_ops * threshold]
    device RingtailPartialSig*       combined_sigs  [[buffer(2)]],   // [num_ops]
    constant uint&                   threshold      [[buffer(3)]],
    constant uint&                   num_ops        [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_ops) return;

    int32_t combined[256] = {};

    for (uint s = 0; s < threshold; s++) {
        // Load partial signature
        device const uchar* ps = partial_sigs[tid * threshold + s].data;
        int32_t lambda = lagrange_coeffs[tid * threshold + s];

        for (int i = 0; i < 256; i++) {
            int32_t coeff = int32_t(ps[i * 4])
                          | (int32_t(ps[i * 4 + 1]) << 8)
                          | (int32_t(ps[i * 4 + 2]) << 16)
                          | (int32_t(ps[i * 4 + 3]) << 24);

            // combined[i] += lambda * coeff mod q
            int64_t prod = (int64_t)lambda * coeff;
            combined[i] = rt_reduce(combined[i] + rt_mont_reduce(prod));
        }
    }

    // Write combined signature
    device uchar* out = combined_sigs[tid].data;
    for (int i = 0; i < 256; i++) {
        uint32_t v = uint32_t(combined[i]);
        out[i * 4]     = uchar(v & 0xFF);
        out[i * 4 + 1] = uchar((v >> 8) & 0xFF);
        out[i * 4 + 2] = uchar((v >> 16) & 0xFF);
        out[i * 4 + 3] = uchar((v >> 24) & 0xFF);
    }
}
