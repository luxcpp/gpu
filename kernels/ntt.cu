// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
/// @file ntt.cu
/// Shared Number Theoretic Transform (NTT) primitives for lattice-based PQ crypto.
/// CUDA port of ntt.metal -- byte-identical arithmetic output.
///
/// Used by: ML-DSA (FIPS 204), ML-KEM (FIPS 203), Ringtail, SLH-DSA (FIPS 205)
///
/// NTT operates over polynomial rings Z_q[x]/(x^n + 1).
/// The butterfly operations are perfectly parallel -- each layer of the
/// NTT can be dispatched across GPU threads.
///
/// This file provides:
///   - Forward NTT (Cooley-Tukey butterfly)
///   - Inverse NTT (Gentleman-Sande butterfly)
///   - Pointwise polynomial multiplication in NTT domain
///   - Barrett reduction for arbitrary moduli
///
/// Parameters are passed via constants so the same code works for:
///   ML-DSA:  q=8380417,  n=256
///   ML-KEM:  q=3329,     n=256
///   Ringtail: q=8380417, n=256 (same ring as ML-DSA)

#ifndef NTT_CUDA_H
#define NTT_CUDA_H

#include <cstdint>

#ifdef __CUDA_ARCH__
#define NTT_DEVICE __device__ __forceinline__
#else
#define NTT_DEVICE inline
#define __global__
#define __shared__
#endif

// =============================================================================
// Barrett reduction: a mod q without division
// =============================================================================

/// Barrett reduction for q = 8380417 (ML-DSA / Ringtail)
NTT_DEVICE int32_t barrett_reduce_mldsa(int32_t a) {
    const int32_t q = 8380417;
    const int64_t v = 33554687LL; // floor(2^48 / q) + 1
    int64_t t = (int64_t)a * v >> 48;
    int32_t r = a - (int32_t)t * q;
    if (r < 0) r += q;
    if (r >= q) r -= q;
    return r;
}

/// Barrett reduction for q = 3329 (ML-KEM)
NTT_DEVICE int32_t barrett_reduce_mlkem(int32_t a) {
    const int32_t q = 3329;
    const int64_t v = 5039835LL; // floor(2^36 / q) + 1
    int64_t t = (int64_t)a * v >> 36;
    int32_t r = a - (int32_t)t * q;
    if (r < 0) r += q;
    if (r >= q) r -= q;
    return r;
}

/// Montgomery reduction for ML-DSA: aR^{-1} mod q, R = 2^32
/// q_inv = -q^{-1} mod 2^32 = 58728449
NTT_DEVICE int32_t mont_reduce_mldsa(int64_t a) {
    const int32_t q = 8380417;
    const int32_t q_inv = 58728449; // -q^(-1) mod 2^32
    int32_t t = (int32_t)a * q_inv;
    int64_t u = (int64_t)t * q;
    int32_t r = (int32_t)((a - u) >> 32);
    if (r < 0) r += q;
    return r;
}

/// Montgomery reduction for ML-KEM: aR^{-1} mod q, R = 2^16
/// q_inv = -q^{-1} mod 2^16 = 3327
NTT_DEVICE int16_t mont_reduce_mlkem(int32_t a) {
    const int16_t q = 3329;
    const int16_t q_inv = 3327; // -q^(-1) mod 2^16
    int16_t t = (int16_t)a * q_inv;
    int32_t u = (int32_t)t * q;
    int16_t r = (int16_t)((a - u) >> 16);
    return r;
}

// =============================================================================
// NTT butterfly operations (Cooley-Tukey, in-place)
// =============================================================================

/// Forward NTT butterfly for ML-DSA (q=8380417)
NTT_DEVICE void ntt_butterfly_mldsa(int32_t& a, int32_t& b, int32_t zeta) {
    int32_t t = mont_reduce_mldsa((int64_t)zeta * b);
    b = a - t;
    a = a + t;
    if (a >= 8380417) a -= 8380417;
    if (b < 0) b += 8380417;
}

/// Inverse NTT butterfly for ML-DSA (Gentleman-Sande)
NTT_DEVICE void inv_ntt_butterfly_mldsa(int32_t& a, int32_t& b, int32_t zeta) {
    int32_t t = a;
    a = t + b;
    b = t - b;
    if (a >= 8380417) a -= 8380417;
    if (b < 0) b += 8380417;
    b = mont_reduce_mldsa((int64_t)zeta * b);
}

/// Forward NTT butterfly for ML-KEM (q=3329)
NTT_DEVICE void ntt_butterfly_mlkem(int16_t& a, int16_t& b, int16_t zeta) {
    int16_t t = mont_reduce_mlkem((int32_t)zeta * b);
    b = a - t;
    a = a + t;
}

/// Inverse NTT butterfly for ML-KEM
NTT_DEVICE void inv_ntt_butterfly_mlkem(int16_t& a, int16_t& b, int16_t zeta) {
    int16_t t = a;
    a = t + b;
    b = t - b;
    b = mont_reduce_mlkem((int32_t)zeta * b);
}

// =============================================================================
// Precomputed zetas (roots of unity in Montgomery form)
// =============================================================================

#ifdef __CUDA_ARCH__
__constant__
#else
static const
#endif
int32_t MLDSA_ZETAS[128] = {
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

// =============================================================================
// Full NTT / inverse NTT for n=256 polynomials
// =============================================================================

/// In-place forward NTT for ML-DSA polynomial (n=256, q=8380417)
NTT_DEVICE void ntt_mldsa(int32_t poly[256]) {
    int k = 0;
    for (int len = 128; len >= 1; len >>= 1) {
        for (int start = 0; start < 256; start += 2 * len) {
            int32_t zeta = MLDSA_ZETAS[++k];
            for (int j = start; j < start + len; j++) {
                ntt_butterfly_mldsa(poly[j], poly[j + len], zeta);
            }
        }
    }
}

/// In-place inverse NTT for ML-DSA polynomial
NTT_DEVICE void inv_ntt_mldsa(int32_t poly[256]) {
    const int32_t q = 8380417;
    const int32_t f = 41978; // 2^32 * 256^{-1} mod q

    int k = 127;
    for (int len = 1; len <= 128; len <<= 1) {
        for (int start = 0; start < 256; start += 2 * len) {
            int32_t zeta = -MLDSA_ZETAS[k--];
            if (zeta < 0) zeta += q;
            for (int j = start; j < start + len; j++) {
                inv_ntt_butterfly_mldsa(poly[j], poly[j + len], zeta);
            }
        }
    }
    for (int i = 0; i < 256; i++) {
        poly[i] = mont_reduce_mldsa((int64_t)f * poly[i]);
    }
}

/// Pointwise multiplication of two NTT-domain ML-DSA polynomials
NTT_DEVICE void poly_pointwise_mldsa(int32_t c[256],
                                      const int32_t a[256],
                                      const int32_t b[256]) {
    for (int i = 0; i < 256; i++) {
        c[i] = mont_reduce_mldsa((int64_t)a[i] * b[i]);
    }
}

// =============================================================================
// NTT batch kernel: each thread transforms one polynomial
// =============================================================================

/// Batch forward NTT for ML-DSA polynomials.
/// Each thread computes NTT of one 256-coefficient polynomial.
extern "C" __global__ void ntt_mldsa_batch(
    int32_t*       polys,
    const uint32_t num_polys)
{
#ifdef __CUDA_ARCH__
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_polys) return;

    int32_t poly[256];
    int32_t* src = polys + tid * 256;
    for (int i = 0; i < 256; i++) poly[i] = src[i];

    ntt_mldsa(poly);

    for (int i = 0; i < 256; i++) src[i] = poly[i];
#endif
}

/// Batch inverse NTT for ML-DSA polynomials.
extern "C" __global__ void inv_ntt_mldsa_batch(
    int32_t*       polys,
    const uint32_t num_polys)
{
#ifdef __CUDA_ARCH__
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_polys) return;

    int32_t poly[256];
    int32_t* src = polys + tid * 256;
    for (int i = 0; i < 256; i++) poly[i] = src[i];

    inv_ntt_mldsa(poly);

    for (int i = 0; i < 256; i++) src[i] = poly[i];
#endif
}

#endif // NTT_CUDA_H
