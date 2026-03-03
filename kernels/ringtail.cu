// Ringtail lattice-based threshold signatures -- CUDA implementation
// Matches ringtail.metal output byte-for-byte
// One thread per partial sign / combine operation

#include <cstdint>

#ifndef __CUDA_ARCH__
#define __device__
#define __global__
#define __shared__
struct dim3 { unsigned x, y, z; };
static dim3 blockIdx, blockDim, threadIdx;
#endif

// =============================================================================
// Ringtail parameters (same ring as ML-DSA)
// =============================================================================

#define RT_Q 8380417

// =============================================================================
// Modular arithmetic
// =============================================================================

__device__ static int32_t rt_reduce(int32_t a) {
    int32_t t = (int32_t)((int64_t)a * 33554687LL >> 48);
    int32_t r = a - t * RT_Q;
    if (r < 0) r += RT_Q;
    if (r >= RT_Q) r -= RT_Q;
    return r;
}

__device__ static int32_t rt_mont_reduce(int64_t a) {
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

__device__ static const int32_t RT_ZETAS[128] = {
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

__device__ static void rt_ntt_bf(int32_t& a, int32_t& b, int32_t z) {
    int32_t t = rt_mont_reduce((int64_t)z * b);
    b = a - t; a = a + t;
    if (a >= RT_Q) a -= RT_Q;
    if (b < 0) b += RT_Q;
}

__device__ static void rt_inv_ntt_bf(int32_t& a, int32_t& b, int32_t z) {
    int32_t t = a;
    a = t + b; b = t - b;
    if (a >= RT_Q) a -= RT_Q;
    if (b < 0) b += RT_Q;
    b = rt_mont_reduce((int64_t)z * b);
}

__device__ static void rt_ntt(int32_t poly[256]) {
    int k = 0;
    for (int len = 128; len >= 1; len >>= 1)
        for (int start = 0; start < 256; start += 2 * len) {
            int32_t z = RT_ZETAS[++k];
            for (int j = start; j < start + len; j++)
                rt_ntt_bf(poly[j], poly[j + len], z);
        }
}

__device__ static void rt_inv_ntt(int32_t poly[256]) {
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

// Pointwise multiply: c[i] = a[i] * b[i] mod q (NTT domain)
__device__ static void rt_poly_mul_ntt(int32_t c[256],
                                        const int32_t a[256],
                                        const int32_t b[256]) {
    for (int i = 0; i < 256; i++)
        c[i] = rt_mont_reduce((int64_t)a[i] * b[i]);
}

// Polynomial add: c[i] = a[i] + b[i] mod q
__device__ static void rt_poly_add(int32_t c[256],
                                    const int32_t a[256],
                                    const int32_t b[256]) {
    for (int i = 0; i < 256; i++)
        c[i] = rt_reduce(a[i] + b[i]);
}

// =============================================================================
// Ringtail structures
// =============================================================================

struct RingtailShare {
    uint8_t data[1024]; // 256 int32_t coefficients
};

struct RingtailMessage {
    uint8_t data[32];
};

struct RingtailPartialSig {
    uint8_t data[1024];
};

// =============================================================================
// Partial signing kernel
// =============================================================================

extern "C" __global__ void ringtail_partial_sign_batch(
    const RingtailShare*    __restrict__ shares,
    const RingtailMessage*  __restrict__ messages,
    RingtailPartialSig*     __restrict__ partial_sigs,
    const uint32_t*         __restrict__ num_ops_ptr)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t num_ops = *num_ops_ptr;
    if (tid >= num_ops) return;

    // Load share polynomial
    int32_t share[256];
    const uint8_t* sp = shares[tid].data;
    for (int i = 0; i < 256; i++) {
        share[i] = (int32_t)sp[i * 4]
                 | ((int32_t)sp[i * 4 + 1] << 8)
                 | ((int32_t)sp[i * 4 + 2] << 16)
                 | ((int32_t)sp[i * 4 + 3] << 24);
    }

    // Derive challenge polynomial from message hash
    const uint8_t* msg = messages[tid].data;
    int32_t challenge[256];
    for (int i = 0; i < 256; i++) {
        uint32_t idx = (i * 4) % 32;
        uint32_t val = (uint32_t)msg[idx]
                     | ((uint32_t)msg[(idx + 1) % 32] << 8)
                     | ((uint32_t)msg[(idx + 2) % 32] << 16)
                     | ((uint32_t)msg[(idx + 3) % 32] << 24);
        val ^= (uint32_t)(i * 2654435761u);
        challenge[i] = (int32_t)(val % (uint32_t)RT_Q);
    }

    // NTT of challenge
    rt_ntt(challenge);

    // NTT of share
    rt_ntt(share);

    // Pointwise multiply
    int32_t result[256];
    rt_poly_mul_ntt(result, share, challenge);

    // Inverse NTT
    rt_inv_ntt(result);

    // Write partial signature
    uint8_t* out = partial_sigs[tid].data;
    for (int i = 0; i < 256; i++) {
        uint32_t v = (uint32_t)result[i];
        out[i * 4]     = (uint8_t)(v & 0xFF);
        out[i * 4 + 1] = (uint8_t)((v >> 8) & 0xFF);
        out[i * 4 + 2] = (uint8_t)((v >> 16) & 0xFF);
        out[i * 4 + 3] = (uint8_t)((v >> 24) & 0xFF);
    }
}

// =============================================================================
// Combine kernel
// =============================================================================

extern "C" __global__ void ringtail_combine_batch(
    const RingtailPartialSig*  __restrict__ partial_sigs,    // [num_ops * threshold]
    const int32_t*             __restrict__ lagrange_coeffs,  // [num_ops * threshold]
    RingtailPartialSig*        __restrict__ combined_sigs,    // [num_ops]
    const uint32_t*            __restrict__ threshold_ptr,
    const uint32_t*            __restrict__ num_ops_ptr)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t num_ops = *num_ops_ptr;
    uint32_t threshold = *threshold_ptr;
    if (tid >= num_ops) return;

    int32_t combined[256];
    for (int i = 0; i < 256; i++) combined[i] = 0;

    for (uint32_t s = 0; s < threshold; s++) {
        const uint8_t* ps = partial_sigs[tid * threshold + s].data;
        int32_t lambda = lagrange_coeffs[tid * threshold + s];

        for (int i = 0; i < 256; i++) {
            int32_t coeff = (int32_t)ps[i * 4]
                          | ((int32_t)ps[i * 4 + 1] << 8)
                          | ((int32_t)ps[i * 4 + 2] << 16)
                          | ((int32_t)ps[i * 4 + 3] << 24);

            int64_t prod = (int64_t)lambda * coeff;
            combined[i] = rt_reduce(combined[i] + rt_mont_reduce(prod));
        }
    }

    // Write combined signature
    uint8_t* out = combined_sigs[tid].data;
    for (int i = 0; i < 256; i++) {
        uint32_t v = (uint32_t)combined[i];
        out[i * 4]     = (uint8_t)(v & 0xFF);
        out[i * 4 + 1] = (uint8_t)((v >> 8) & 0xFF);
        out[i * 4 + 2] = (uint8_t)((v >> 16) & 0xFF);
        out[i * 4 + 3] = (uint8_t)((v >> 24) & 0xFF);
    }
}
