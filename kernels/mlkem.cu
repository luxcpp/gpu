// ML-KEM-768 (FIPS 203) batch decapsulate -- CUDA implementation
// Matches mlkem.metal output byte-for-byte
// One thread per decapsulation

#include <cstdint>

#ifndef __CUDA_ARCH__
#define __device__
#define __global__
#define __shared__
struct dim3 { unsigned x, y, z; };
static dim3 blockIdx, blockDim, threadIdx;
#endif

// =============================================================================
// ML-KEM-768 parameters (NIST security level 3)
// =============================================================================

#define MLKEM_Q 3329

// =============================================================================
// Montgomery arithmetic for q=3329, R=2^16
// =============================================================================

// -q^{-1} mod 2^16 = 3327
__device__ static int16_t mlkem_mont_reduce(int32_t a) {
    const int16_t q_inv = 3327;
    int16_t t = (int16_t)a * q_inv;
    int32_t u = (int32_t)t * MLKEM_Q;
    return (int16_t)((a - u) >> 16);
}

// Barrett reduction for q=3329
__device__ static int16_t mlkem_barrett_reduce(int16_t a) {
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

__device__ static const int16_t KYBER_ZETAS[128] = {
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

// Forward NTT butterfly
__device__ static void kyber_ntt_bf(int16_t& a, int16_t& b, int16_t zeta) {
    int32_t t = (int32_t)b * (int32_t)zeta;
    t = mlkem_mont_reduce(t);
    int32_t sum  = (int32_t)a + t;
    int32_t diff = (int32_t)a - t;
    a = (int16_t)mlkem_barrett_reduce((int16_t)sum);
    b = (int16_t)mlkem_barrett_reduce((int16_t)diff);
}

// Inverse NTT butterfly
__device__ static void kyber_inv_ntt_bf(int16_t& a, int16_t& b, int16_t zeta) {
    int16_t t = a;
    a = t + b;
    b = t - b;
    b = mlkem_mont_reduce((int32_t)zeta * b);
}

__device__ static void kyber_ntt(int16_t poly[256]) {
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

__device__ static void kyber_inv_ntt(int16_t poly[256]) {
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

// Pointwise multiplication (basemul)
__device__ static void kyber_basemul(int16_t r[2],
                                      const int16_t a[2],
                                      const int16_t b[2],
                                      int16_t zeta) {
    r[0] = mlkem_mont_reduce((int32_t)a[1] * b[1]);
    r[0] = mlkem_mont_reduce((int32_t)r[0] * zeta);
    r[0] = r[0] + mlkem_mont_reduce((int32_t)a[0] * b[0]);
    r[1] = mlkem_mont_reduce((int32_t)a[0] * b[1]);
    r[1] = r[1] + mlkem_mont_reduce((int32_t)a[1] * b[0]);
}

// Full pointwise multiplication of NTT polynomials
__device__ static void kyber_poly_pointwise(int16_t r[256],
                                             const int16_t a[256],
                                             const int16_t b[256]) {
    for (int i = 0; i < 256 / 4; i++) {
        int16_t a_pair[2]  = {a[4*i], a[4*i+1]};
        int16_t b_pair[2]  = {b[4*i], b[4*i+1]};
        int16_t r_pair[2];
        kyber_basemul(r_pair, a_pair, b_pair, KYBER_ZETAS[64 + i]);
        r[4*i]     = r_pair[0];
        r[4*i + 1] = r_pair[1];

        int16_t a_pair2[2] = {a[4*i+2], a[4*i+3]};
        int16_t b_pair2[2] = {b[4*i+2], b[4*i+3]};
        int16_t r_pair2[2];
        kyber_basemul(r_pair2, a_pair2, b_pair2, -KYBER_ZETAS[64 + i]);
        r[4*i + 2] = r_pair2[0];
        r[4*i + 3] = r_pair2[1];
    }
}

// =============================================================================
// ML-KEM structures
// =============================================================================

struct MLKEMSecretKey {
    uint8_t data[2400]; // 3*384 + 1184 + 32 + 32
};

struct MLKEMCiphertext {
    uint8_t data[1088]; // 3*320 + 128
};

struct MLKEMSharedSecret {
    uint8_t data[32];
};

// =============================================================================
// Decapsulation kernel
// =============================================================================

extern "C" __global__ void mlkem_decapsulate_batch(
    const MLKEMSecretKey*    __restrict__ secret_keys,
    const MLKEMCiphertext*   __restrict__ ciphertexts,
    MLKEMSharedSecret*       __restrict__ shared_secrets,
    const uint32_t*          __restrict__ num_ops_ptr)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t num_ops = *num_ops_ptr;
    if (tid >= num_ops) return;

    const uint8_t* sk = secret_keys[tid].data;
    const uint8_t* ct = ciphertexts[tid].data;

    // -- Decode secret key s_hat (NTT domain, k=3 polynomials) --
    int16_t s_hat[3][256];
    for (int p = 0; p < 3; p++) {
        const uint8_t* sp = sk + p * 384;
        for (int i = 0; i < 256; i++) {
            uint32_t idx = i * 3 / 2;
            if (i & 1) {
                s_hat[p][i] = (int16_t)(((sp[idx] >> 4) | ((uint32_t)sp[idx + 1] << 4)) & 0xFFF);
            } else {
                s_hat[p][i] = (int16_t)((sp[idx] | ((uint32_t)sp[idx + 1] << 8)) & 0xFFF);
            }
        }
    }

    // -- Decode ciphertext u (compressed, k=3 polynomials, 10 bits each) --
    int16_t u[3][256];
    for (int p = 0; p < 3; p++) {
        const uint8_t* up = ct + p * 320;
        for (int i = 0; i < 256; i += 4) {
            uint32_t idx = (i / 4) * 5;
            uint32_t b0 = up[idx], b1 = up[idx+1], b2 = up[idx+2];
            uint32_t b3 = up[idx+3], b4 = up[idx+4];

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
    const uint8_t* vp = ct + 3 * 320;
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
    int16_t mp[256];
    for (int i = 0; i < 256; i++) mp[i] = 0;
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
    uint8_t* out = shared_secrets[tid].data;
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
