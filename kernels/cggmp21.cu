// CGGMP21 threshold ECDSA partial signing -- CUDA implementation
// Matches cggmp21.metal output byte-for-byte
// One thread per partial signature

#include <cstdint>

#ifndef __CUDA_ARCH__
#define __device__
#define __global__
#define __shared__
struct dim3 { unsigned x, y, z; };
static dim3 blockIdx, blockDim, threadIdx;
#endif

// =============================================================================
// 2048-bit unsigned integer (32 x 64-bit limbs, little-endian)
// =============================================================================

struct uint2048 {
    uint64_t limbs[32];
};

// =============================================================================
// 2048-bit arithmetic
// =============================================================================

__device__ static void u2048_zero(uint2048& a) {
    for (int i = 0; i < 32; i++) a.limbs[i] = 0;
}

__device__ static bool u2048_is_zero(const uint2048& a) {
    uint64_t acc = 0;
    for (int i = 0; i < 32; i++) acc |= a.limbs[i];
    return acc == 0;
}

__device__ static int u2048_cmp(const uint2048& a, const uint2048& b) {
    for (int i = 31; i >= 0; i--) {
        if (a.limbs[i] < b.limbs[i]) return -1;
        if (a.limbs[i] > b.limbs[i]) return 1;
    }
    return 0;
}

__device__ static uint2048 u2048_add(const uint2048& a, const uint2048& b, uint64_t& carry) {
    uint2048 r;
    uint64_t c = 0;
    for (int i = 0; i < 32; i++) {
        uint64_t sum = a.limbs[i] + c;
        c = (sum < a.limbs[i]) ? 1ULL : 0ULL;
        uint64_t sum2 = sum + b.limbs[i];
        c += (sum2 < sum) ? 1ULL : 0ULL;
        r.limbs[i] = sum2;
    }
    carry = c;
    return r;
}

__device__ static uint2048 u2048_sub(const uint2048& a, const uint2048& b, uint64_t& borrow) {
    uint2048 r;
    uint64_t bw = 0;
    for (int i = 0; i < 32; i++) {
        uint64_t diff = a.limbs[i] - bw;
        bw = (diff > a.limbs[i]) ? 1ULL : 0ULL;
        uint64_t diff2 = diff - b.limbs[i];
        bw += (diff2 > diff) ? 1ULL : 0ULL;
        r.limbs[i] = diff2;
    }
    borrow = bw;
    return r;
}

// 64x64->128 multiply using CUDA __int128
__device__ static void mul64(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi) {
#ifdef __CUDA_ARCH__
    unsigned __int128 prod = (unsigned __int128)a * b;
    lo = (uint64_t)prod;
    hi = (uint64_t)(prod >> 64);
#else
    uint64_t a_lo = a & 0xFFFFFFFFULL, a_hi = a >> 32;
    uint64_t b_lo = b & 0xFFFFFFFFULL, b_hi = b >> 32;
    uint64_t ll = a_lo * b_lo, lh = a_lo * b_hi;
    uint64_t hl = a_hi * b_lo, hh = a_hi * b_hi;
    uint64_t mid = lh + (ll >> 32);
    uint64_t mid2 = mid + hl;
    if (mid2 < mid) hh += (1ULL << 32);
    lo = (mid2 << 32) | (ll & 0xFFFFFFFFULL);
    hi = hh + (mid2 >> 32);
#endif
}

// =============================================================================
// Montgomery multiplication for 2048-bit modulus
// =============================================================================

__device__ static void mont_reduce_2048(uint64_t t[64],
                                         const uint2048& m,
                                         uint64_t m_inv,
                                         uint2048& result) {
    uint64_t a[65];
    for (int i = 0; i < 64; i++) a[i] = t[i];
    a[64] = 0;

    for (int i = 0; i < 32; i++) {
        uint64_t u = a[i] * m_inv;
        uint64_t carry = 0;
        for (int j = 0; j < 32; j++) {
            uint64_t lo, hi;
            mul64(u, m.limbs[j], lo, hi);
            uint64_t sum = lo + carry;
            if (sum < lo) hi++;
            lo = sum;
            sum = a[i + j] + lo;
            if (sum < a[i + j]) hi++;
            a[i + j] = sum;
            carry = hi;
        }
        for (int j = 32; i + j <= 64; j++) {
            uint64_t sum = a[i + j] + carry;
            carry = (sum < a[i + j]) ? 1ULL : 0ULL;
            a[i + j] = sum;
            if (!carry) break;
        }
    }

    for (int i = 0; i < 32; i++) result.limbs[i] = a[i + 32];

    if (a[64] || u2048_cmp(result, m) >= 0) {
        uint64_t bw;
        result = u2048_sub(result, m, bw);
    }
}

__device__ static void mont_mul_2048(const uint2048& a,
                                      const uint2048& b,
                                      const uint2048& m,
                                      uint64_t m_inv,
                                      uint2048& result) {
    uint64_t t[64];
    for (int i = 0; i < 64; i++) t[i] = 0;

    for (int i = 0; i < 32; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 32; j++) {
            uint64_t lo, hi;
            mul64(a.limbs[i], b.limbs[j], lo, hi);
            uint64_t sum = lo + carry;
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

__device__ static void mont_sqr_2048(const uint2048& a,
                                      const uint2048& m,
                                      uint64_t m_inv,
                                      uint2048& result) {
    mont_mul_2048(a, a, m, m_inv, result);
}

__device__ static void mont_pow_2048(const uint2048& base,
                                      const uint2048& exp,
                                      const uint2048& m,
                                      uint64_t m_inv,
                                      const uint2048& mont_one,
                                      uint2048& result) {
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

struct PaillierPubKey {
    uint8_t n_data[256];     // N in big-endian
    uint8_t n_inv64[8];      // -N^{-1} mod 2^64 for Montgomery
};

// =============================================================================
// CGGMP21 structures
// =============================================================================

struct CGGMP21Input {
    uint8_t k_share[32];      // k_i share (secp256k1 scalar)
    uint8_t chi_share[32];    // chi_i = k_i * x_i share
    uint8_t msg_hash[32];     // Message hash
    uint8_t gamma_share[32];  // gamma_i share
};

struct CGGMP21PartialSig {
    uint8_t sigma_i[32];      // sigma_i = k_i * m + r * chi_i (mod n)
};

// =============================================================================
// secp256k1 order for scalar arithmetic
// =============================================================================

__device__ static const uint64_t SECP_N[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};

// Modular multiplication mod secp256k1 order (256-bit)
__device__ static void scalar_mul_mod_n(const uint64_t a[4],
                                         const uint64_t b[4],
                                         uint64_t result[4]) {
    // Full 512-bit product
    uint64_t t[8];
    for (int i = 0; i < 8; i++) t[i] = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t lo, hi;
            mul64(a[i], b[j], lo, hi);
            uint64_t sum = lo + carry; if (sum < lo) hi++;
            sum = t[i + j] + sum; if (sum < t[i + j]) hi++;
            t[i + j] = sum;
            carry = hi;
        }
        t[i + 4] = carry;
    }

    // Barrett reduction mod n (iterate subtraction)
    uint64_t r[4] = {t[0], t[1], t[2], t[3]};

    for (int iter = 0; iter < 4; iter++) {
        uint64_t borrow = 0;
        uint64_t diff[4];
        for (int i = 0; i < 4; i++) {
            uint64_t d = r[i] - borrow;
            borrow = (d > r[i]) ? 1ULL : 0ULL;
            uint64_t d2 = d - SECP_N[i];
            borrow += (d2 > d) ? 1ULL : 0ULL;
            diff[i] = d2;
        }
        if (!borrow) {
            for (int i = 0; i < 4; i++) r[i] = diff[i];
        }
    }

    for (int i = 0; i < 4; i++) result[i] = r[i];
}

// Modular addition mod n
__device__ static void scalar_add_mod_n(const uint64_t a[4],
                                         const uint64_t b[4],
                                         uint64_t result[4]) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t sum = a[i] + carry;
        carry = (sum < a[i]) ? 1ULL : 0ULL;
        uint64_t sum2 = sum + b[i];
        carry += (sum2 < sum) ? 1ULL : 0ULL;
        result[i] = sum2;
    }
    // Reduce
    uint64_t borrow = 0;
    uint64_t diff[4];
    for (int i = 0; i < 4; i++) {
        uint64_t d = result[i] - borrow;
        borrow = (d > result[i]) ? 1ULL : 0ULL;
        uint64_t d2 = d - SECP_N[i];
        borrow += (d2 > d) ? 1ULL : 0ULL;
        diff[i] = d2;
    }
    if (!borrow || carry) {
        for (int i = 0; i < 4; i++) result[i] = diff[i];
    }
}

// =============================================================================
// Partial signing kernel
// =============================================================================

extern "C" __global__ void cggmp21_partial_sign_batch(
    const CGGMP21Input*       __restrict__ inputs,
    CGGMP21PartialSig*        __restrict__ partial_sigs,
    const uint8_t*            __restrict__ r_x,      // 32 bytes: R.x mod n
    const uint32_t*           __restrict__ num_ops_ptr)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t num_ops = *num_ops_ptr;
    if (tid >= num_ops) return;

    // Read k_i share
    uint64_t k[4];
    for (int i = 0; i < 4; i++) {
        k[i] = 0;
        for (int b = 0; b < 8; b++)
            k[i] |= (uint64_t)inputs[tid].k_share[i * 8 + b] << (b * 8);
    }

    // Read chi_i share
    uint64_t chi[4];
    for (int i = 0; i < 4; i++) {
        chi[i] = 0;
        for (int b = 0; b < 8; b++)
            chi[i] |= (uint64_t)inputs[tid].chi_share[i * 8 + b] << (b * 8);
    }

    // Read message hash
    uint64_t msg[4];
    for (int i = 0; i < 4; i++) {
        msg[i] = 0;
        for (int b = 0; b < 8; b++)
            msg[i] |= (uint64_t)inputs[tid].msg_hash[i * 8 + b] << (b * 8);
    }

    // Read r (x-coordinate of nonce point)
    uint64_t r[4];
    for (int i = 0; i < 4; i++) {
        r[i] = 0;
        for (int b = 0; b < 8; b++)
            r[i] |= (uint64_t)r_x[i * 8 + b] << (b * 8);
    }

    // sigma_i = k_i * m + r * chi_i  (mod n)
    uint64_t km[4], rchi[4], sigma[4];
    scalar_mul_mod_n(k, msg, km);
    scalar_mul_mod_n(r, chi, rchi);
    scalar_add_mod_n(km, rchi, sigma);

    // Write output
    uint8_t* out = partial_sigs[tid].sigma_i;
    for (int i = 0; i < 4; i++) {
        for (int b = 0; b < 8; b++) {
            out[i * 8 + b] = (uint8_t)(sigma[i] >> (b * 8));
        }
    }
}
