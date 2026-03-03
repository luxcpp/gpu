// Ed25519 batch verification — CUDA implementation
// Matches ed25519.metal output byte-for-byte
// One thread per signature verification

#include <cstdint>

#ifndef __CUDA_ARCH__
#define __device__
#define __global__
#define __shared__
struct dim3 { unsigned x, y, z; };
static dim3 blockIdx, blockDim, threadIdx;
#endif

// =============================================================================
// 256-bit integer (4 x 64-bit limbs, little-endian)
// =============================================================================

struct uint256 {
    uint64_t limbs[4];
};

// =============================================================================
// Ed25519 constants (field prime p = 2^255 - 19)
// =============================================================================

__device__ static const uint256 ED_P = {{
    0xFFFFFFFFFFFFFFEDULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL
}};

__device__ static const uint256 ED_D = {{
    0x75EB4DCA135978A3ULL, 0x00700A4D4141D8ABULL,
    0x8CC740797779E898ULL, 0x52036CBC148B6DE8ULL
}};

__device__ static const uint256 ED_2D = {{
    0xEBD69B9426B2F159ULL, 0x00E0149A8283B156ULL,
    0x198E80F2EEF3D130ULL, 0x2406D9DC56DFFCE7ULL
}};

__device__ static const uint256 ED_L = {{
    0x5812631A5CF5D3EDULL, 0x14DEF9DEA2F79CD6ULL,
    0x0000000000000000ULL, 0x1000000000000000ULL
}};

__device__ static const uint256 ED_BY = {{
    0x6666666666666658ULL, 0x6666666666666666ULL,
    0x6666666666666666ULL, 0x6666666666666666ULL
}};

__device__ static const uint256 ED_BX = {{
    0xC9562D608F25D51AULL, 0x692CC7609525A7B2ULL,
    0xC0A4E231FDD6DC5CULL, 0x216936D3CD6E53FEULL
}};

__device__ static const uint256 ZERO = {{0, 0, 0, 0}};
__device__ static const uint256 ONE  = {{1, 0, 0, 0}};

// =============================================================================
// 256-bit arithmetic
// =============================================================================

__device__ static int u256_cmp(uint256 a, uint256 b) {
    for (int i = 3; i >= 0; i--) {
        if (a.limbs[i] < b.limbs[i]) return -1;
        if (a.limbs[i] > b.limbs[i]) return 1;
    }
    return 0;
}

__device__ static bool u256_is_zero(uint256 a) {
    return (a.limbs[0] | a.limbs[1] | a.limbs[2] | a.limbs[3]) == 0;
}

__device__ static uint256 u256_add(uint256 a, uint256 b, uint64_t& carry) {
    uint256 r;
    uint64_t c = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t sum = a.limbs[i] + c;
        c = (sum < a.limbs[i]) ? 1ULL : 0ULL;
        uint64_t sum2 = sum + b.limbs[i];
        c += (sum2 < sum) ? 1ULL : 0ULL;
        r.limbs[i] = sum2;
    }
    carry = c;
    return r;
}

__device__ static uint256 u256_sub(uint256 a, uint256 b, uint64_t& borrow) {
    uint256 r;
    uint64_t bw = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t diff = a.limbs[i] - bw;
        bw = (diff > a.limbs[i]) ? 1ULL : 0ULL;
        uint64_t diff2 = diff - b.limbs[i];
        bw += (diff2 > diff) ? 1ULL : 0ULL;
        r.limbs[i] = diff2;
    }
    borrow = bw;
    return r;
}

// =============================================================================
// Field arithmetic mod p = 2^255 - 19
// Uses __int128 for 64x64->128 multiply (native on CUDA/sm_50+)
// =============================================================================

__device__ static uint256 fp_reduce(uint256 a) {
    while (u256_cmp(a, ED_P) >= 0) {
        uint64_t bw;
        a = u256_sub(a, ED_P, bw);
    }
    return a;
}

__device__ static uint256 fp_add(uint256 a, uint256 b) {
    uint64_t c;
    uint256 r = u256_add(a, b, c);
    if (c || u256_cmp(r, ED_P) >= 0) {
        uint64_t bw;
        r = u256_sub(r, ED_P, bw);
    }
    return r;
}

__device__ static uint256 fp_sub(uint256 a, uint256 b) {
    uint64_t bw;
    uint256 r = u256_sub(a, b, bw);
    if (bw) {
        uint64_t c;
        r = u256_add(r, ED_P, c);
    }
    return r;
}

__device__ static uint256 fp_mul(uint256 a, uint256 b) {
    // Full 512-bit multiply using __int128, then reduce mod p = 2^255 - 19
    uint64_t t[8] = {};
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
#ifdef __CUDA_ARCH__
            unsigned __int128 prod = (unsigned __int128)a.limbs[i] * b.limbs[j];
            unsigned __int128 acc = prod + carry + t[i + j];
            t[i + j] = (uint64_t)acc;
            carry = (uint64_t)(acc >> 64);
#else
            // CPU fallback: split multiply
            uint64_t a_lo = a.limbs[i] & 0xFFFFFFFFULL;
            uint64_t a_hi = a.limbs[i] >> 32;
            uint64_t b_lo = b.limbs[j] & 0xFFFFFFFFULL;
            uint64_t b_hi = b.limbs[j] >> 32;
            uint64_t ll = a_lo * b_lo;
            uint64_t lh = a_lo * b_hi;
            uint64_t hl = a_hi * b_lo;
            uint64_t hh = a_hi * b_hi;
            uint64_t mid = lh + (ll >> 32);
            uint64_t mid2 = mid + hl;
            if (mid2 < mid) hh += (1ULL << 32);
            uint64_t lo = (mid2 << 32) | (ll & 0xFFFFFFFFULL);
            uint64_t hi = hh + (mid2 >> 32);
            uint64_t sum = lo + carry;
            if (sum < lo) hi++;
            lo = sum;
            sum = t[i + j] + lo;
            if (sum < t[i + j]) hi++;
            t[i + j] = sum;
            carry = hi;
#endif
        }
        t[i + 4] = carry;
    }

    // Reduce mod 2^255 - 19: 2^256 mod p = 38
    uint256 lo_part = {{t[0], t[1], t[2], t[3]}};
    uint256 hi_part = {{t[4], t[5], t[6], t[7]}};

    // Multiply hi by 38 and add to lo
    uint64_t c2 = 0;
    uint256 hi38;
    for (int i = 0; i < 4; i++) {
#ifdef __CUDA_ARCH__
        unsigned __int128 prod = (unsigned __int128)hi_part.limbs[i] * 38ULL + c2;
        hi38.limbs[i] = (uint64_t)prod;
        c2 = (uint64_t)(prod >> 64);
#else
        uint64_t a_lo = hi_part.limbs[i] & 0xFFFFFFFFULL;
        uint64_t a_hi = hi_part.limbs[i] >> 32;
        uint64_t ll = a_lo * 38ULL;
        uint64_t hl = a_hi * 38ULL;
        uint64_t lo = ll + (hl << 32);
        uint64_t hi = (hl >> 32) + ((lo < ll) ? 1ULL : 0ULL);
        uint64_t sum = lo + c2;
        if (sum < lo) hi++;
        c2 = hi;
        hi38.limbs[i] = sum;
#endif
    }

    uint64_t c;
    uint256 result = u256_add(lo_part, hi38, c);
    if (c || c2) {
        uint64_t extra = (c + c2) * 38;
        uint256 extra256 = {{extra, 0, 0, 0}};
        result = u256_add(result, extra256, c);
    }

    return fp_reduce(result);
}

__device__ static uint256 fp_sqr(uint256 a) { return fp_mul(a, a); }

__device__ static uint256 fp_neg(uint256 a) {
    if (u256_is_zero(a)) return a;
    uint64_t bw;
    return u256_sub(ED_P, a, bw);
}

__device__ static uint256 fp_inv(uint256 a) {
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

__device__ static EdPoint ed_identity() {
    EdPoint p;
    p.X = ZERO; p.Y = ONE; p.Z = ONE; p.T = ZERO;
    return p;
}

__device__ static EdPoint ed_double(EdPoint P) {
    uint256 A = fp_sqr(P.X);
    uint256 B = fp_sqr(P.Y);
    uint256 C = fp_add(fp_sqr(P.Z), fp_sqr(P.Z));
    uint256 D = fp_neg(A);
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

__device__ static EdPoint ed_add(EdPoint P, EdPoint Q) {
    uint256 A = fp_mul(P.X, Q.X);
    uint256 B = fp_mul(P.Y, Q.Y);
    uint256 C = fp_mul(P.T, fp_mul(ED_2D, Q.T));
    uint256 D = fp_mul(P.Z, Q.Z);
    D = fp_add(D, D);
    uint256 E = fp_sub(fp_mul(fp_add(P.X, P.Y), fp_add(Q.X, Q.Y)), fp_add(A, B));
    uint256 F = fp_sub(D, C);
    uint256 G = fp_add(D, C);
    uint256 H = fp_add(B, A);

    EdPoint R;
    R.X = fp_mul(E, F);
    R.Y = fp_mul(G, H);
    R.T = fp_mul(E, H);
    R.Z = fp_mul(F, G);
    return R;
}

__device__ static EdPoint ed_mul(uint256 k, EdPoint P) {
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

__device__ static void ed_to_affine(EdPoint p, uint256& x, uint256& y) {
    uint256 z_inv = fp_inv(p.Z);
    x = fp_mul(p.X, z_inv);
    y = fp_mul(p.Y, z_inv);
}

// =============================================================================
// Point decompression
// =============================================================================

__device__ static bool ed_decompress(const uint8_t* encoded, EdPoint& P) {
    uint256 y;
    for (int i = 0; i < 4; i++) {
        y.limbs[i] = 0;
        for (int b = 0; b < 8 && i * 8 + b < 32; b++) {
            y.limbs[i] |= (uint64_t)encoded[i * 8 + b] << (b * 8);
        }
    }
    bool x_sign = (encoded[31] >> 7) & 1;
    y.limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;

    if (u256_cmp(y, ED_P) >= 0) return false;

    uint256 y2 = fp_sqr(y);
    uint256 num = fp_sub(y2, ONE);
    uint256 den = fp_add(fp_mul(ED_D, y2), ONE);
    uint256 den_inv = fp_inv(den);
    uint256 x2 = fp_mul(num, den_inv);

    if (u256_is_zero(x2)) {
        if (x_sign) return false;
        P.X = ZERO; P.Y = y; P.Z = ONE; P.T = ZERO;
        return true;
    }

    // x = x2^((p+3)/8)
    uint256 exp_val = ED_P;
    exp_val.limbs[0] += 3;
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

    if (u256_cmp(fp_sqr(x), x2) != 0) {
        const uint256 SQRT_M1 = {{
            0xC4EE1B274A0EA0B0ULL, 0x2F431806AD2FE478ULL,
            0x2B4D00993DFBD7A7ULL, 0x2B8324804FC1DF0BULL
        }};
        x = fp_mul(x, SQRT_M1);
        if (u256_cmp(fp_sqr(x), x2) != 0) return false;
    }

    bool x_is_odd = x.limbs[0] & 1;
    if (x_is_odd != x_sign) x = fp_neg(x);

    P.X = x;
    P.Y = y;
    P.Z = ONE;
    P.T = fp_mul(x, y);
    return true;
}

// =============================================================================
// Structures
// =============================================================================

struct Ed25519PublicKey {
    uint8_t data[32];
};

struct Ed25519Signature {
    uint8_t data[64];
};

struct Ed25519Message {
    uint8_t hash[64];
};

// =============================================================================
// Verification kernel
// =============================================================================

extern "C" __global__ void ed25519_verify_batch(
    const Ed25519PublicKey* __restrict__  pubkeys,
    const Ed25519Message* __restrict__   messages,
    const Ed25519Signature* __restrict__ signatures,
    uint32_t* __restrict__               results,
    const uint32_t                       num_sigs)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_sigs) return;

    // Decompress public key A
    EdPoint A;
    if (!ed_decompress(pubkeys[tid].data, A)) {
        results[tid] = 0;
        return;
    }

    // Decompress signature point R
    EdPoint R;
    if (!ed_decompress(signatures[tid].data, R)) {
        results[tid] = 0;
        return;
    }

    // Read scalar S from signature (bytes 32..63, little-endian)
    uint256 S;
    for (int i = 0; i < 4; i++) {
        S.limbs[i] = 0;
        for (int b = 0; b < 8; b++) {
            S.limbs[i] |= (uint64_t)signatures[tid].data[32 + i * 8 + b] << (b * 8);
        }
    }

    if (u256_cmp(S, ED_L) >= 0) {
        results[tid] = 0;
        return;
    }

    // Read pre-computed hash scalar h (reduced mod L by host)
    uint256 h;
    for (int i = 0; i < 4; i++) {
        h.limbs[i] = 0;
        for (int b = 0; b < 8; b++) {
            h.limbs[i] |= (uint64_t)messages[tid].hash[i * 8 + b] << (b * 8);
        }
    }

    // Verify: [S]B == R + [h]A
    EdPoint B;
    B.X = ED_BX; B.Y = ED_BY; B.Z = ONE; B.T = fp_mul(ED_BX, ED_BY);
    EdPoint SB = ed_mul(S, B);

    EdPoint hA = ed_mul(h, A);
    EdPoint RhA = ed_add(R, hA);

    uint256 sb_x, sb_y, rha_x, rha_y;
    ed_to_affine(SB, sb_x, sb_y);
    ed_to_affine(RhA, rha_x, rha_y);

    bool valid = (u256_cmp(sb_x, rha_x) == 0) && (u256_cmp(sb_y, rha_y) == 0);
    results[tid] = valid ? 1u : 0u;
}
