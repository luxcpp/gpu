// sr25519/Ristretto255 batch verification — CUDA implementation
// Matches sr25519.metal output byte-for-byte
// One thread per Schnorr signature verification

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
// Constants (same field as Ed25519: p = 2^255 - 19)
// =============================================================================

__device__ static const uint256 SR_P = {{
    0xFFFFFFFFFFFFFFEDULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL
}};

__device__ static const uint256 SR_L = {{
    0x5812631A5CF5D3EDULL, 0x14DEF9DEA2F79CD6ULL,
    0x0000000000000000ULL, 0x1000000000000000ULL
}};

__device__ static const uint256 SR_ZERO = {{0, 0, 0, 0}};
__device__ static const uint256 SR_ONE  = {{1, 0, 0, 0}};

__device__ static const uint256 SR_D = {{
    0x75EB4DCA135978A3ULL, 0x00700A4D4141D8ABULL,
    0x8CC740797779E898ULL, 0x52036CBC148B6DE8ULL
}};

__device__ static const uint256 SR_2D = {{
    0xEBD69B9426B2F159ULL, 0x00E0149A8283B156ULL,
    0x198E80F2EEF3D130ULL, 0x2406D9DC56DFFCE7ULL
}};

__device__ static const uint256 SR_SQRT_M1 = {{
    0xC4EE1B274A0EA0B0ULL, 0x2F431806AD2FE478ULL,
    0x2B4D00993DFBD7A7ULL, 0x2B8324804FC1DF0BULL
}};

// =============================================================================
// Field arithmetic (standalone, same as ed25519.metal)
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

__device__ static uint256 fp_add(uint256 a, uint256 b) {
    uint64_t c;
    uint256 r = u256_add(a, b, c);
    if (c || u256_cmp(r, SR_P) >= 0) { uint64_t bw; r = u256_sub(r, SR_P, bw); }
    return r;
}

__device__ static uint256 fp_sub(uint256 a, uint256 b) {
    uint64_t bw;
    uint256 r = u256_sub(a, b, bw);
    if (bw) { uint64_t c; r = u256_add(r, SR_P, c); }
    return r;
}

__device__ static uint256 fp_mul(uint256 a, uint256 b) {
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
            uint64_t a_lo = a.limbs[i] & 0xFFFFFFFFULL, a_hi = a.limbs[i] >> 32;
            uint64_t b_lo = b.limbs[j] & 0xFFFFFFFFULL, b_hi = b.limbs[j] >> 32;
            uint64_t ll = a_lo * b_lo, lh = a_lo * b_hi;
            uint64_t hl = a_hi * b_lo, hh = a_hi * b_hi;
            uint64_t mid = lh + (ll >> 32);
            uint64_t mid2 = mid + hl;
            if (mid2 < mid) hh += (1ULL << 32);
            uint64_t lo = (mid2 << 32) | (ll & 0xFFFFFFFFULL);
            uint64_t hi = hh + (mid2 >> 32);
            uint64_t sum = lo + carry; if (sum < lo) hi++;
            lo = sum;
            sum = t[i + j] + lo; if (sum < t[i + j]) hi++;
            t[i + j] = sum;
            carry = hi;
#endif
        }
        t[i + 4] = carry;
    }
    uint256 lo_part = {{t[0], t[1], t[2], t[3]}};
    uint256 hi_part = {{t[4], t[5], t[6], t[7]}};
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
    while (u256_cmp(result, SR_P) >= 0) { uint64_t bw; result = u256_sub(result, SR_P, bw); }
    return result;
}

__device__ static uint256 fp_sqr(uint256 a) { return fp_mul(a, a); }

__device__ static uint256 fp_neg(uint256 a) {
    if (u256_is_zero(a)) return a;
    uint64_t bw; return u256_sub(SR_P, a, bw);
}

__device__ static uint256 fp_inv(uint256 a) {
    uint256 exp = SR_P; exp.limbs[0] -= 2;
    uint256 result = SR_ONE, base = a;
    for (int i = 0; i < 4; i++)
        for (int bit = 0; bit < 64; bit++) {
            if ((exp.limbs[i] >> bit) & 1) result = fp_mul(result, base);
            base = fp_sqr(base);
        }
    return result;
}

// =============================================================================
// Extended Edwards point (same curve as Ed25519)
// =============================================================================

struct RistrettoPoint {
    uint256 X, Y, Z, T;
};

__device__ static RistrettoPoint ristretto_identity() {
    RistrettoPoint p;
    p.X = SR_ZERO; p.Y = SR_ONE; p.Z = SR_ONE; p.T = SR_ZERO;
    return p;
}

__device__ static RistrettoPoint ristretto_double(RistrettoPoint P) {
    uint256 A = fp_sqr(P.X);
    uint256 B = fp_sqr(P.Y);
    uint256 C = fp_add(fp_sqr(P.Z), fp_sqr(P.Z));
    uint256 D = fp_neg(A);
    uint256 E = fp_sub(fp_sqr(fp_add(P.X, P.Y)), fp_add(A, B));
    uint256 G = fp_add(D, B);
    uint256 F = fp_sub(G, C);
    uint256 H = fp_sub(D, B);
    RistrettoPoint R;
    R.X = fp_mul(E, F); R.Y = fp_mul(G, H);
    R.T = fp_mul(E, H); R.Z = fp_mul(F, G);
    return R;
}

__device__ static RistrettoPoint ristretto_add(RistrettoPoint P, RistrettoPoint Q) {
    uint256 A = fp_mul(P.X, Q.X);
    uint256 B = fp_mul(P.Y, Q.Y);
    uint256 C = fp_mul(P.T, fp_mul(SR_2D, Q.T));
    uint256 D = fp_add(fp_mul(P.Z, Q.Z), fp_mul(P.Z, Q.Z));
    uint256 E = fp_sub(fp_mul(fp_add(P.X, P.Y), fp_add(Q.X, Q.Y)), fp_add(A, B));
    uint256 F = fp_sub(D, C);
    uint256 G = fp_add(D, C);
    uint256 H = fp_add(B, A);
    RistrettoPoint R;
    R.X = fp_mul(E, F); R.Y = fp_mul(G, H);
    R.T = fp_mul(E, H); R.Z = fp_mul(F, G);
    return R;
}

__device__ static RistrettoPoint ristretto_mul(uint256 k, RistrettoPoint P) {
    RistrettoPoint result = ristretto_identity();
    for (int i = 3; i >= 0; i--)
        for (int bit = 63; bit >= 0; bit--) {
            result = ristretto_double(result);
            if ((k.limbs[i] >> bit) & 1) result = ristretto_add(result, P);
        }
    return result;
}

// =============================================================================
// Ristretto255 decoding
// =============================================================================

__device__ static bool ristretto_decode(const uint8_t* encoded, RistrettoPoint& P) {
    uint256 s;
    for (int i = 0; i < 4; i++) {
        s.limbs[i] = 0;
        for (int b = 0; b < 8 && i * 8 + b < 32; b++)
            s.limbs[i] |= (uint64_t)encoded[i * 8 + b] << (b * 8);
    }

    if (s.limbs[3] >> 63) return false;
    if (u256_cmp(s, SR_P) >= 0) return false;

    uint256 ss = fp_sqr(s);
    uint256 u1 = fp_sub(SR_ONE, ss);
    uint256 u2 = fp_add(SR_ONE, ss);
    uint256 u2_sq = fp_sqr(u2);
    uint256 v = fp_sub(fp_neg(fp_mul(SR_D, fp_sqr(u1))), u2_sq);

    uint256 vu2sq = fp_mul(v, u2_sq);

    // (v * u2^2)^((p-5)/8)
    uint256 exp58 = SR_P;
    exp58.limbs[0] -= 5;
    for (int i = 0; i < 3; i++)
        exp58.limbs[i] = (exp58.limbs[i] >> 3) | (exp58.limbs[i + 1] << 61);
    exp58.limbs[3] >>= 3;

    uint256 inv_sqrt = SR_ONE;
    uint256 base = vu2sq;
    for (int i = 0; i < 4; i++)
        for (int bit = 0; bit < 64; bit++) {
            if ((exp58.limbs[i] >> bit) & 1) inv_sqrt = fp_mul(inv_sqrt, base);
            base = fp_sqr(base);
        }

    uint256 check = fp_mul(fp_sqr(inv_sqrt), vu2sq);
    if (u256_cmp(check, SR_ONE) != 0) {
        uint256 neg1 = fp_neg(SR_ONE);
        if (u256_cmp(check, neg1) == 0) {
            inv_sqrt = fp_mul(inv_sqrt, SR_SQRT_M1);
        } else {
            return false;
        }
    }

    uint256 x = fp_mul(fp_add(s, s), fp_mul(inv_sqrt, u2));
    if (x.limbs[0] & 1) x = fp_neg(x);

    uint256 y = fp_mul(u1, fp_mul(inv_sqrt, u2));

    P.X = x;
    P.Y = y;
    P.Z = SR_ONE;
    P.T = fp_mul(x, y);
    return true;
}

// =============================================================================
// Structures
// =============================================================================

struct Sr25519PublicKey {
    uint8_t data[32];
};

struct Sr25519Signature {
    uint8_t data[64];
};

struct Sr25519Message {
    uint8_t hash[64];
};

// =============================================================================
// Verification kernel
// =============================================================================

extern "C" __global__ void sr25519_verify_batch(
    const Sr25519PublicKey* __restrict__  pubkeys,
    const Sr25519Message* __restrict__   messages,
    const Sr25519Signature* __restrict__ signatures,
    uint32_t* __restrict__               results,
    const uint32_t                       num_sigs)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_sigs) return;

    // Decode public key (Ristretto255 point)
    RistrettoPoint A;
    if (!ristretto_decode(pubkeys[tid].data, A)) {
        results[tid] = 0;
        return;
    }

    // Decode R from signature
    RistrettoPoint R;
    if (!ristretto_decode(signatures[tid].data, R)) {
        results[tid] = 0;
        return;
    }

    // Read scalar s
    uint256 s;
    for (int i = 0; i < 4; i++) {
        s.limbs[i] = 0;
        for (int b = 0; b < 8; b++)
            s.limbs[i] |= (uint64_t)signatures[tid].data[32 + i * 8 + b] << (b * 8);
    }

    if (u256_cmp(s, SR_L) >= 0) {
        results[tid] = 0;
        return;
    }

    // Read pre-computed challenge scalar (reduced mod L by host)
    uint256 c;
    for (int i = 0; i < 4; i++) {
        c.limbs[i] = 0;
        for (int b = 0; b < 8; b++)
            c.limbs[i] |= (uint64_t)messages[tid].hash[i * 8 + b] << (b * 8);
    }

    // Generator point B (same as Ed25519)
    const uint256 BX = {{0xC9562D608F25D51AULL, 0x692CC7609525A7B2ULL,
                         0xC0A4E231FDD6DC5CULL, 0x216936D3CD6E53FEULL}};
    const uint256 BY = {{0x6666666666666658ULL, 0x6666666666666666ULL,
                         0x6666666666666666ULL, 0x6666666666666666ULL}};
    RistrettoPoint B;
    B.X = BX; B.Y = BY; B.Z = SR_ONE; B.T = fp_mul(BX, BY);

    // Verify: s*B == R + c*A
    RistrettoPoint sB = ristretto_mul(s, B);
    RistrettoPoint cA = ristretto_mul(c, A);
    RistrettoPoint RcA = ristretto_add(R, cA);

    // Compare in affine
    uint256 sb_x = fp_mul(sB.X, fp_inv(sB.Z));
    uint256 sb_y = fp_mul(sB.Y, fp_inv(sB.Z));
    uint256 rca_x = fp_mul(RcA.X, fp_inv(RcA.Z));
    uint256 rca_y = fp_mul(RcA.Y, fp_inv(RcA.Z));

    bool valid = (u256_cmp(sb_x, rca_x) == 0) && (u256_cmp(sb_y, rca_y) == 0);
    results[tid] = valid ? 1u : 0u;
}
