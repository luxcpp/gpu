// FROST threshold Schnorr signature verification -- CUDA implementation
// Matches frost.metal output byte-for-byte
// One thread per partial signature verification

#include <cstdint>

#ifndef __CUDA_ARCH__
#define __device__
#define __global__
#define __shared__
struct dim3 { unsigned x, y, z; };
static dim3 blockIdx, blockDim, threadIdx;
#endif

// =============================================================================
// 256-bit integer
// =============================================================================

struct uint256 {
    uint64_t limbs[4];
};

// secp256k1 constants
__device__ static const uint256 FROST_P = {{
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
}};

__device__ static const uint256 FROST_N = {{
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
}};

__device__ static const uint256 FROST_GX = {{
    0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL
}};

__device__ static const uint256 FROST_GY = {{
    0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL
}};

// Montgomery constants for field p
__device__ static const uint256 FROST_MONT_R2_P = {{
    0x000007A2000E90A1ULL, 0x0000000000000001ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL
}};
__device__ static const uint64_t FROST_P_INV = 0xD838091DD2253531ULL;

__device__ static const uint256 FROST_MONT_R = {{
    0x00000001000003D1ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL
}};

__device__ static const uint256 FROST_ZERO = {{0, 0, 0, 0}};

// =============================================================================
// 256-bit arithmetic
// CUDA has __int128, use it for 64x64->128 multiply
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
    uint256 r; uint64_t c = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t sum = a.limbs[i] + c;
        c = (sum < a.limbs[i]) ? 1ULL : 0ULL;
        uint64_t sum2 = sum + b.limbs[i];
        c += (sum2 < sum) ? 1ULL : 0ULL;
        r.limbs[i] = sum2;
    }
    carry = c; return r;
}

__device__ static uint256 u256_sub(uint256 a, uint256 b, uint64_t& borrow) {
    uint256 r; uint64_t bw = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t diff = a.limbs[i] - bw;
        bw = (diff > a.limbs[i]) ? 1ULL : 0ULL;
        uint64_t diff2 = diff - b.limbs[i];
        bw += (diff2 > diff) ? 1ULL : 0ULL;
        r.limbs[i] = diff2;
    }
    borrow = bw; return r;
}

// CUDA has __int128 -- use it instead of the manual 32-bit split
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

// Montgomery reduction for 256-bit field
__device__ static uint256 mont_reduce(uint64_t t[8], uint256 m, uint64_t inv) {
    uint64_t a[9];
    for (int i = 0; i < 8; i++) a[i] = t[i];
    a[8] = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t u = a[i] * inv;
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t lo, hi;
            mul64(u, m.limbs[j], lo, hi);
            uint64_t sum = lo + carry; if (sum < lo) hi++;
            lo = sum;
            sum = a[i + j] + lo; if (sum < a[i + j]) hi++;
            a[i + j] = sum;
            carry = hi;
        }
        for (int j = 4; i + j <= 8; j++) {
            uint64_t sum = a[i + j] + carry;
            carry = (sum < a[i + j]) ? 1ULL : 0ULL;
            a[i + j] = sum;
            if (!carry) break;
        }
    }
    uint256 r = {{a[4], a[5], a[6], a[7]}};
    if (a[8] || u256_cmp(r, m) >= 0) { uint64_t bw; r = u256_sub(r, m, bw); }
    return r;
}

__device__ static uint256 fp_mul(uint256 a, uint256 b) {
    uint64_t t[8] = {};
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t lo, hi;
            mul64(a.limbs[i], b.limbs[j], lo, hi);
            uint64_t sum = lo + carry; if (sum < lo) hi++;
            sum = t[i + j] + sum; if (sum < t[i + j]) hi++;
            t[i + j] = sum;
            carry = hi;
        }
        t[i + 4] = carry;
    }
    return mont_reduce(t, FROST_P, FROST_P_INV);
}

__device__ static uint256 fp_sqr(uint256 a) { return fp_mul(a, a); }

__device__ static uint256 fp_add(uint256 a, uint256 b) {
    uint64_t c; uint256 r = u256_add(a, b, c);
    if (c || u256_cmp(r, FROST_P) >= 0) { uint64_t bw; r = u256_sub(r, FROST_P, bw); }
    return r;
}

__device__ static uint256 fp_sub(uint256 a, uint256 b) {
    uint64_t bw; uint256 r = u256_sub(a, b, bw);
    if (bw) { uint64_t c; r = u256_add(r, FROST_P, c); }
    return r;
}

__device__ static uint256 to_mont(uint256 a) { return fp_mul(a, FROST_MONT_R2_P); }

__device__ static uint256 fp_inv(uint256 a) {
    uint256 exp = FROST_P; exp.limbs[0] -= 2;
    uint256 result = FROST_MONT_R, base = a;
    for (int i = 0; i < 4; i++)
        for (int bit = 0; bit < 64; bit++) {
            if ((exp.limbs[i] >> bit) & 1) result = fp_mul(result, base);
            base = fp_sqr(base);
        }
    return result;
}

// =============================================================================
// secp256k1 point (Jacobian)
// =============================================================================

struct Point {
    uint256 x, y, z;
};

__device__ static Point point_identity() {
    Point p; p.x = FROST_MONT_R; p.y = FROST_MONT_R; p.z = FROST_ZERO;
    return p;
}

__device__ static bool point_is_inf(Point p) { return u256_is_zero(p.z); }

__device__ static Point point_double(Point p) {
    if (point_is_inf(p)) return p;
    uint256 A = fp_sqr(p.y);
    uint256 B = fp_mul(p.x, A);
    uint256 S = fp_add(B, B); S = fp_add(S, S);
    uint256 C = fp_sqr(A);
    uint256 X2 = fp_sqr(p.x);
    uint256 M = fp_add(X2, fp_add(X2, X2));
    uint256 X3 = fp_sub(fp_sqr(M), fp_add(S, S));
    uint256 C8 = fp_add(C, C); C8 = fp_add(C8, C8); C8 = fp_add(C8, C8);
    uint256 Y3 = fp_sub(fp_mul(M, fp_sub(S, X3)), C8);
    uint256 Z3 = fp_mul(p.y, p.z); Z3 = fp_add(Z3, Z3);
    Point r; r.x = X3; r.y = Y3; r.z = Z3; return r;
}

__device__ static Point point_add_mixed(Point P, uint256 Qx, uint256 Qy) {
    if (point_is_inf(P)) { Point r; r.x = Qx; r.y = Qy; r.z = FROST_MONT_R; return r; }
    uint256 Z2 = fp_sqr(P.z);
    uint256 U2 = fp_mul(Qx, Z2);
    uint256 S2 = fp_mul(Qy, fp_mul(Z2, P.z));
    uint256 H = fp_sub(U2, P.x);
    uint256 R = fp_sub(S2, P.y);
    if (u256_is_zero(H)) {
        if (u256_is_zero(R)) return point_double(P);
        return point_identity();
    }
    uint256 H2 = fp_sqr(H);
    uint256 H3 = fp_mul(H, H2);
    uint256 U1H2 = fp_mul(P.x, H2);
    uint256 X3 = fp_sub(fp_sub(fp_sqr(R), H3), fp_add(U1H2, U1H2));
    uint256 Y3 = fp_sub(fp_mul(R, fp_sub(U1H2, X3)), fp_mul(P.y, H3));
    uint256 Z3 = fp_mul(H, P.z);
    Point r; r.x = X3; r.y = Y3; r.z = Z3; return r;
}

__device__ static Point point_mul(uint256 k, uint256 Px, uint256 Py) {
    Point result = point_identity();
    for (int i = 3; i >= 0; i--)
        for (int bit = 63; bit >= 0; bit--) {
            result = point_double(result);
            if ((k.limbs[i] >> bit) & 1)
                result = point_add_mixed(result, Px, Py);
        }
    return result;
}

__device__ static void point_to_affine(Point p, uint256& ax, uint256& ay) {
    if (point_is_inf(p)) { ax = FROST_ZERO; ay = FROST_ZERO; return; }
    uint256 zi = fp_inv(p.z);
    uint256 zi2 = fp_sqr(zi);
    ax = fp_mul(p.x, zi2);
    ay = fp_mul(p.y, fp_mul(zi2, zi));
}

// =============================================================================
// FROST structures
// =============================================================================

struct FROSTCommitment {
    uint8_t data[66]; // D[33] || E[33]
};

struct FROSTPartialSig {
    uint8_t data[32]; // z_i scalar
};

struct FROSTPublicKey {
    uint8_t data[33]; // compressed secp256k1 point
};

struct FROSTChallenge {
    uint8_t data[32]; // c * lambda_i scalar
};

// =============================================================================
// Verification kernel
// =============================================================================

extern "C" __global__ void frost_partial_verify_batch(
    const FROSTCommitment*  __restrict__ commitments,
    const FROSTPartialSig*  __restrict__ signatures,
    const FROSTPublicKey*   __restrict__ pubkeys,
    const FROSTChallenge*   __restrict__ challenges,
    uint32_t*               __restrict__ results,
    const uint32_t*         __restrict__ num_ops_ptr)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t num_ops = *num_ops_ptr;
    if (tid >= num_ops) return;

    // Read z_i scalar
    uint256 z;
    for (int i = 0; i < 4; i++) {
        z.limbs[i] = 0;
        for (int b = 0; b < 8; b++)
            z.limbs[i] |= (uint64_t)signatures[tid].data[i * 8 + b] << (b * 8);
    }

    // z must be < n
    if (u256_cmp(z, FROST_N) >= 0) {
        results[tid] = 0;
        return;
    }

    // Read c * lambda_i scalar
    uint256 cl;
    for (int i = 0; i < 4; i++) {
        cl.limbs[i] = 0;
        for (int b = 0; b < 8; b++)
            cl.limbs[i] |= (uint64_t)challenges[tid].data[i * 8 + b] << (b * 8);
    }

    // Decompress commitment D (first 33 bytes)
    const uint8_t* comm = commitments[tid].data;
    uint256 dx_raw;
    for (int i = 0; i < 4; i++) {
        dx_raw.limbs[i] = 0;
        for (int b = 0; b < 8 && i * 8 + b < 32; b++) {
            int src = 32 - (i * 8 + b);
            if (src >= 1 && src <= 32)
                dx_raw.limbs[i] |= (uint64_t)comm[src] << (b * 8);
        }
    }

    uint256 dx_mont = to_mont(dx_raw);

    // Recover y from x on secp256k1: y^2 = x^3 + 7
    uint256 x2 = fp_sqr(dx_mont);
    uint256 x3 = fp_mul(x2, dx_mont);
    uint256 b7 = to_mont(uint256{{7, 0, 0, 0}});
    uint256 y2 = fp_add(x3, b7);

    // sqrt via (p+1)/4 (p = 3 mod 4)
    uint256 exp = FROST_P;
    exp.limbs[0] += 1;
    for (int i = 0; i < 3; i++)
        exp.limbs[i] = (exp.limbs[i] >> 2) | (exp.limbs[i + 1] << 62);
    exp.limbs[3] >>= 2;

    uint256 dy_mont = FROST_MONT_R;
    uint256 base_y = y2;
    for (int i = 0; i < 4; i++)
        for (int bit = 0; bit < 64; bit++) {
            if ((exp.limbs[i] >> bit) & 1) dy_mont = fp_mul(dy_mont, base_y);
            base_y = fp_sqr(base_y);
        }

    // Compute z*G
    uint256 gx_mont = to_mont(FROST_GX);
    uint256 gy_mont = to_mont(FROST_GY);
    Point zG = point_mul(z, gx_mont, gy_mont);

    // Decompress public key Y_i
    const uint8_t* pk = pubkeys[tid].data;
    uint256 yx_raw;
    for (int i = 0; i < 4; i++) {
        yx_raw.limbs[i] = 0;
        for (int b = 0; b < 8 && i * 8 + b < 32; b++) {
            int src = 32 - (i * 8 + b);
            if (src >= 1 && src <= 32)
                yx_raw.limbs[i] |= (uint64_t)pk[src] << (b * 8);
        }
    }
    uint256 yx_mont = to_mont(yx_raw);

    // Recover y for public key
    uint256 yx2 = fp_sqr(yx_mont);
    uint256 yx3 = fp_mul(yx2, yx_mont);
    uint256 yy2 = fp_add(yx3, b7);

    uint256 yy_mont = FROST_MONT_R;
    uint256 base_yy = yy2;
    for (int i = 0; i < 4; i++)
        for (int bit = 0; bit < 64; bit++) {
            if ((exp.limbs[i] >> bit) & 1) yy_mont = fp_mul(yy_mont, base_yy);
            base_yy = fp_sqr(base_yy);
        }

    // Compute c*lambda_i * Y_i
    Point clY = point_mul(cl, yx_mont, yy_mont);

    // Compute R + c*lambda_i*Y_i
    uint256 cl_ax, cl_ay;
    point_to_affine(clY, cl_ax, cl_ay);

    Point R_point;
    R_point.x = dx_mont; R_point.y = dy_mont; R_point.z = FROST_MONT_R;
    Point sum = point_add_mixed(R_point, cl_ax, cl_ay);

    // Compare z*G == R + c*lambda_i*Y_i
    uint256 zg_x, zg_y, sum_x, sum_y;
    point_to_affine(zG, zg_x, zg_y);
    point_to_affine(sum, sum_x, sum_y);

    bool valid = (u256_cmp(zg_x, sum_x) == 0) && (u256_cmp(zg_y, sum_y) == 0);
    results[tid] = valid ? 1u : 0u;
}
