// BLS12-381 batch signature verification — CUDA implementation
// Matches bls12_381.metal output byte-for-byte
// 384-bit Montgomery field arithmetic for G1 point operations

#include <cstdint>

#ifndef __CUDA_ARCH__
#define __device__
#define __global__
#define __shared__
struct dim3 { unsigned x, y, z; };
static dim3 blockIdx, blockDim, threadIdx;
static inline void __syncthreads() {}
template<typename T> static inline T atomicAdd(T* addr, T val) { T old = *addr; *addr += val; return old; }
#endif

// =============================================================================
// 384-bit unsigned integer (6 x 64-bit limbs, little-endian)
// =============================================================================

struct uint384 {
    uint64_t limbs[6];
};

// =============================================================================
// BLS12-381 constants
// =============================================================================

__device__ static const uint384 BLS_P = {{
    0xB9FEFFFFFFFFAAABULL,
    0x1EABFFFEB153FFFFULL,
    0x6730D2A0F6B0F624ULL,
    0x64774B84F38512BFULL,
    0x4B1BA7B6434BACD7ULL,
    0x1A0111EA397FE69AULL
}};

__device__ static const uint384 BLS_R2 = {{
    0xF4DF1F341C341746ULL,
    0x0A76E6A609D104F1ULL,
    0x8DE5476C4C95B6D5ULL,
    0x67EB88A9939D83C0ULL,
    0x9A793E85B519952DULL,
    0x11988FE592CAE3AAULL
}};

__device__ static const uint384 BLS_R = {{
    0x760900000002FFCDULL,
    0xEBF4000BC40C0002ULL,
    0x5F48985753C758BAULL,
    0x77CE585370525745ULL,
    0x5C071A97A256EC6DULL,
    0x15F65EC3FA80E493ULL
}};

__device__ static const uint64_t BLS_P_INV = 0x89F3FFFCFFFCFFFDULL;

__device__ static const uint384 G1_X = {{
    0x5CB38790FD666E19ULL,
    0xF85DDE8F09FE5D5CULL,
    0x2C0B0A5CAFB74CD8ULL,
    0x95F7B3B14AAE717DULL,
    0x70E02F1AB69D14E3ULL,
    0x03C26A6D58B32048ULL
}};

__device__ static const uint384 G1_Y = {{
    0xA402B931448DC5C8ULL,
    0xFBD6AA1ADEAD1CF6ULL,
    0x5B9D93D1BA1F5B57ULL,
    0x6DC08AFF5B3AF6DDULL,
    0xA4CF5B5C1B6CE90CULL,
    0x13F48FFF25F51018ULL
}};

__device__ static const uint384 ZERO384 = {{0, 0, 0, 0, 0, 0}};

// =============================================================================
// 384-bit arithmetic
// =============================================================================

__device__ static int u384_cmp(uint384 a, uint384 b) {
    for (int i = 5; i >= 0; i--) {
        if (a.limbs[i] < b.limbs[i]) return -1;
        if (a.limbs[i] > b.limbs[i]) return 1;
    }
    return 0;
}

__device__ static bool u384_is_zero(uint384 a) {
    return (a.limbs[0] | a.limbs[1] | a.limbs[2] |
            a.limbs[3] | a.limbs[4] | a.limbs[5]) == 0;
}

__device__ static uint384 u384_add(uint384 a, uint384 b, uint64_t& carry) {
    uint384 r;
    uint64_t c = 0;
    for (int i = 0; i < 6; i++) {
        uint64_t sum = a.limbs[i] + c;
        c = (sum < a.limbs[i]) ? 1ULL : 0ULL;
        uint64_t sum2 = sum + b.limbs[i];
        c += (sum2 < sum) ? 1ULL : 0ULL;
        r.limbs[i] = sum2;
    }
    carry = c;
    return r;
}

__device__ static uint384 u384_sub(uint384 a, uint384 b, uint64_t& borrow) {
    uint384 r;
    uint64_t bw = 0;
    for (int i = 0; i < 6; i++) {
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
// Montgomery arithmetic over Fp (384-bit)
// Uses __int128 for 64x64->128 multiply on CUDA
// =============================================================================

__device__ static uint384 mont_reduce_384(uint64_t t[12]) {
    uint64_t a[13];
    for (int i = 0; i < 12; i++) a[i] = t[i];
    a[12] = 0;

    for (int i = 0; i < 6; i++) {
        uint64_t u = a[i] * BLS_P_INV;

        uint64_t carry = 0;
        for (int j = 0; j < 6; j++) {
#ifdef __CUDA_ARCH__
            unsigned __int128 prod = (unsigned __int128)u * BLS_P.limbs[j];
            unsigned __int128 acc = prod + carry + a[i + j];
            a[i + j] = (uint64_t)acc;
            carry = (uint64_t)(acc >> 64);
#else
            uint64_t u_lo = u & 0xFFFFFFFFULL;
            uint64_t u_hi = u >> 32;
            uint64_t m_lo = BLS_P.limbs[j] & 0xFFFFFFFFULL;
            uint64_t m_hi = BLS_P.limbs[j] >> 32;
            uint64_t ll = u_lo * m_lo;
            uint64_t lh = u_lo * m_hi;
            uint64_t hl = u_hi * m_lo;
            uint64_t hh = u_hi * m_hi;
            uint64_t mid = lh + (ll >> 32);
            uint64_t mid2 = mid + hl;
            if (mid2 < mid) hh += (1ULL << 32);
            uint64_t lo = (mid2 << 32) | (ll & 0xFFFFFFFFULL);
            uint64_t hi = hh + (mid2 >> 32);
            uint64_t sum = lo + carry;
            if (sum < lo) hi++;
            lo = sum;
            sum = a[i + j] + lo;
            if (sum < a[i + j]) hi++;
            a[i + j] = sum;
            carry = hi;
#endif
        }
        for (int j = 6; i + j <= 12; j++) {
            uint64_t sum = a[i + j] + carry;
            carry = (sum < a[i + j]) ? 1ULL : 0ULL;
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
        uint64_t bw;
        r = u384_sub(r, BLS_P, bw);
    }
    return r;
}

__device__ static uint384 fp_mul(uint384 a, uint384 b) {
    uint64_t t[12] = {};

    for (int i = 0; i < 6; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 6; j++) {
#ifdef __CUDA_ARCH__
            unsigned __int128 prod = (unsigned __int128)a.limbs[i] * b.limbs[j];
            unsigned __int128 acc = prod + carry + t[i + j];
            t[i + j] = (uint64_t)acc;
            carry = (uint64_t)(acc >> 64);
#else
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
        for (int j = 6; i + j < 12; j++) {
            uint64_t sum = t[i + j] + carry;
            carry = (sum < t[i + j]) ? 1ULL : 0ULL;
            t[i + j] = sum;
            if (carry == 0) break;
        }
    }

    return mont_reduce_384(t);
}

__device__ static uint384 fp_sqr(uint384 a) {
    return fp_mul(a, a);
}

__device__ static uint384 fp_add(uint384 a, uint384 b) {
    uint64_t carry;
    uint384 r = u384_add(a, b, carry);
    if (carry || u384_cmp(r, BLS_P) >= 0) {
        uint64_t bw;
        r = u384_sub(r, BLS_P, bw);
    }
    return r;
}

__device__ static uint384 fp_sub(uint384 a, uint384 b) {
    uint64_t bw;
    uint384 r = u384_sub(a, b, bw);
    if (bw) {
        uint64_t c;
        r = u384_add(r, BLS_P, c);
    }
    return r;
}

__device__ static uint384 fp_neg(uint384 a) {
    if (u384_is_zero(a)) return a;
    uint64_t bw;
    return u384_sub(BLS_P, a, bw);
}

__device__ static uint384 to_mont(uint384 a) {
    return fp_mul(a, BLS_R2);
}

__device__ static uint384 from_mont(uint384 a) {
    uint64_t t[12] = {a.limbs[0], a.limbs[1], a.limbs[2],
                      a.limbs[3], a.limbs[4], a.limbs[5],
                      0, 0, 0, 0, 0, 0};
    return mont_reduce_384(t);
}

__device__ static uint384 fp_inv(uint384 a) {
    uint384 exp = BLS_P;
    exp.limbs[0] -= 2;

    uint384 result = BLS_R;
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

__device__ static G1Point g1_identity() {
    G1Point p;
    p.x = BLS_R;
    p.y = BLS_R;
    p.z = ZERO384;
    return p;
}

__device__ static bool g1_is_infinity(G1Point p) {
    return u384_is_zero(p.z);
}

__device__ static G1Point g1_double(G1Point p) {
    if (g1_is_infinity(p)) return p;

    uint384 A = fp_sqr(p.y);
    uint384 B = fp_mul(p.x, A);
    uint384 C = fp_sqr(A);

    uint384 S = fp_add(B, B);
    S = fp_add(S, S);

    uint384 X2 = fp_sqr(p.x);
    uint384 M = fp_add(X2, fp_add(X2, X2));

    uint384 X3 = fp_sub(fp_sqr(M), fp_add(S, S));

    uint384 C8 = fp_add(C, C);
    C8 = fp_add(C8, C8);
    C8 = fp_add(C8, C8);
    uint384 Y3 = fp_sub(fp_mul(M, fp_sub(S, X3)), C8);

    uint384 Z3 = fp_mul(p.y, p.z);
    Z3 = fp_add(Z3, Z3);

    G1Point r;
    r.x = X3; r.y = Y3; r.z = Z3;
    return r;
}

__device__ static G1Point g1_add_mixed(G1Point P, uint384 Qx, uint384 Qy) {
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

__device__ static G1Point g1_mul(uint384 k, uint384 Px, uint384 Py) {
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

__device__ static void g1_to_affine(G1Point p, uint384& ax, uint384& ay) {
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

struct BLSSignature {
    uint8_t data[48];
};

struct BLSPublicKey {
    uint8_t data[96];
};

struct BLSMessage {
    uint8_t data[32];
};

// =============================================================================
// Deserialization
// =============================================================================

__device__ static uint384 deserialize_fp(const uint8_t* data) {
    uint384 r = {};
    for (int limb = 0; limb < 6; limb++) {
        uint64_t val = 0;
        for (int byte_idx = 0; byte_idx < 8; byte_idx++) {
            int src = (5 - limb) * 8 + (7 - byte_idx);
            if (src < 48)
                val |= (uint64_t)data[src] << (byte_idx * 8);
        }
        r.limbs[limb] = val;
    }
    return r;
}

__device__ static bool decompress_g1(uint384 x_raw, bool y_positive,
                                     uint384& x_mont, uint384& y_mont) {
    x_mont = to_mont(x_raw);

    uint384 x2 = fp_sqr(x_mont);
    uint384 x3 = fp_mul(x2, x_mont);
    uint384 b_mont = to_mont(uint384{{4, 0, 0, 0, 0, 0}});
    uint384 y2 = fp_add(x3, b_mont);

    // sqrt via a^((p+1)/4) since p = 3 mod 4
    uint384 exp = {{
        0xEE7FBFFFFFFFEAAFULL,
        0x07AAFFFFAC54FFFFULL,
        0xD9CC34A83DAC3D89ULL,
        0xD91DD2E13CE144AFULL,
        0x92C6E9ED90D2EB35ULL,
        0x0680447A8E5FF9A6ULL
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

    uint384 check = fp_sqr(y_cand);
    if (u384_cmp(check, y2) != 0)
        return false;

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

extern "C" __global__ void bls_verify_batch(
    const BLSSignature* __restrict__  sigs,
    const BLSPublicKey* __restrict__  pubkeys,
    const BLSMessage* __restrict__    messages,
    uint32_t* __restrict__            results,
    const uint32_t                    num_sigs)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_sigs) return;

    const uint8_t* sig_data = sigs[tid].data;

    uint8_t flags = sig_data[0];
    bool compressed = (flags & 0x80) != 0;
    bool infinity = (flags & 0x40) != 0;
    bool y_sign = (flags & 0x20) != 0;

    if (infinity) {
        results[tid] = 0;
        return;
    }

    if (!compressed) {
        results[tid] = 0;
        return;
    }

    uint8_t clean_data[48];
    for (int i = 0; i < 48; i++) clean_data[i] = sig_data[i];
    clean_data[0] &= 0x1F;

    uint384 x_raw = {};
    for (int limb = 0; limb < 6; limb++) {
        uint64_t val = 0;
        for (int b = 0; b < 8; b++) {
            int src = (5 - limb) * 8 + (7 - b);
            if (src < 48)
                val |= (uint64_t)clean_data[src] << (b * 8);
        }
        x_raw.limbs[limb] = val;
    }

    uint384 sig_x, sig_y;
    bool on_curve = decompress_g1(x_raw, !y_sign, sig_x, sig_y);
    if (!on_curve) {
        results[tid] = 0;
        return;
    }

    results[tid] = 0x3;  // on_curve=1, needs_subgroup_check=1
}

// =============================================================================
// BLS G1 Aggregation kernel
// =============================================================================

extern "C" __global__ void bls_aggregate_g1(
    const BLSSignature* __restrict__  sigs,
    uint8_t* __restrict__             agg_out,
    uint32_t* __restrict__            counter,
    const uint32_t                    num_sigs)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lid = threadIdx.x;
    uint32_t tgid = blockIdx.x;
    uint32_t tg_size = blockDim.x;

    // Each thread deserializes and decompresses one signature
    G1Point local_sum = g1_identity();

    if (tid < num_sigs) {
        const uint8_t* sig_data = sigs[tid].data;

        uint8_t flags = sig_data[0];
        bool infinity = (flags & 0x40) != 0;
        bool y_sign = (flags & 0x20) != 0;

        if (!infinity) {
            uint8_t clean_data[48];
            for (int i = 0; i < 48; i++) clean_data[i] = sig_data[i];
            clean_data[0] &= 0x1F;

            uint384 x_raw = {};
            for (int limb = 0; limb < 6; limb++) {
                uint64_t val = 0;
                for (int b = 0; b < 8; b++) {
                    int src = (5 - limb) * 8 + (7 - b);
                    if (src < 48)
                        val |= (uint64_t)clean_data[src] << (b * 8);
                }
                x_raw.limbs[limb] = val;
            }

            uint384 sx, sy;
            if (decompress_g1(x_raw, !y_sign, sx, sy)) {
                local_sum.x = sx;
                local_sum.y = sy;
                local_sum.z = BLS_R;
            }
        }
    }

    // Threadgroup reduction via shared memory
    __shared__ uint384 shared_x[256];
    __shared__ uint384 shared_y[256];
    __shared__ uint384 shared_z[256];

    shared_x[lid] = local_sum.x;
    shared_y[lid] = local_sum.y;
    shared_z[lid] = local_sum.z;
    __syncthreads();

    // Binary reduction
    for (uint32_t stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            G1Point a;
            a.x = shared_x[lid]; a.y = shared_y[lid]; a.z = shared_z[lid];

            G1Point b;
            b.x = shared_x[lid + stride]; b.y = shared_y[lid + stride]; b.z = shared_z[lid + stride];

            if (!g1_is_infinity(b)) {
                if (g1_is_infinity(a)) {
                    a = b;
                } else {
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
        __syncthreads();
    }

    // Thread 0 of each block writes partial result
    if (lid == 0) {
        G1Point partial;
        partial.x = shared_x[0]; partial.y = shared_y[0]; partial.z = shared_z[0];

        uint384 ax, ay;
        g1_to_affine(partial, ax, ay);

        uint384 ax_norm = from_mont(ax);
        uint384 ay_norm = from_mont(ay);

        uint32_t tg_offset = tgid * 96;
        for (int limb = 0; limb < 6; limb++) {
            for (int b = 0; b < 8; b++) {
                int dst = (5 - limb) * 8 + (7 - b);
                if (dst < 48) {
                    agg_out[tg_offset + dst] = (uint8_t)((ax_norm.limbs[limb] >> (b * 8)) & 0xFF);
                    agg_out[tg_offset + 48 + dst] = (uint8_t)((ay_norm.limbs[limb] >> (b * 8)) & 0xFF);
                }
            }
        }

        atomicAdd(counter, 1u);
    }
}
