// secp256k1 ECDSA batch recovery — CUDA implementation
// Matches secp256k1_recover.metal output byte-for-byte
// One thread per ecrecover (r, s, v, msg_hash) → 20-byte Ethereum address

#include <cstdint>

#ifndef __CUDA_ARCH__
#define __device__
#define __global__
#define __shared__
struct dim3 { unsigned x, y, z; };
static dim3 blockIdx, blockDim, threadIdx;
#endif

// =============================================================================
// 256-bit unsigned integer (4 x 64-bit limbs, little-endian)
// =============================================================================

struct uint256 {
    uint64_t limbs[4];
};

// =============================================================================
// secp256k1 constants
// =============================================================================

__device__ static const uint256 SECP256K1_P = {{
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
}};

__device__ static const uint256 SECP256K1_N = {{
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
}};

__device__ static const uint256 GX = {{
    0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL
}};

__device__ static const uint256 GY = {{
    0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL
}};

__device__ static const uint256 MONT_R_P = {{
    0x00000001000003D1ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL
}};

__device__ static const uint256 MONT_R2_P = {{
    0x000007A2000E90A1ULL, 0x0000000000000001ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL
}};

__device__ static const uint64_t P_INV = 0xD838091DD2253531ULL;

__device__ static const uint256 MONT_R2_N = {{
    0x896CF21467D7D140ULL, 0x741496C20E7CF878ULL,
    0xE697F5E45BCD07C6ULL, 0x9D671CD581C69BC5ULL
}};

__device__ static const uint64_t N_INV = 0x4B0DFF665588B13FULL;

__device__ static const uint256 ZERO256 = {{0, 0, 0, 0}};
__device__ static const uint256 ONE256 = {{1, 0, 0, 0}};

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
// Montgomery arithmetic (parameterized by modulus m and inv = -m^(-1) mod 2^64)
// Uses __int128 on CUDA for 64x64->128 multiply
// =============================================================================

__device__ static uint256 mont_reduce(uint64_t t[8], uint256 m, uint64_t inv) {
    uint64_t a[9];
    for (int i = 0; i < 8; i++) a[i] = t[i];
    a[8] = 0;

    for (int i = 0; i < 4; i++) {
        uint64_t u = a[i] * inv;

        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
#ifdef __CUDA_ARCH__
            unsigned __int128 prod = (unsigned __int128)u * m.limbs[j];
            unsigned __int128 acc = prod + carry + a[i + j];
            a[i + j] = (uint64_t)acc;
            carry = (uint64_t)(acc >> 64);
#else
            uint64_t u_lo = u & 0xFFFFFFFFULL;
            uint64_t u_hi = u >> 32;
            uint64_t m_lo = m.limbs[j] & 0xFFFFFFFFULL;
            uint64_t m_hi = m.limbs[j] >> 32;
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
        for (int j = 4; i + j <= 8; j++) {
            uint64_t sum = a[i + j] + carry;
            carry = (sum < a[i + j]) ? 1ULL : 0ULL;
            a[i + j] = sum;
            if (carry == 0) break;
        }
    }

    uint256 r;
    r.limbs[0] = a[4];
    r.limbs[1] = a[5];
    r.limbs[2] = a[6];
    r.limbs[3] = a[7];

    if (a[8] || u256_cmp(r, m) >= 0) {
        uint64_t bw;
        r = u256_sub(r, m, bw);
    }
    return r;
}

__device__ static uint256 mont_mul(uint256 a, uint256 b, uint256 m, uint64_t inv) {
    uint64_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
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
        for (int j = 4; i + j < 8; j++) {
            uint64_t sum = t[i + j] + carry;
            carry = (sum < t[i + j]) ? 1ULL : 0ULL;
            t[i + j] = sum;
            if (carry == 0) break;
        }
    }

    return mont_reduce(t, m, inv);
}

__device__ static uint256 to_mont(uint256 a, uint256 r2, uint256 m, uint64_t inv) {
    return mont_mul(a, r2, m, inv);
}

__device__ static uint256 from_mont(uint256 a, uint256 m, uint64_t inv) {
    uint64_t t[8] = {a.limbs[0], a.limbs[1], a.limbs[2], a.limbs[3], 0, 0, 0, 0};
    return mont_reduce(t, m, inv);
}

// Field operations over p (Montgomery form)
__device__ static uint256 fp_add(uint256 a, uint256 b) {
    uint64_t carry;
    uint256 r = u256_add(a, b, carry);
    if (carry || u256_cmp(r, SECP256K1_P) >= 0) {
        uint64_t bw;
        r = u256_sub(r, SECP256K1_P, bw);
    }
    return r;
}

__device__ static uint256 fp_sub(uint256 a, uint256 b) {
    uint64_t bw;
    uint256 r = u256_sub(a, b, bw);
    if (bw) {
        uint64_t c;
        r = u256_add(r, SECP256K1_P, c);
    }
    return r;
}

__device__ static uint256 fp_mul(uint256 a, uint256 b) {
    return mont_mul(a, b, SECP256K1_P, P_INV);
}

__device__ static uint256 fp_sqr(uint256 a) {
    return fp_mul(a, a);
}

// Scalar field operations over n
__device__ static uint256 fn_mul(uint256 a, uint256 b) {
    return mont_mul(a, b, SECP256K1_N, N_INV);
}

// Fermat inversion over p
__device__ static uint256 fp_inv(uint256 a) {
    uint256 result = to_mont(ONE256, MONT_R2_P, SECP256K1_P, P_INV);
    uint256 base = a;

    uint64_t exp[4] = {
        0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    };

    for (int i = 0; i < 4; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((exp[i] >> bit) & 1) {
                result = fp_mul(result, base);
            }
            base = fp_sqr(base);
        }
    }
    return result;
}

// Scalar inversion over n
__device__ static uint256 fn_inv(uint256 a) {
    uint64_t exp[4] = {
        0xBFD25E8CD036413FULL, 0xBAAEDCE6AF48A03BULL,
        0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
    };

    uint256 result = to_mont(ONE256, MONT_R2_N, SECP256K1_N, N_INV);
    uint256 base = a;

    for (int i = 0; i < 4; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((exp[i] >> bit) & 1) {
                result = fn_mul(result, base);
            }
            base = fn_mul(base, base);
        }
    }
    return result;
}

// =============================================================================
// secp256k1 EC point operations (Jacobian coordinates, Montgomery F_p)
// =============================================================================

struct ECPoint {
    uint256 x, y, z;
};

__device__ static ECPoint ec_identity() {
    ECPoint p;
    p.x = to_mont(ONE256, MONT_R2_P, SECP256K1_P, P_INV);
    p.y = to_mont(ONE256, MONT_R2_P, SECP256K1_P, P_INV);
    p.z = ZERO256;
    return p;
}

__device__ static bool ec_is_infinity(ECPoint p) {
    return u256_is_zero(p.z);
}

__device__ static ECPoint ec_double(ECPoint p) {
    if (ec_is_infinity(p)) return p;

    uint256 A = fp_sqr(p.y);
    uint256 B = fp_mul(p.x, A);
    uint256 C = fp_sqr(A);

    uint256 S = fp_add(B, B);
    S = fp_add(S, S);

    uint256 X2 = fp_sqr(p.x);
    uint256 M = fp_add(X2, fp_add(X2, X2));

    uint256 X3 = fp_sub(fp_sqr(M), fp_add(S, S));

    uint256 C8 = fp_add(C, C);
    C8 = fp_add(C8, C8);
    C8 = fp_add(C8, C8);
    uint256 Y3 = fp_sub(fp_mul(M, fp_sub(S, X3)), C8);

    uint256 Z3 = fp_mul(p.y, p.z);
    Z3 = fp_add(Z3, Z3);

    ECPoint r;
    r.x = X3;
    r.y = Y3;
    r.z = Z3;
    return r;
}

__device__ static ECPoint ec_add_mixed(ECPoint P, uint256 Qx, uint256 Qy) {
    if (ec_is_infinity(P)) {
        ECPoint r;
        r.x = Qx;
        r.y = Qy;
        r.z = to_mont(ONE256, MONT_R2_P, SECP256K1_P, P_INV);
        return r;
    }

    uint256 Z2 = fp_sqr(P.z);
    uint256 U2 = fp_mul(Qx, Z2);
    uint256 Z3 = fp_mul(Z2, P.z);
    uint256 S2 = fp_mul(Qy, Z3);

    uint256 H = fp_sub(U2, P.x);
    uint256 R = fp_sub(S2, P.y);

    if (u256_is_zero(H)) {
        if (u256_is_zero(R)) {
            return ec_double(P);
        }
        return ec_identity();
    }

    uint256 H2 = fp_sqr(H);
    uint256 H3 = fp_mul(H, H2);
    uint256 U1H2 = fp_mul(P.x, H2);

    uint256 X3 = fp_sub(fp_sub(fp_sqr(R), H3), fp_add(U1H2, U1H2));
    uint256 Y3 = fp_sub(fp_mul(R, fp_sub(U1H2, X3)), fp_mul(P.y, H3));
    uint256 Zr = fp_mul(H, P.z);

    ECPoint res;
    res.x = X3;
    res.y = Y3;
    res.z = Zr;
    return res;
}

__device__ static void ec_to_affine(ECPoint p, uint256& ax, uint256& ay) {
    if (ec_is_infinity(p)) {
        ax = ZERO256;
        ay = ZERO256;
        return;
    }
    uint256 z_inv = fp_inv(p.z);
    uint256 z_inv2 = fp_sqr(z_inv);
    uint256 z_inv3 = fp_mul(z_inv2, z_inv);
    ax = fp_mul(p.x, z_inv2);
    ay = fp_mul(p.y, z_inv3);
}

// NOT constant-time: branches on scalar bits. Safe for ecrecover (all inputs
// are public). MUST NOT be reused for signing where the nonce k is secret.
__device__ static ECPoint ec_mul_affine(uint256 k, uint256 Px, uint256 Py) {
    ECPoint result = ec_identity();

    for (int i = 3; i >= 0; i--) {
        for (int bit = 63; bit >= 0; bit--) {
            result = ec_double(result);
            if ((k.limbs[i] >> bit) & 1) {
                result = ec_add_mixed(result, Px, Py);
            }
        }
    }
    return result;
}

// =============================================================================
// Keccak-256 (inline, for address derivation)
// =============================================================================

__device__ static const uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

__device__ static const int KECCAK_PI_LANE[24] = {
    10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
    15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
};

__device__ static const int KECCAK_RHO[24] = {
     1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
};

__device__ static uint64_t keccak_rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__ static void keccak_f1600(uint64_t st[25]) {
    for (int round = 0; round < 24; ++round) {
        uint64_t C[5];
        for (int x = 0; x < 5; ++x)
            C[x] = st[x] ^ st[x + 5] ^ st[x + 10] ^ st[x + 15] ^ st[x + 20];
        for (int x = 0; x < 5; ++x) {
            uint64_t d = C[(x + 4) % 5] ^ keccak_rotl64(C[(x + 1) % 5], 1);
            for (int y = 0; y < 5; ++y)
                st[x + 5 * y] ^= d;
        }
        uint64_t t = st[1];
        for (int i = 0; i < 24; ++i) {
            uint64_t tmp = st[KECCAK_PI_LANE[i]];
            st[KECCAK_PI_LANE[i]] = keccak_rotl64(t, KECCAK_RHO[i]);
            t = tmp;
        }
        for (int y = 0; y < 5; ++y) {
            uint64_t row[5];
            for (int x = 0; x < 5; ++x) row[x] = st[x + 5 * y];
            for (int x = 0; x < 5; ++x)
                st[x + 5 * y] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
        }
        st[0] ^= KECCAK_RC[round];
    }
}

__device__ static void keccak256_64(const uint8_t data[64], uint8_t out[32]) {
    uint64_t state[25] = {};

    for (uint32_t w = 0; w < 8; ++w) {
        uint64_t lane = 0;
        for (uint32_t b = 0; b < 8; ++b)
            lane |= (uint64_t)data[w * 8 + b] << (b * 8);
        state[w] ^= lane;
    }

    state[8] ^= 0x01ULL;
    state[16] ^= 0x80ULL << 56;

    keccak_f1600(state);

    for (uint32_t w = 0; w < 4; ++w) {
        uint64_t lane = state[w];
        for (uint32_t b = 0; b < 8; ++b)
            out[w * 8 + b] = (uint8_t)(lane >> (b * 8));
    }
}

// =============================================================================
// Input/Output structures
// =============================================================================

struct EcrecoverInput {
    uint8_t r[32];
    uint8_t s[32];
    uint8_t v;
    uint8_t _pad[3];
    uint8_t msg_hash[32];
    uint8_t _pad2[28];
};

struct EcrecoverOutput {
    uint8_t address[20];
    uint8_t valid;
    uint8_t _pad[11];
};

// =============================================================================
// Helpers: big-endian load/store
// =============================================================================

__device__ static uint256 load_be32(const uint8_t bytes[32]) {
    uint256 r;
    for (int limb = 0; limb < 4; limb++) {
        uint64_t v = 0;
        int base = (3 - limb) * 8;
        for (int b = 0; b < 8; b++) {
            v = (v << 8) | (uint64_t)bytes[base + b];
        }
        r.limbs[limb] = v;
    }
    return r;
}

__device__ static void store_be32(uint256 val, uint8_t bytes[32]) {
    for (int limb = 0; limb < 4; limb++) {
        int base = (3 - limb) * 8;
        uint64_t v = val.limbs[limb];
        for (int b = 7; b >= 0; b--) {
            bytes[base + b] = (uint8_t)(v & 0xFF);
            v >>= 8;
        }
    }
}

// =============================================================================
// Main kernel: batch secp256k1 ecrecover
// =============================================================================

extern "C" __global__ void secp256k1_ecrecover_batch(
    const EcrecoverInput* __restrict__  inputs,
    EcrecoverOutput* __restrict__       outputs,
    const uint32_t                      num_sigs)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_sigs) return;

    const EcrecoverInput& inp = inputs[tid];
    EcrecoverOutput& out = outputs[tid];

    // Clear output
    for (int i = 0; i < 20; i++) out.address[i] = 0;
    out.valid = 0;
    for (int i = 0; i < 11; i++) out._pad[i] = 0;

    // Load r, s, v, hash
    uint256 r = load_be32(inp.r);
    uint256 s = load_be32(inp.s);
    uint256 e = load_be32(inp.msg_hash);
    uint32_t v = (uint32_t)inp.v;

    // Normalize v
    if (v >= 27) v -= 27;
    if (v >= 2) v = v % 2;

    // Validate
    if (u256_is_zero(r) || u256_cmp(r, SECP256K1_N) >= 0) return;
    if (u256_is_zero(s) || u256_cmp(s, SECP256K1_N) >= 0) return;
    if (v > 1) return;

    // Step 1: Decompress r → R = (r, y) on secp256k1
    uint256 r_mont = to_mont(r, MONT_R2_P, SECP256K1_P, P_INV);
    uint256 r2 = fp_sqr(r_mont);
    uint256 r3 = fp_mul(r2, r_mont);
    uint256 seven_mont = to_mont(uint256{{7, 0, 0, 0}}, MONT_R2_P, SECP256K1_P, P_INV);
    uint256 y2 = fp_add(r3, seven_mont);

    // sqrt(y2) via a^((p+1)/4) since p = 3 mod 4
    uint256 y_mont;
    {
        uint64_t exp[4] = {
            0xFFFFFFFFBFFFFF0CULL, 0xFFFFFFFFFFFFFFFFULL,
            0xFFFFFFFFFFFFFFFFULL, 0x3FFFFFFFFFFFFFFFULL
        };
        uint256 result = to_mont(ONE256, MONT_R2_P, SECP256K1_P, P_INV);
        uint256 base = y2;
        for (int i = 0; i < 4; i++) {
            for (int bit = 0; bit < 64; bit++) {
                if ((exp[i] >> bit) & 1) {
                    result = fp_mul(result, base);
                }
                base = fp_sqr(base);
            }
        }
        y_mont = result;
    }

    // Verify sqrt exists
    if (u256_cmp(fp_sqr(y_mont), y2) != 0) return;

    // Select correct y parity
    uint256 y_normal = from_mont(y_mont, SECP256K1_P, P_INV);
    bool y_is_odd = (y_normal.limbs[0] & 1) != 0;
    if ((v == 0 && y_is_odd) || (v == 1 && !y_is_odd)) {
        y_mont = fp_sub(ZERO256, y_mont);
    }

    uint256 Rx_mont = r_mont;
    uint256 Ry_mont = y_mont;

    // Step 2: r_inv = r^(-1) mod n
    uint256 r_n_mont = to_mont(r, MONT_R2_N, SECP256K1_N, N_INV);
    uint256 r_inv_mont = fn_inv(r_n_mont);

    // Step 3: u1 = -(e * r_inv) mod n, u2 = s * r_inv mod n
    uint256 e_n_mont = to_mont(e, MONT_R2_N, SECP256K1_N, N_INV);
    uint256 s_n_mont = to_mont(s, MONT_R2_N, SECP256K1_N, N_INV);

    uint256 u1_mont = fn_mul(e_n_mont, r_inv_mont);
    uint256 u1 = from_mont(u1_mont, SECP256K1_N, N_INV);
    if (!u256_is_zero(u1)) {
        uint64_t bw;
        u1 = u256_sub(SECP256K1_N, u1, bw);
    }

    uint256 u2 = from_mont(fn_mul(s_n_mont, r_inv_mont), SECP256K1_N, N_INV);

    // Step 4: Q = u1*G + u2*R
    uint256 Gx_mont = to_mont(GX, MONT_R2_P, SECP256K1_P, P_INV);
    uint256 Gy_mont = to_mont(GY, MONT_R2_P, SECP256K1_P, P_INV);

    ECPoint Q1 = ec_mul_affine(u1, Gx_mont, Gy_mont);
    ECPoint Q2 = ec_mul_affine(u2, Rx_mont, Ry_mont);

    // Add Q1 + Q2
    ECPoint Q;
    if (ec_is_infinity(Q1)) {
        Q = Q2;
    } else if (ec_is_infinity(Q2)) {
        Q = Q1;
    } else {
        uint256 Q2x_aff, Q2y_aff;
        ec_to_affine(Q2, Q2x_aff, Q2y_aff);
        Q = ec_add_mixed(Q1, Q2x_aff, Q2y_aff);
    }

    if (ec_is_infinity(Q)) return;

    // Step 5: Convert Q to affine, serialize big-endian
    uint256 Qx_aff, Qy_aff;
    ec_to_affine(Q, Qx_aff, Qy_aff);

    uint256 Qx_norm = from_mont(Qx_aff, SECP256K1_P, P_INV);
    uint256 Qy_norm = from_mont(Qy_aff, SECP256K1_P, P_INV);

    uint8_t pubkey[64];
    store_be32(Qx_norm, pubkey);
    store_be32(Qy_norm, pubkey + 32);

    // Step 6: address = keccak256(pubkey)[12:]
    uint8_t hash[32];
    keccak256_64(pubkey, hash);

    for (int i = 0; i < 20; i++) {
        out.address[i] = hash[12 + i];
    }
    out.valid = 1;
}
