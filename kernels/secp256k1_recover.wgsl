// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// secp256k1 ECDSA public key recovery (ecrecover) in WGSL.
// Matches secp256k1_recover.metal output byte-for-byte.
// 256-bit arithmetic uses 8 x u32 limbs (no native u64 in WGSL).
//
// Per thread: (r, s, v, msg_hash) -> 20-byte Ethereum address
// Algorithm: Q = r^{-1} * (s*R - e*G), address = keccak256(Q)[12:]

// Input: [r[32], s[32], v[1], pad[3], msg_hash[32], pad[28]] = 128 bytes per sig
// Output: [address[20], valid[1], pad[11]] = 32 bytes per sig

@group(0) @binding(0) var<storage, read> inputs: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputs: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    num_items: u32,
}

// ============================================================================
// 256-bit integer as 8 x u32 (little-endian)
// ============================================================================

fn u256_zero() -> array<u32, 8> {
    return array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
}

fn u256_is_zero(a: ptr<function, array<u32, 8>>) -> bool {
    var acc = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) { acc = acc | (*a)[i]; }
    return acc == 0u;
}

fn u256_cmp(a: ptr<function, array<u32, 8>>, b: ptr<function, array<u32, 8>>) -> i32 {
    for (var i = 7i; i >= 0; i = i - 1) {
        let ui = u32(i);
        if ((*a)[ui] > (*b)[ui]) { return 1; }
        if ((*a)[ui] < (*b)[ui]) { return -1; }
    }
    return 0;
}

fn u256_add(a: ptr<function, array<u32, 8>>, b: ptr<function, array<u32, 8>>,
            r: ptr<function, array<u32, 8>>) -> u32 {
    var c = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let s1 = (*a)[i] + c;
        c = select(0u, 1u, s1 < (*a)[i]);
        let s2 = s1 + (*b)[i];
        c = c + select(0u, 1u, s2 < s1);
        (*r)[i] = s2;
    }
    return c;
}

fn u256_sub(a: ptr<function, array<u32, 8>>, b: ptr<function, array<u32, 8>>,
            r: ptr<function, array<u32, 8>>) -> u32 {
    var bw = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let d1 = (*a)[i] - bw;
        bw = select(0u, 1u, d1 > (*a)[i]);
        let d2 = d1 - (*b)[i];
        bw = bw + select(0u, 1u, d2 > d1);
        (*r)[i] = d2;
    }
    return bw;
}

// ============================================================================
// secp256k1 constants (8 x u32 little-endian)
// ============================================================================

// Field prime p = 0xFFFFFFFF...FFFFFFFEFFFFFC2F
const SECP_P = array<u32, 8>(
    0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
);
// Curve order n
const SECP_N = array<u32, 8>(
    0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
    0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
);
// Montgomery R mod p
const MONT_R_P = array<u32, 8>(
    0x000003D1u, 0x00000001u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u
);
// R^2 mod p
const MONT_R2_P = array<u32, 8>(
    0x000E90A1u, 0x000007A2u, 0x00000001u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u
);
// -p^{-1} mod 2^32
const P_INV: u32 = 0xD2253531u;
// R^2 mod n
const MONT_R2_N = array<u32, 8>(
    0x67D7D140u, 0x896CF214u, 0x0E7CF878u, 0x741496C2u,
    0x5BCD07C6u, 0xE697F5E4u, 0x81C69BC5u, 0x9D671CD5u
);
// -n^{-1} mod 2^32
const N_INV: u32 = 0x5588B13Fu;
// Generator G.x
const GX = array<u32, 8>(
    0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDB,
    0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu
);
// Generator G.y
const GY = array<u32, 8>(
    0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
    0x0E1108A8u, 0x5DA4FBFC, 0x26A3C465u, 0x483ADA77u
);

// ============================================================================
// Montgomery multiplication (256-bit, 8x u32 limbs)
// ============================================================================

fn mont_reduce(t: ptr<function, array<u32, 16>>, m: ptr<function, array<u32, 8>>,
               inv: u32, r: ptr<function, array<u32, 8>>) {
    // Extended to 17 limbs for carry
    var a: array<u32, 17>;
    for (var i = 0u; i < 16u; i = i + 1u) { a[i] = (*t)[i]; }
    a[16] = 0u;

    for (var i = 0u; i < 8u; i = i + 1u) {
        let u = a[i] * inv;
        var carry = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            // u * m[j] -> (hi, lo)
            let u_lo = u & 0xFFFFu; let u_hi = u >> 16u;
            let m_lo = (*m)[j] & 0xFFFFu; let m_hi = (*m)[j] >> 16u;
            let ll = u_lo * m_lo;
            let lh = u_lo * m_hi;
            let hl = u_hi * m_lo;
            let hh = u_hi * m_hi;
            let mid = lh + hl;
            var lo = ll + (mid << 16u);
            var hi = hh + (mid >> 16u) + select(0u, 1u, lo < ll) + select(0u, 0x10000u, mid < lh);

            // lo += carry
            let s1 = lo + carry;
            hi = hi + select(0u, 1u, s1 < lo);
            // a[i+j] += s1
            let s2 = a[i + j] + s1;
            hi = hi + select(0u, 1u, s2 < a[i + j]);
            a[i + j] = s2;
            carry = hi;
        }
        for (var j = 8u; i + j <= 16u; j = j + 1u) {
            let s = a[i + j] + carry;
            carry = select(0u, 1u, s < a[i + j]);
            a[i + j] = s;
            if (carry == 0u) { break; }
        }
    }

    for (var i = 0u; i < 8u; i = i + 1u) { (*r)[i] = a[i + 8u]; }

    // Final subtraction if r >= m
    if (a[16] != 0u || u256_cmp(r, m) >= 0) {
        let _ = u256_sub(r, m, r);
    }
}

fn mont_mul(a: ptr<function, array<u32, 8>>, b: ptr<function, array<u32, 8>>,
            m: ptr<function, array<u32, 8>>, inv: u32, r: ptr<function, array<u32, 8>>) {
    var t: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { t[i] = 0u; }

    for (var i = 0u; i < 8u; i = i + 1u) {
        var carry = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let al = (*a)[i] & 0xFFFFu; let ah = (*a)[i] >> 16u;
            let bl = (*b)[j] & 0xFFFFu; let bh = (*b)[j] >> 16u;
            let ll = al * bl;
            let lh = al * bh;
            let hl = ah * bl;
            let hh = ah * bh;
            let mid = lh + hl;
            var lo = ll + (mid << 16u);
            var hi = hh + (mid >> 16u) + select(0u, 1u, lo < ll) + select(0u, 0x10000u, mid < lh);
            let s1 = lo + carry; hi = hi + select(0u, 1u, s1 < lo);
            let s2 = t[i + j] + s1; hi = hi + select(0u, 1u, s2 < t[i + j]);
            t[i + j] = s2;
            carry = hi;
        }
        for (var j = 8u; i + j < 16u; j = j + 1u) {
            let s = t[i + j] + carry;
            carry = select(0u, 1u, s < t[i + j]);
            t[i + j] = s;
            if (carry == 0u) { break; }
        }
    }
    mont_reduce(&t, m, inv, r);
}

// Field ops over p (Montgomery form)
fn fp_add(a: ptr<function, array<u32, 8>>, b: ptr<function, array<u32, 8>>,
          r: ptr<function, array<u32, 8>>) {
    var p = SECP_P;
    let c = u256_add(a, b, r);
    if (c != 0u || u256_cmp(r, &p) >= 0) {
        let _ = u256_sub(r, &p, r);
    }
}

fn fp_sub(a: ptr<function, array<u32, 8>>, b: ptr<function, array<u32, 8>>,
          r: ptr<function, array<u32, 8>>) {
    var p = SECP_P;
    let bw = u256_sub(a, b, r);
    if (bw != 0u) {
        let _ = u256_add(r, &p, r);
    }
}

fn fp_mul(a: ptr<function, array<u32, 8>>, b: ptr<function, array<u32, 8>>,
          r: ptr<function, array<u32, 8>>) {
    var p = SECP_P;
    mont_mul(a, b, &p, P_INV, r);
}

fn fp_sqr(a: ptr<function, array<u32, 8>>, r: ptr<function, array<u32, 8>>) {
    var p = SECP_P;
    mont_mul(a, a, &p, P_INV, r);
}

fn fn_mul(a: ptr<function, array<u32, 8>>, b: ptr<function, array<u32, 8>>,
          r: ptr<function, array<u32, 8>>) {
    var n = SECP_N;
    mont_mul(a, b, &n, N_INV, r);
}

fn to_mont_p(a: ptr<function, array<u32, 8>>, r: ptr<function, array<u32, 8>>) {
    var r2 = MONT_R2_P;
    fp_mul(a, &r2, r);
}

fn from_mont_p(a: ptr<function, array<u32, 8>>, r: ptr<function, array<u32, 8>>) {
    var p = SECP_P;
    var t: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { t[i] = 0u; }
    for (var i = 0u; i < 8u; i = i + 1u) { t[i] = (*a)[i]; }
    mont_reduce(&t, &p, P_INV, r);
}

fn to_mont_n(a: ptr<function, array<u32, 8>>, r: ptr<function, array<u32, 8>>) {
    var r2 = MONT_R2_N;
    fn_mul(a, &r2, r);
}

fn from_mont_n(a: ptr<function, array<u32, 8>>, r: ptr<function, array<u32, 8>>) {
    var n = SECP_N;
    var t: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { t[i] = 0u; }
    for (var i = 0u; i < 8u; i = i + 1u) { t[i] = (*a)[i]; }
    mont_reduce(&t, &n, N_INV, r);
}

// Modular inversion via Fermat: a^(m-2) mod m
fn fp_inv(a: ptr<function, array<u32, 8>>, r: ptr<function, array<u32, 8>>) {
    // p-2 little-endian u32 limbs
    var exp = array<u32, 8>(
        0xFFFFFC2Du, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    );
    var one = array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    var result: array<u32, 8>;
    to_mont_p(&one, &result);
    var base: array<u32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) { base[i] = (*a)[i]; }

    for (var i = 0u; i < 8u; i = i + 1u) {
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            if (((exp[i] >> bit) & 1u) != 0u) {
                var tmp: array<u32, 8>;
                fp_mul(&result, &base, &tmp);
                result = tmp;
            }
            var tmp2: array<u32, 8>;
            fp_sqr(&base, &tmp2);
            base = tmp2;
        }
    }
    *r = result;
}

fn fn_inv(a: ptr<function, array<u32, 8>>, r: ptr<function, array<u32, 8>>) {
    // n-2
    var exp = array<u32, 8>(
        0xD036413Fu, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
        0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    );
    var one = array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    var result: array<u32, 8>;
    to_mont_n(&one, &result);
    var base: array<u32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) { base[i] = (*a)[i]; }

    for (var i = 0u; i < 8u; i = i + 1u) {
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            if (((exp[i] >> bit) & 1u) != 0u) {
                var tmp: array<u32, 8>;
                fn_mul(&result, &base, &tmp);
                result = tmp;
            }
            var tmp2: array<u32, 8>;
            fn_mul(&base, &base, &tmp2);
            base = tmp2;
        }
    }
    *r = result;
}

// ============================================================================
// EC point operations (Jacobian, Montgomery Fp)
// Point = (x[8], y[8], z[8]) = 24 u32 words
// ============================================================================

struct ECPoint {
    x: array<u32, 8>,
    y: array<u32, 8>,
    z: array<u32, 8>,
}

fn ec_identity() -> ECPoint {
    var p: ECPoint;
    var one = array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    to_mont_p(&one, &p.x);
    p.y = p.x;
    p.z = u256_zero();
    return p;
}

fn ec_is_inf(p: ptr<function, ECPoint>) -> bool {
    var z = (*p).z;
    return u256_is_zero(&z);
}

fn ec_double(p: ptr<function, ECPoint>, r: ptr<function, ECPoint>) {
    if (ec_is_inf(p)) { *r = *p; return; }
    var A: array<u32, 8>; fp_sqr(&(*p).y, &A);
    var B: array<u32, 8>; fp_mul(&(*p).x, &A, &B);
    var C: array<u32, 8>; fp_sqr(&A, &C);
    // S = 4*B
    var S: array<u32, 8>; fp_add(&B, &B, &S); fp_add(&S, &S, &S);
    // M = 3*X^2 (a=0)
    var X2: array<u32, 8>; fp_sqr(&(*p).x, &X2);
    var X2_2: array<u32, 8>; fp_add(&X2, &X2, &X2_2);
    var M: array<u32, 8>; fp_add(&X2_2, &X2, &M);
    // X3 = M^2 - 2S
    var M2: array<u32, 8>; fp_sqr(&M, &M2);
    var S2: array<u32, 8>; fp_add(&S, &S, &S2);
    var X3: array<u32, 8>; fp_sub(&M2, &S2, &X3);
    // Y3 = M*(S-X3) - 8C
    var SX: array<u32, 8>; fp_sub(&S, &X3, &SX);
    var MSX: array<u32, 8>; fp_mul(&M, &SX, &MSX);
    var C2: array<u32, 8>; fp_add(&C, &C, &C2);
    var C4: array<u32, 8>; fp_add(&C2, &C2, &C4);
    var C8: array<u32, 8>; fp_add(&C4, &C4, &C8);
    var Y3: array<u32, 8>; fp_sub(&MSX, &C8, &Y3);
    // Z3 = 2*Y*Z
    var YZ: array<u32, 8>; fp_mul(&(*p).y, &(*p).z, &YZ);
    var Z3: array<u32, 8>; fp_add(&YZ, &YZ, &Z3);
    (*r).x = X3; (*r).y = Y3; (*r).z = Z3;
}

fn ec_add_mixed(P: ptr<function, ECPoint>, Qx: ptr<function, array<u32, 8>>,
                Qy: ptr<function, array<u32, 8>>, r: ptr<function, ECPoint>) {
    if (ec_is_inf(P)) {
        (*r).x = *Qx; (*r).y = *Qy;
        var one = array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
        to_mont_p(&one, &(*r).z);
        return;
    }
    var Z2: array<u32, 8>; fp_sqr(&(*P).z, &Z2);
    var U2: array<u32, 8>; fp_mul(Qx, &Z2, &U2);
    var Z3: array<u32, 8>; fp_mul(&Z2, &(*P).z, &Z3);
    var S2: array<u32, 8>; fp_mul(Qy, &Z3, &S2);
    var H: array<u32, 8>; fp_sub(&U2, &(*P).x, &H);
    var R: array<u32, 8>; fp_sub(&S2, &(*P).y, &R);

    if (u256_is_zero(&H)) {
        if (u256_is_zero(&R)) { ec_double(P, r); return; }
        *r = ec_identity();
        return;
    }

    var H2: array<u32, 8>; fp_sqr(&H, &H2);
    var H3: array<u32, 8>; fp_mul(&H, &H2, &H3);
    var U1H2: array<u32, 8>; fp_mul(&(*P).x, &H2, &U1H2);
    // X3 = R^2 - H^3 - 2*U1H2
    var R2: array<u32, 8>; fp_sqr(&R, &R2);
    var U1H2_2: array<u32, 8>; fp_add(&U1H2, &U1H2, &U1H2_2);
    var t1: array<u32, 8>; fp_sub(&R2, &H3, &t1);
    var X3: array<u32, 8>; fp_sub(&t1, &U1H2_2, &X3);
    // Y3 = R*(U1H2 - X3) - Y1*H3
    var UX: array<u32, 8>; fp_sub(&U1H2, &X3, &UX);
    var RUX: array<u32, 8>; fp_mul(&R, &UX, &RUX);
    var YH3: array<u32, 8>; fp_mul(&(*P).y, &H3, &YH3);
    var Y3: array<u32, 8>; fp_sub(&RUX, &YH3, &Y3);
    // Z3 = H * P.Z
    var Zr: array<u32, 8>; fp_mul(&H, &(*P).z, &Zr);
    (*r).x = X3; (*r).y = Y3; (*r).z = Zr;
}

fn ec_mul_affine(k: ptr<function, array<u32, 8>>,
                 Px: ptr<function, array<u32, 8>>,
                 Py: ptr<function, array<u32, 8>>) -> ECPoint {
    var result = ec_identity();
    for (var i = 7i; i >= 0; i = i - 1) {
        for (var bit = 31i; bit >= 0; bit = bit - 1) {
            var dbl: ECPoint;
            ec_double(&result, &dbl);
            result = dbl;
            if ((((*k)[u32(i)] >> u32(bit)) & 1u) != 0u) {
                var tmp: ECPoint;
                ec_add_mixed(&result, Px, Py, &tmp);
                result = tmp;
            }
        }
    }
    return result;
}

fn ec_to_affine(p: ptr<function, ECPoint>, ax: ptr<function, array<u32, 8>>,
                ay: ptr<function, array<u32, 8>>) {
    if (ec_is_inf(p)) { *ax = u256_zero(); *ay = u256_zero(); return; }
    var z_inv: array<u32, 8>; fp_inv(&(*p).z, &z_inv);
    var z_inv2: array<u32, 8>; fp_sqr(&z_inv, &z_inv2);
    var z_inv3: array<u32, 8>; fp_mul(&z_inv2, &z_inv, &z_inv3);
    fp_mul(&(*p).x, &z_inv2, ax);
    fp_mul(&(*p).y, &z_inv3, ay);
}

// ============================================================================
// Inline Keccak-256 for 64 bytes (public key -> address)
// ============================================================================

var<private> kst_lo: array<u32, 25>;
var<private> kst_hi: array<u32, 25>;

const KRC_LO = array<u32, 24>(
    0x00000001u, 0x00008082u, 0x0000808Au, 0x80008000u,
    0x0000808Bu, 0x80000001u, 0x80008081u, 0x00008009u,
    0x0000008Au, 0x00000088u, 0x80008009u, 0x8000000Au,
    0x8000808Bu, 0x0000008Bu, 0x00008089u, 0x00008003u,
    0x00008002u, 0x00000080u, 0x0000800Au, 0x8000000Au,
    0x80008081u, 0x00008080u, 0x80000001u, 0x80008008u
);
const KRC_HI = array<u32, 24>(
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x80000000u, 0x80000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u
);
const KPI = array<u32, 24>(
    10u, 7u, 11u, 17u, 18u, 3u, 5u, 16u, 8u, 21u, 24u, 4u,
    15u, 23u, 19u, 13u, 12u, 2u, 20u, 14u, 22u, 9u, 6u, 1u
);
const KRHO = array<u32, 24>(
    1u, 3u, 6u, 10u, 15u, 21u, 28u, 36u, 45u, 55u, 2u, 14u,
    27u, 41u, 56u, 8u, 25u, 43u, 62u, 18u, 39u, 61u, 20u, 44u
);

fn krotl64(lo: u32, hi: u32, n: u32) -> vec2<u32> {
    if (n == 0u) { return vec2<u32>(lo, hi); }
    if (n == 32u) { return vec2<u32>(hi, lo); }
    if (n < 32u) {
        return vec2<u32>((lo << n) | (hi >> (32u - n)), (hi << n) | (lo >> (32u - n)));
    }
    let m = n - 32u;
    return vec2<u32>((hi << m) | (lo >> (32u - m)), (lo << m) | (hi >> (32u - m)));
}

fn keccak_f() {
    for (var round = 0u; round < 24u; round = round + 1u) {
        var c_lo: array<u32, 5>; var c_hi: array<u32, 5>;
        for (var x = 0u; x < 5u; x = x + 1u) {
            c_lo[x] = kst_lo[x] ^ kst_lo[x+5u] ^ kst_lo[x+10u] ^ kst_lo[x+15u] ^ kst_lo[x+20u];
            c_hi[x] = kst_hi[x] ^ kst_hi[x+5u] ^ kst_hi[x+10u] ^ kst_hi[x+15u] ^ kst_hi[x+20u];
        }
        for (var x = 0u; x < 5u; x = x + 1u) {
            let r = krotl64(c_lo[(x+1u)%5u], c_hi[(x+1u)%5u], 1u);
            let d_lo = c_lo[(x+4u)%5u] ^ r.x;
            let d_hi = c_hi[(x+4u)%5u] ^ r.y;
            for (var y = 0u; y < 5u; y = y + 1u) {
                let idx = x + 5u * y;
                kst_lo[idx] = kst_lo[idx] ^ d_lo;
                kst_hi[idx] = kst_hi[idx] ^ d_hi;
            }
        }
        var t_lo = kst_lo[1u]; var t_hi = kst_hi[1u];
        for (var i = 0u; i < 24u; i = i + 1u) {
            let dst = KPI[i];
            let tmp_lo = kst_lo[dst]; let tmp_hi = kst_hi[dst];
            let r = krotl64(t_lo, t_hi, KRHO[i]);
            kst_lo[dst] = r.x; kst_hi[dst] = r.y;
            t_lo = tmp_lo; t_hi = tmp_hi;
        }
        for (var y = 0u; y < 5u; y = y + 1u) {
            var rl: array<u32, 5>; var rh: array<u32, 5>;
            for (var x = 0u; x < 5u; x = x + 1u) {
                rl[x] = kst_lo[x + 5u*y]; rh[x] = kst_hi[x + 5u*y];
            }
            for (var x = 0u; x < 5u; x = x + 1u) {
                kst_lo[x+5u*y] = rl[x] ^ ((~rl[(x+1u)%5u]) & rl[(x+2u)%5u]);
                kst_hi[x+5u*y] = rh[x] ^ ((~rh[(x+1u)%5u]) & rh[(x+2u)%5u]);
            }
        }
        kst_lo[0] = kst_lo[0] ^ KRC_LO[round];
        kst_hi[0] = kst_hi[0] ^ KRC_HI[round];
    }
}

fn keccak256_64(data: ptr<function, array<u32, 16>>, hash: ptr<function, array<u32, 8>>) {
    for (var i = 0u; i < 25u; i = i + 1u) { kst_lo[i] = 0u; kst_hi[i] = 0u; }
    // Absorb 64 bytes = 8 lanes (each lane = 8 bytes = 2 u32 words)
    for (var w = 0u; w < 8u; w = w + 1u) {
        kst_lo[w] = kst_lo[w] ^ (*data)[w * 2u];
        kst_hi[w] = kst_hi[w] ^ (*data)[w * 2u + 1u];
    }
    // Keccak padding: byte 64 = 0x01, byte 135 = 0x80
    kst_lo[8] = kst_lo[8] ^ 0x01u;
    kst_hi[16] = kst_hi[16] ^ 0x80000000u;
    keccak_f();
    for (var w = 0u; w < 4u; w = w + 1u) {
        (*hash)[w * 2u] = kst_lo[w];
        (*hash)[w * 2u + 1u] = kst_hi[w];
    }
}

// ============================================================================
// Load/store helpers (big-endian 32 bytes <-> u256 little-endian u32 limbs)
// ============================================================================

fn load_be32(word_base: u32) -> array<u32, 8> {
    // Input is 32 bytes = 8 u32 words in the inputs array (byte-packed)
    // Stored as big-endian in the input. We need to reverse byte order within words
    // and reverse word order for little-endian limbs.
    var r: array<u32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let w = inputs[word_base + 7u - i];
        // Byte-swap u32 (big-endian to little-endian)
        r[i] = ((w >> 24u) & 0xFFu) | (((w >> 16u) & 0xFFu) << 8u)
             | (((w >> 8u) & 0xFFu) << 16u) | ((w & 0xFFu) << 24u);
    }
    return r;
}

// ============================================================================
// Main kernel
// ============================================================================

@compute @workgroup_size(256)
fn secp256k1_ecrecover(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.num_items) { return; }

    // Clear output
    let out_base = tid * 8u; // 32 bytes = 8 u32
    for (var i = 0u; i < 8u; i = i + 1u) { outputs[out_base + i] = 0u; }

    // Load signature: 128 bytes = 32 u32 per sig
    let in_base = tid * 32u;
    var r = load_be32(in_base);       // r: bytes 0..31
    var s = load_be32(in_base + 8u);  // s: bytes 32..63
    let v_byte = (inputs[in_base + 16u]) & 0xFFu; // v: byte 64
    var e = load_be32(in_base + 17u); // msg_hash: bytes 68..99

    var v = v_byte;
    if (v >= 27u) { v = v - 27u; }
    if (v >= 2u) { v = v % 2u; }

    // Validate r, s in [1, n-1]
    var n = SECP_N;
    if (u256_is_zero(&r) || u256_cmp(&r, &n) >= 0) { return; }
    if (u256_is_zero(&s) || u256_cmp(&s, &n) >= 0) { return; }
    if (v > 1u) { return; }

    // Decompress r -> R = (r, y)
    var r_mont: array<u32, 8>; to_mont_p(&r, &r_mont);
    var r2: array<u32, 8>; fp_sqr(&r_mont, &r2);
    var r3: array<u32, 8>; fp_mul(&r2, &r_mont, &r3);
    var seven = array<u32, 8>(7u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    var seven_mont: array<u32, 8>; to_mont_p(&seven, &seven_mont);
    var y2: array<u32, 8>; fp_add(&r3, &seven_mont, &y2);

    // sqrt via a^((p+1)/4) since p = 3 mod 4
    var exp_sqrt = array<u32, 8>(
        0xBFFFFF0Cu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x3FFFFFFFu
    );
    var one = array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    var y_mont: array<u32, 8>; to_mont_p(&one, &y_mont);
    var base_y = y2;
    for (var i = 0u; i < 8u; i = i + 1u) {
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            if (((exp_sqrt[i] >> bit) & 1u) != 0u) {
                var tmp: array<u32, 8>;
                fp_mul(&y_mont, &base_y, &tmp);
                y_mont = tmp;
            }
            var tmp2: array<u32, 8>;
            fp_sqr(&base_y, &tmp2);
            base_y = tmp2;
        }
    }

    // Verify sqrt: y^2 == y2
    var check: array<u32, 8>; fp_sqr(&y_mont, &check);
    if (u256_cmp(&check, &y2) != 0) { return; }

    // Select y parity
    var y_normal: array<u32, 8>; from_mont_p(&y_mont, &y_normal);
    let y_is_odd = (y_normal[0] & 1u) != 0u;
    if ((v == 0u && y_is_odd) || (v == 1u && !y_is_odd)) {
        var zero_val = u256_zero();
        fp_sub(&zero_val, &y_mont, &y_mont);
    }

    // r_inv = r^{-1} mod n
    var r_n_mont: array<u32, 8>; to_mont_n(&r, &r_n_mont);
    var r_inv_mont: array<u32, 8>; fn_inv(&r_n_mont, &r_inv_mont);

    // u1 = -(e * r_inv) mod n, u2 = s * r_inv mod n
    var e_n_mont: array<u32, 8>; to_mont_n(&e, &e_n_mont);
    var s_n_mont: array<u32, 8>; to_mont_n(&s, &s_n_mont);

    var u1_mont: array<u32, 8>; fn_mul(&e_n_mont, &r_inv_mont, &u1_mont);
    var u1: array<u32, 8>; from_mont_n(&u1_mont, &u1);
    if (!u256_is_zero(&u1)) {
        var nn = SECP_N;
        let _ = u256_sub(&nn, &u1, &u1);
    }

    var u2_mont: array<u32, 8>; fn_mul(&s_n_mont, &r_inv_mont, &u2_mont);
    var u2: array<u32, 8>; from_mont_n(&u2_mont, &u2);

    // Q = u1*G + u2*R
    var Gx_mont: array<u32, 8>; var gx = GX; to_mont_p(&gx, &Gx_mont);
    var Gy_mont: array<u32, 8>; var gy = GY; to_mont_p(&gy, &Gy_mont);

    var Q1 = ec_mul_affine(&u1, &Gx_mont, &Gy_mont);
    var Q2 = ec_mul_affine(&u2, &r_mont, &y_mont);

    // Add Q1 + Q2
    var Q: ECPoint;
    if (ec_is_inf(&Q1)) {
        Q = Q2;
    } else if (ec_is_inf(&Q2)) {
        Q = Q1;
    } else {
        var Q2x_aff: array<u32, 8>; var Q2y_aff: array<u32, 8>;
        ec_to_affine(&Q2, &Q2x_aff, &Q2y_aff);
        ec_add_mixed(&Q1, &Q2x_aff, &Q2y_aff, &Q);
    }

    if (ec_is_inf(&Q)) { return; }

    var Qx_aff: array<u32, 8>; var Qy_aff: array<u32, 8>;
    ec_to_affine(&Q, &Qx_aff, &Qy_aff);
    var Qx_norm: array<u32, 8>; from_mont_p(&Qx_aff, &Qx_norm);
    var Qy_norm: array<u32, 8>; from_mont_p(&Qy_aff, &Qy_norm);

    // Serialize Q.x || Q.y as 16 u32 words (big-endian bytes within each 32-byte half)
    var pubkey: array<u32, 16>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let w = Qx_norm[7u - i];
        pubkey[i] = ((w >> 24u) & 0xFFu) | (((w >> 16u) & 0xFFu) << 8u)
                  | (((w >> 8u) & 0xFFu) << 16u) | ((w & 0xFFu) << 24u);
    }
    for (var i = 0u; i < 8u; i = i + 1u) {
        let w = Qy_norm[7u - i];
        pubkey[8u + i] = ((w >> 24u) & 0xFFu) | (((w >> 16u) & 0xFFu) << 8u)
                       | (((w >> 8u) & 0xFFu) << 16u) | ((w & 0xFFu) << 24u);
    }

    // address = keccak256(pubkey)[12:]
    var hash: array<u32, 8>;
    keccak256_64(&pubkey, &hash);

    // Output: address (bytes 12-31 of hash) = last 20 bytes
    // hash is 32 bytes = 8 u32 words. Bytes 12..31 = words 3..7 (but byte offset 12 = word 3 byte 0)
    // Store as 5 u32 words at output (20 bytes), then valid byte
    outputs[out_base] = hash[3];
    outputs[out_base + 1u] = hash[4];
    outputs[out_base + 2u] = hash[5];
    outputs[out_base + 3u] = hash[6];
    outputs[out_base + 4u] = hash[7];
    // valid byte at output byte 20 = word 5
    outputs[out_base + 5u] = 1u;
}
