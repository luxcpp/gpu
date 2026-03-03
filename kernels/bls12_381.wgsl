// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// BLS12-381 G1 point operations in WGSL.
// 384-bit field arithmetic (Fp) in Montgomery form for batch BLS verification.
// Uses 12 x u32 limbs (WGSL has no u64).
// Matches bls12_381.metal output byte-for-byte.

@group(0) @binding(0) var<storage, read> sig_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> results: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    num_items: u32,
    mode: u32, // 0 = verify, 1 = aggregate
}

// ============================================================================
// 384-bit integer as 12 x u32 (little-endian)
// ============================================================================

fn u384_zero() -> array<u32, 12> {
    return array<u32, 12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
}

fn u384_is_zero(a: ptr<function, array<u32, 12>>) -> bool {
    var acc = 0u;
    for (var i = 0u; i < 12u; i = i + 1u) { acc = acc | (*a)[i]; }
    return acc == 0u;
}

fn u384_cmp(a: ptr<function, array<u32, 12>>, b: ptr<function, array<u32, 12>>) -> i32 {
    for (var i = 11i; i >= 0; i = i - 1) {
        let ui = u32(i);
        if ((*a)[ui] > (*b)[ui]) { return 1; }
        if ((*a)[ui] < (*b)[ui]) { return -1; }
    }
    return 0;
}

fn u384_add(a: ptr<function, array<u32, 12>>, b: ptr<function, array<u32, 12>>,
            r: ptr<function, array<u32, 12>>) -> u32 {
    var c = 0u;
    for (var i = 0u; i < 12u; i = i + 1u) {
        let s1 = (*a)[i] + c;
        c = select(0u, 1u, s1 < (*a)[i]);
        let s2 = s1 + (*b)[i];
        c = c + select(0u, 1u, s2 < s1);
        (*r)[i] = s2;
    }
    return c;
}

fn u384_sub(a: ptr<function, array<u32, 12>>, b: ptr<function, array<u32, 12>>,
            r: ptr<function, array<u32, 12>>) -> u32 {
    var bw = 0u;
    for (var i = 0u; i < 12u; i = i + 1u) {
        let d1 = (*a)[i] - bw;
        bw = select(0u, 1u, d1 > (*a)[i]);
        let d2 = d1 - (*b)[i];
        bw = bw + select(0u, 1u, d2 > d1);
        (*r)[i] = d2;
    }
    return bw;
}

// BLS12-381 field modulus p (384 bits, 12 x u32 LE)
const BLS_P = array<u32, 12>(
    0xFFFFAAABu, 0xB9FEFFFFu, 0xB153FFFFu, 0x1EABFFFEu,
    0xF6B0F624u, 0x6730D2A0u, 0xF38512BFu, 0x64774B84u,
    0x434BACD7u, 0x4B1BA7B6u, 0x397FE69Au, 0x1A0111EAu
);

// Montgomery R^2 mod p
const BLS_R2 = array<u32, 12>(
    0x1C341746u, 0xF4DF1F34u, 0x09D104F1u, 0x0A76E6A6u,
    0x4C95B6D5u, 0x8DE5476Cu, 0x939D83C0u, 0x67EB88A9u,
    0xB519952Du, 0x9A793E85u, 0x92CAE3AAu, 0x11988FE5u
);

// Montgomery R mod p (1 in Montgomery form)
const BLS_R = array<u32, 12>(
    0x0002FFCDu, 0x76090000u, 0xC40C0002u, 0xEBF4000Bu,
    0x53C758BAu, 0x5F489857u, 0x70525745u, 0x77CE5853u,
    0xA256EC6Du, 0x5C071A97u, 0xFA80E493u, 0x15F65EC3u
);

// -p^{-1} mod 2^32
const BLS_P_INV: u32 = 0xFFFCFFFDu;

// ============================================================================
// Montgomery reduction/multiplication for 384-bit (12 x u32)
// ============================================================================

fn bls_mont_reduce(t: ptr<function, array<u32, 24>>,
                   r: ptr<function, array<u32, 12>>) {
    var a: array<u32, 25>;
    for (var i = 0u; i < 24u; i = i + 1u) { a[i] = (*t)[i]; }
    a[24] = 0u;

    for (var i = 0u; i < 12u; i = i + 1u) {
        let u = a[i] * BLS_P_INV;
        var carry = 0u;
        for (var j = 0u; j < 12u; j = j + 1u) {
            let ul = u & 0xFFFFu; let uh = u >> 16u;
            let ml = BLS_P[j] & 0xFFFFu; let mh = BLS_P[j] >> 16u;
            let ll = ul * ml;
            let mid = ul * mh + uh * ml;
            let hh = uh * mh;
            var lo = ll + (mid << 16u);
            var hi = hh + (mid >> 16u) + select(0u, 1u, lo < ll) + select(0u, 0x10000u, (ul*mh + uh*ml) < (ul*mh));

            let s1 = lo + carry; hi = hi + select(0u, 1u, s1 < lo);
            let s2 = a[i + j] + s1; hi = hi + select(0u, 1u, s2 < a[i + j]);
            a[i + j] = s2;
            carry = hi;
        }
        for (var j = 12u; i + j <= 24u; j = j + 1u) {
            let s = a[i + j] + carry;
            carry = select(0u, 1u, s < a[i + j]);
            a[i + j] = s;
            if (carry == 0u) { break; }
        }
    }

    for (var i = 0u; i < 12u; i = i + 1u) { (*r)[i] = a[i + 12u]; }

    var p = BLS_P;
    if (a[24] != 0u || u384_cmp(r, &p) >= 0) {
        let _ = u384_sub(r, &p, r);
    }
}

fn bls_fp_mul(a: ptr<function, array<u32, 12>>, b: ptr<function, array<u32, 12>>,
              r: ptr<function, array<u32, 12>>) {
    var t: array<u32, 24>;
    for (var i = 0u; i < 24u; i = i + 1u) { t[i] = 0u; }

    for (var i = 0u; i < 12u; i = i + 1u) {
        var carry = 0u;
        for (var j = 0u; j < 12u; j = j + 1u) {
            let al = (*a)[i] & 0xFFFFu; let ah = (*a)[i] >> 16u;
            let bl = (*b)[j] & 0xFFFFu; let bh = (*b)[j] >> 16u;
            let ll = al * bl;
            let mid = al * bh + ah * bl;
            let hh = ah * bh;
            var lo = ll + (mid << 16u);
            var hi = hh + (mid >> 16u) + select(0u, 1u, lo < ll);
            let s1 = lo + carry; hi = hi + select(0u, 1u, s1 < lo);
            let s2 = t[i + j] + s1; hi = hi + select(0u, 1u, s2 < t[i + j]);
            t[i + j] = s2;
            carry = hi;
        }
        for (var j = 12u; i + j < 24u; j = j + 1u) {
            let s = t[i + j] + carry;
            carry = select(0u, 1u, s < t[i + j]);
            t[i + j] = s;
            if (carry == 0u) { break; }
        }
    }
    bls_mont_reduce(&t, r);
}

fn bls_fp_sqr(a: ptr<function, array<u32, 12>>, r: ptr<function, array<u32, 12>>) {
    bls_fp_mul(a, a, r);
}

fn bls_fp_add(a: ptr<function, array<u32, 12>>, b: ptr<function, array<u32, 12>>,
              r: ptr<function, array<u32, 12>>) {
    var p = BLS_P;
    let c = u384_add(a, b, r);
    if (c != 0u || u384_cmp(r, &p) >= 0) {
        let _ = u384_sub(r, &p, r);
    }
}

fn bls_fp_sub(a: ptr<function, array<u32, 12>>, b: ptr<function, array<u32, 12>>,
              r: ptr<function, array<u32, 12>>) {
    var p = BLS_P;
    let bw = u384_sub(a, b, r);
    if (bw != 0u) {
        let _ = u384_add(r, &p, r);
    }
}

fn bls_fp_neg(a: ptr<function, array<u32, 12>>, r: ptr<function, array<u32, 12>>) {
    if (u384_is_zero(a)) { *r = u384_zero(); return; }
    var p = BLS_P;
    let _ = u384_sub(&p, a, r);
}

fn bls_to_mont(a: ptr<function, array<u32, 12>>, r: ptr<function, array<u32, 12>>) {
    var r2 = BLS_R2;
    bls_fp_mul(a, &r2, r);
}

fn bls_from_mont(a: ptr<function, array<u32, 12>>, r: ptr<function, array<u32, 12>>) {
    var t: array<u32, 24>;
    for (var i = 0u; i < 24u; i = i + 1u) { t[i] = 0u; }
    for (var i = 0u; i < 12u; i = i + 1u) { t[i] = (*a)[i]; }
    bls_mont_reduce(&t, r);
}

fn bls_fp_inv(a: ptr<function, array<u32, 12>>, r: ptr<function, array<u32, 12>>) {
    // p-2 (LE u32 limbs)
    var exp = BLS_P;
    exp[0] = exp[0] - 2u;
    var result = BLS_R;
    var base: array<u32, 12>;
    for (var i = 0u; i < 12u; i = i + 1u) { base[i] = (*a)[i]; }

    for (var i = 0u; i < 12u; i = i + 1u) {
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            if (((exp[i] >> bit) & 1u) != 0u) {
                var tmp: array<u32, 12>;
                bls_fp_mul(&result, &base, &tmp);
                result = tmp;
            }
            var tmp2: array<u32, 12>;
            bls_fp_sqr(&base, &tmp2);
            base = tmp2;
        }
    }
    *r = result;
}

// ============================================================================
// G1 point operations (Jacobian, Montgomery Fp)
// ============================================================================

struct G1Point {
    x: array<u32, 12>,
    y: array<u32, 12>,
    z: array<u32, 12>,
}

fn g1_identity() -> G1Point {
    var p: G1Point;
    p.x = BLS_R; p.y = BLS_R; p.z = u384_zero();
    return p;
}

fn g1_is_inf(p: ptr<function, G1Point>) -> bool {
    var z = (*p).z;
    return u384_is_zero(&z);
}

fn g1_double(p: ptr<function, G1Point>, r: ptr<function, G1Point>) {
    if (g1_is_inf(p)) { *r = *p; return; }
    var A: array<u32, 12>; bls_fp_sqr(&(*p).y, &A);
    var B: array<u32, 12>; bls_fp_mul(&(*p).x, &A, &B);
    var C: array<u32, 12>; bls_fp_sqr(&A, &C);
    var S: array<u32, 12>; bls_fp_add(&B, &B, &S); bls_fp_add(&S, &S, &S);
    var X2: array<u32, 12>; bls_fp_sqr(&(*p).x, &X2);
    var X2_2: array<u32, 12>; bls_fp_add(&X2, &X2, &X2_2);
    var M: array<u32, 12>; bls_fp_add(&X2_2, &X2, &M);
    var M2: array<u32, 12>; bls_fp_sqr(&M, &M2);
    var S2: array<u32, 12>; bls_fp_add(&S, &S, &S2);
    var X3: array<u32, 12>; bls_fp_sub(&M2, &S2, &X3);
    var SX: array<u32, 12>; bls_fp_sub(&S, &X3, &SX);
    var MSX: array<u32, 12>; bls_fp_mul(&M, &SX, &MSX);
    var C2: array<u32, 12>; bls_fp_add(&C, &C, &C2);
    var C4: array<u32, 12>; bls_fp_add(&C2, &C2, &C4);
    var C8: array<u32, 12>; bls_fp_add(&C4, &C4, &C8);
    var Y3: array<u32, 12>; bls_fp_sub(&MSX, &C8, &Y3);
    var YZ: array<u32, 12>; bls_fp_mul(&(*p).y, &(*p).z, &YZ);
    var Z3: array<u32, 12>; bls_fp_add(&YZ, &YZ, &Z3);
    (*r).x = X3; (*r).y = Y3; (*r).z = Z3;
}

fn g1_add_mixed(P: ptr<function, G1Point>, Qx: ptr<function, array<u32, 12>>,
                Qy: ptr<function, array<u32, 12>>, r: ptr<function, G1Point>) {
    if (g1_is_inf(P)) {
        (*r).x = *Qx; (*r).y = *Qy; (*r).z = BLS_R;
        return;
    }
    var Z2: array<u32, 12>; bls_fp_sqr(&(*P).z, &Z2);
    var U2: array<u32, 12>; bls_fp_mul(Qx, &Z2, &U2);
    var Z3: array<u32, 12>; bls_fp_mul(&Z2, &(*P).z, &Z3);
    var S2: array<u32, 12>; bls_fp_mul(Qy, &Z3, &S2);
    var H: array<u32, 12>; bls_fp_sub(&U2, &(*P).x, &H);
    var R: array<u32, 12>; bls_fp_sub(&S2, &(*P).y, &R);
    if (u384_is_zero(&H)) {
        if (u384_is_zero(&R)) { g1_double(P, r); return; }
        *r = g1_identity(); return;
    }
    var H2: array<u32, 12>; bls_fp_sqr(&H, &H2);
    var H3: array<u32, 12>; bls_fp_mul(&H, &H2, &H3);
    var U1H2: array<u32, 12>; bls_fp_mul(&(*P).x, &H2, &U1H2);
    var R2: array<u32, 12>; bls_fp_sqr(&R, &R2);
    var U1H2_2: array<u32, 12>; bls_fp_add(&U1H2, &U1H2, &U1H2_2);
    var t1: array<u32, 12>; bls_fp_sub(&R2, &H3, &t1);
    var X3: array<u32, 12>; bls_fp_sub(&t1, &U1H2_2, &X3);
    var UX: array<u32, 12>; bls_fp_sub(&U1H2, &X3, &UX);
    var RUX: array<u32, 12>; bls_fp_mul(&R, &UX, &RUX);
    var YH3: array<u32, 12>; bls_fp_mul(&(*P).y, &H3, &YH3);
    var Y3: array<u32, 12>; bls_fp_sub(&RUX, &YH3, &Y3);
    var Zr: array<u32, 12>; bls_fp_mul(&H, &(*P).z, &Zr);
    (*r).x = X3; (*r).y = Y3; (*r).z = Zr;
}

fn g1_to_affine(p: ptr<function, G1Point>,
                ax: ptr<function, array<u32, 12>>,
                ay: ptr<function, array<u32, 12>>) {
    if (g1_is_inf(p)) { *ax = u384_zero(); *ay = u384_zero(); return; }
    var z_inv: array<u32, 12>; bls_fp_inv(&(*p).z, &z_inv);
    var z_inv2: array<u32, 12>; bls_fp_sqr(&z_inv, &z_inv2);
    var z_inv3: array<u32, 12>; bls_fp_mul(&z_inv2, &z_inv, &z_inv3);
    bls_fp_mul(&(*p).x, &z_inv2, ax);
    bls_fp_mul(&(*p).y, &z_inv3, ay);
}

// ============================================================================
// BLS verify batch: decompress G1 signature, check on-curve
// ============================================================================

@compute @workgroup_size(256)
fn bls_verify_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.num_items) { return; }

    // Each signature is 48 bytes = 12 u32 words
    let sig_base = tid * 12u;

    // Read flag byte (first byte of first word)
    let first_word = sig_data[sig_base];
    let flags = first_word & 0xFFu;
    let compressed = (flags & 0x80u) != 0u;
    let infinity = (flags & 0x40u) != 0u;
    let y_sign = (flags & 0x20u) != 0u;

    if (infinity || !compressed) {
        results[tid] = 0u;
        return;
    }

    // Clear flag bits and deserialize x-coordinate (big-endian 48 bytes -> 12 x u32 LE)
    var x_raw: array<u32, 12>;
    for (var i = 0u; i < 12u; i = i + 1u) {
        var w = sig_data[sig_base + 11u - i];
        // Byte-swap
        w = ((w >> 24u) & 0xFFu) | (((w >> 16u) & 0xFFu) << 8u)
          | (((w >> 8u) & 0xFFu) << 16u) | ((w & 0xFFu) << 24u);
        x_raw[i] = w;
    }
    // Clear flag bits from the most significant byte
    x_raw[11] = x_raw[11] & 0x1FFFFFFFu;

    // Decompress: y^2 = x^3 + 4
    var x_mont: array<u32, 12>; bls_to_mont(&x_raw, &x_mont);
    var x2: array<u32, 12>; bls_fp_sqr(&x_mont, &x2);
    var x3: array<u32, 12>; bls_fp_mul(&x2, &x_mont, &x3);
    var four_raw = array<u32, 12>(4u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    var four_mont: array<u32, 12>; bls_to_mont(&four_raw, &four_mont);
    var y2: array<u32, 12>; bls_fp_add(&x3, &four_mont, &y2);

    // sqrt(y2) = y2^((p+1)/4) since p = 3 mod 4
    var exp = array<u32, 12>(
        0xFFFEAAAFu, 0xEE7FBFFFu, 0xAC54FFFFu, 0x07AAFFFFu,
        0x3DAC3D89u, 0xD9CC34A8u, 0x3CE144AFu, 0xD91DD2E1u,
        0x90D2EB35u, 0x92C6E9EDu, 0xE5FF9A6u, 0x0680447Au
    );

    var y_cand = BLS_R;
    var base = y2;
    for (var i = 0u; i < 12u; i = i + 1u) {
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            if (((exp[i] >> bit) & 1u) != 0u) {
                var tmp: array<u32, 12>;
                bls_fp_mul(&y_cand, &base, &tmp);
                y_cand = tmp;
            }
            var tmp2: array<u32, 12>;
            bls_fp_sqr(&base, &tmp2);
            base = tmp2;
        }
    }

    // Verify: y_cand^2 == y2
    var check: array<u32, 12>; bls_fp_sqr(&y_cand, &check);
    if (u384_cmp(&check, &y2) != 0) {
        results[tid] = 0u;
        return;
    }

    // Pick sign
    var y_normal: array<u32, 12>; bls_from_mont(&y_cand, &y_normal);
    let is_positive = (y_normal[0] & 1u) == 0u;
    if (is_positive == y_sign) {
        bls_fp_neg(&y_cand, &y_cand);
    }

    // On-curve check passed, subgroup check deferred to host
    results[tid] = 3u; // bit 0: on_curve, bit 1: needs_subgroup_check
}
