// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Modular reduction kernels for ML-DSA (q=8380417) and ML-KEM (q=3329) in WGSL.
// Barrett, full, centered reductions and Montgomery domain conversion.
// Matches reduce.metal output byte-for-byte.

@group(0) @binding(0) var<storage, read_write> data: array<i32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>;
// params.x = size
// params.y = mode: 0=barrett_dil 1=full_dil 2=centered_dil
//   3=barrett_kyber 4=full_kyber 5=centered_kyber 6=to_mont_dil 7=from_mont_dil

const DILITHIUM_Q: i32 = 8380417;
const DILITHIUM_Q_HALF: i32 = 4190208;
const DILITHIUM_QINV: i32 = 58728449;
const DILITHIUM_R2: i32 = 2365951;
const KYBER_Q: i32 = 3329;
const KYBER_Q_HALF: i32 = 1664;

// 16x16 -> 32 multiply helper (WGSL lacks 64-bit)
fn mul16(a: u32, b: u32) -> u32 {
    return (a & 0xFFFFu) * (b & 0xFFFFu);
}

fn barrett_reduce_dilithium(a: i32) -> i32 {
    // v = floor(2^46/q) ~ 8396807
    // Use 16-bit split: a * v >> 46
    let au = u32(a);
    let a_lo = au & 0xFFFFu;
    let a_hi = au >> 16u;
    let v: u32 = 8396807u;
    let v_lo = v & 0xFFFFu;
    let v_hi = v >> 16u;
    // Approximate product high bits
    let hh = a_hi * v_hi;
    let mid = a_lo * v_hi + a_hi * v_lo;
    let approx_hi = hh + (mid >> 16u);
    let t = i32(approx_hi >> 14u); // 46 - 32 = 14
    return a - t * DILITHIUM_Q;
}

fn full_reduce_dilithium(a: i32) -> i32 {
    var t = barrett_reduce_dilithium(a);
    t = t + ((t >> 31) & DILITHIUM_Q);
    t = t - DILITHIUM_Q;
    t = t + ((t >> 31) & DILITHIUM_Q);
    return t;
}

fn centered_reduce_dilithium(a: i32) -> i32 {
    var t = full_reduce_dilithium(a);
    if (t > DILITHIUM_Q_HALF) { t = t - DILITHIUM_Q; }
    return t;
}

fn montgomery_reduce_dil(a_lo: i32, a_hi: i32) -> i32 {
    let t = a_lo * DILITHIUM_QINV;
    let t_u = u32(t);
    let q_u = u32(DILITHIUM_Q);
    // high(t * Q): split into 16-bit pieces
    let tl = t_u & 0xFFFFu; let th = t_u >> 16u;
    let ql = q_u & 0xFFFFu; let qh = q_u >> 16u;
    let mid = tl * qh + th * ql;
    let high_part = th * qh + (mid >> 16u);
    var r = a_hi - i32(high_part);
    if (r < 0) { r = r + DILITHIUM_Q; }
    if (r >= DILITHIUM_Q) { r = r - DILITHIUM_Q; }
    return r;
}

fn barrett_reduce_kyber(a: i32) -> i32 {
    let t = (a * 20159 + (1 << 25)) >> 26;
    return a - t * KYBER_Q;
}

fn full_reduce_kyber(a: i32) -> i32 {
    var t = barrett_reduce_kyber(a);
    t = t + ((t >> 31) & KYBER_Q);
    t = t - KYBER_Q;
    t = t + ((t >> 31) & KYBER_Q);
    return t;
}

fn centered_reduce_kyber(a: i32) -> i32 {
    var t = full_reduce_kyber(a);
    if (t > KYBER_Q_HALF) { t = t - KYBER_Q; }
    return t;
}

// 32x32 -> (lo, hi) multiply for Montgomery
fn mul32(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xFFFFu; let ah = a >> 16u;
    let bl = b & 0xFFFFu; let bh = b >> 16u;
    let ll = al * bl;
    let lh = al * bh;
    let hl = ah * bl;
    let hh = ah * bh;
    let mid = lh + hl;
    let lo = ll + (mid << 16u);
    let carry = select(0u, 1u, lo < ll) + select(0u, 1u, mid < lh);
    let hi = hh + (mid >> 16u) + carry;
    return vec2<u32>(lo, hi);
}

@compute @workgroup_size(256)
fn reduce_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    let mode = params.y;
    let val = data[idx];

    if (mode == 0u) {
        data[idx] = barrett_reduce_dilithium(val);
    } else if (mode == 1u) {
        data[idx] = full_reduce_dilithium(val);
    } else if (mode == 2u) {
        data[idx] = centered_reduce_dilithium(val);
    } else if (mode == 3u) {
        data[idx] = barrett_reduce_kyber(val);
    } else if (mode == 4u) {
        data[idx] = full_reduce_kyber(val);
    } else if (mode == 5u) {
        data[idx] = centered_reduce_kyber(val);
    } else if (mode == 6u) {
        // To Montgomery: a * R^2 then reduce
        let p = mul32(u32(val), u32(DILITHIUM_R2));
        data[idx] = montgomery_reduce_dil(i32(p.x), i32(p.y));
    } else if (mode == 7u) {
        // From Montgomery: reduce(a, 0)
        data[idx] = montgomery_reduce_dil(val, 0);
    }
}
