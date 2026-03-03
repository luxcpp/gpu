// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Polynomial multiplication in WGSL, ported from poly_mul.metal.
// Schoolbook and NTT-based multiplication for lattice cryptography.
// u64 emulated as vec2<u32>(lo, hi).

@group(0) @binding(0) var<storage, read> poly_a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> poly_b: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> poly_out: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> params: PolyMulParams;

struct PolyMulParams {
    Q_lo: u32, Q_hi: u32,
    Q_inv_lo: u32, Q_inv_hi: u32,
    N: u32, batch_size: u32,
}

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}
fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}
fn u64_gte(a: vec2<u32>, b: vec2<u32>) -> bool {
    if (a.y != b.y) { return a.y > b.y; }
    return a.x >= b.x;
}

// 32x32 -> 64
fn mul32_64(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xFFFFu; let ah = a >> 16u;
    let bl = b & 0xFFFFu; let bh = b >> 16u;
    let ll = al * bl; let mid = al * bh + ah * bl;
    let hh = ah * bh;
    let lo = ll + (mid << 16u);
    let hi = hh + (mid >> 16u) + select(0u, 1u, lo < ll);
    return vec2<u32>(lo, hi);
}

// u64 * u64 -> 128 bit (low and high u64)
fn u64_mul_wide(a: vec2<u32>, b: vec2<u32>,
                lo: ptr<function, vec2<u32>>,
                hi: ptr<function, vec2<u32>>) {
    let p0 = mul32_64(a.x, b.x);
    let p1 = mul32_64(a.x, b.y);
    let p2 = mul32_64(a.y, b.x);
    let p3 = mul32_64(a.y, b.y);

    (*lo).x = p0.x;
    var mid_sum = p0.y + p1.x;
    var c1 = select(0u, 1u, mid_sum < p0.y);
    mid_sum = mid_sum + p2.x;
    c1 = c1 + select(0u, 1u, mid_sum < p2.x);
    (*lo).y = mid_sum;

    (*hi) = u64_add(p3, vec2<u32>(c1 + p1.y + p2.y, 0u));
}

// Montgomery reduction: (lo, hi) * R^{-1} mod Q
fn mont_reduce(lo_val: vec2<u32>, hi_val: vec2<u32>,
               Q: vec2<u32>, q_inv: vec2<u32>) -> vec2<u32> {
    // m = lo * q_inv (mod R, keep low 64 bits)
    let al = lo_val.x & 0xFFFFu; let ah = lo_val.x >> 16u;
    let bl = q_inv.x & 0xFFFFu; let bh = q_inv.x >> 16u;
    let ll = al * bl; let mid = al * bh + ah * bl;
    let m_lo = ll + (mid << 16u);
    let m_hi = ah * bh + (mid >> 16u) + select(0u, 1u, m_lo < ll)
             + lo_val.x * q_inv.y + lo_val.y * q_inv.x;
    let m = vec2<u32>(m_lo, m_hi);

    // t = m * Q
    var t_lo: vec2<u32>; var t_hi: vec2<u32>;
    u64_mul_wide(m, Q, &t_lo, &t_hi);

    // result = (combined - t) >> 64 = hi - t_hi (with borrow from lo)
    let borrow = select(0u, 1u, !u64_gte(lo_val, t_lo));
    var r = u64_sub(hi_val, u64_add(t_hi, vec2<u32>(borrow, 0u)));

    if (u64_gte(r, Q)) { r = u64_sub(r, Q); }
    // Handle negative result
    if (r.y > 0x80000000u) { r = u64_add(r, Q); }
    return r;
}

fn mod_add(a: vec2<u32>, b: vec2<u32>, Q: vec2<u32>) -> vec2<u32> {
    let s = u64_add(a, b);
    if (u64_gte(s, Q)) { return u64_sub(s, Q); }
    return s;
}

fn mod_sub(a: vec2<u32>, b: vec2<u32>, Q: vec2<u32>) -> vec2<u32> {
    if (u64_gte(a, b)) { return u64_sub(a, b); }
    return u64_sub(u64_add(a, Q), b);
}

// ============================================================================
// Schoolbook negacyclic multiplication: c = a * b mod (X^N + 1) mod Q
// One thread per output coefficient.
// ============================================================================

@compute @workgroup_size(256)
fn poly_mul_schoolbook(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.y;
    let coeff_idx = gid.x;
    if (batch_idx >= params.batch_size || coeff_idx >= params.N) { return; }

    let N = params.N;
    let Q = vec2<u32>(params.Q_lo, params.Q_hi);
    let q_inv = vec2<u32>(params.Q_inv_lo, params.Q_inv_hi);
    let base_a = batch_idx * N;
    let base_b = batch_idx * N;

    var acc = vec2<u32>(0u, 0u);

    // c[k] = sum_{i+j=k} a[i]*b[j] - sum_{i+j=k+N} a[i]*b[j]  (negacyclic)
    for (var i = 0u; i < N; i = i + 1u) {
        let a_val = poly_a[base_a + i];

        // Positive contribution: j = k - i (if j >= 0)
        if (coeff_idx >= i) {
            let j = coeff_idx - i;
            let b_val = poly_b[base_b + j];
            var prod_lo: vec2<u32>; var prod_hi: vec2<u32>;
            u64_mul_wide(a_val, b_val, &prod_lo, &prod_hi);
            let reduced = mont_reduce(prod_lo, prod_hi, Q, q_inv);
            acc = mod_add(acc, reduced, Q);
        }

        // Negative contribution: j = k - i + N (wraps, so subtract)
        if (coeff_idx < i) {
            let j = coeff_idx + N - i;
            let b_val = poly_b[base_b + j];
            var prod_lo: vec2<u32>; var prod_hi: vec2<u32>;
            u64_mul_wide(a_val, b_val, &prod_lo, &prod_hi);
            let reduced = mont_reduce(prod_lo, prod_hi, Q, q_inv);
            acc = mod_sub(acc, reduced, Q);
        }
    }

    poly_out[batch_idx * N + coeff_idx] = acc;
}

// ============================================================================
// Pointwise NTT-domain multiplication (for use after forward NTT)
// ============================================================================

@compute @workgroup_size(256)
fn poly_mul_pointwise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.y;
    let coeff_idx = gid.x;
    if (batch_idx >= params.batch_size || coeff_idx >= params.N) { return; }

    let Q = vec2<u32>(params.Q_lo, params.Q_hi);
    let q_inv = vec2<u32>(params.Q_inv_lo, params.Q_inv_hi);
    let idx = batch_idx * params.N + coeff_idx;

    let a_val = poly_a[idx];
    let b_val = poly_b[idx];

    var prod_lo: vec2<u32>; var prod_hi: vec2<u32>;
    u64_mul_wide(a_val, b_val, &prod_lo, &prod_hi);
    poly_out[idx] = mont_reduce(prod_lo, prod_hi, Q, q_inv);
}
