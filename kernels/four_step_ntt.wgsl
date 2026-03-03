// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Four-Step NTT in WGSL, ported from four_step_ntt.metal.
// Column NTTs, twiddle+transpose, row NTTs, scaling.
// u64 emulated as vec2<u32>(lo, hi).

@group(0) @binding(0) var<storage, read_write> data: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> twiddles: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> precon_twiddles: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> params: FourStepParams;

struct FourStepParams {
    Q_lo: u32, Q_hi: u32,
    mu_lo: u32, mu_hi: u32,
    N_inv_lo: u32, N_inv_hi: u32,
    N_inv_precon_lo: u32, N_inv_precon_hi: u32,
    N: u32, N1: u32, N2: u32,
    log_N1: u32, log_N2: u32,
    batch_size: u32,
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
fn u64_mul_lo(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let al = a.x & 0xFFFFu; let ah = a.x >> 16u;
    let bl = b.x & 0xFFFFu; let bh = b.x >> 16u;
    let ll = al * bl; let mid = al * bh + ah * bl;
    let lo = ll + (mid << 16u);
    let hi = ah * bh + (mid >> 16u) + select(0u, 1u, lo < ll) + a.x * b.y + a.y * b.x;
    return vec2<u32>(lo, hi);
}
fn u64_mulhi(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.y * b.y + ((a.x >> 16u) * b.y + a.y * (b.x >> 16u)) >> 16u, 0u);
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
fn barrett_mul(a: vec2<u32>, b: vec2<u32>, Q: vec2<u32>, pre: vec2<u32>) -> vec2<u32> {
    let q_approx = u64_mulhi(a, pre);
    let prod = u64_mul_lo(a, b);
    var r = u64_sub(prod, u64_mul_lo(q_approx, Q));
    if (u64_gte(r, Q)) { r = u64_sub(r, Q); }
    return r;
}

// ============================================================================
// Twiddle multiplication + pointwise operations
// ============================================================================

@compute @workgroup_size(256)
fn four_step_twiddle_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.y;
    let elem_idx = gid.x;
    if (batch_idx >= params.batch_size || elem_idx >= params.N) { return; }

    let Q = vec2<u32>(params.Q_lo, params.Q_hi);
    let idx = batch_idx * params.N + elem_idx;
    let tw = twiddles[elem_idx];
    let pre = precon_twiddles[elem_idx];
    data[idx] = barrett_mul(data[idx], tw, Q, pre);
}

// ============================================================================
// Transpose: data viewed as N1 x N2 -> N2 x N1
// ============================================================================

@compute @workgroup_size(256)
fn four_step_transpose(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.y;
    let elem_idx = gid.x;
    if (batch_idx >= params.batch_size || elem_idx >= params.N) { return; }

    let N1 = params.N1;
    let N2 = params.N2;
    let row = elem_idx / N2;
    let col = elem_idx % N2;

    let src_idx = batch_idx * params.N + row * N2 + col;
    let dst_idx = batch_idx * params.N + col * N1 + row;

    // Read source (need to use a different buffer for out-of-place)
    // For in-place, only swap upper triangle
    if (row < col) {
        let val_a = data[src_idx];
        let val_b = data[dst_idx];
        data[src_idx] = val_b;
        data[dst_idx] = val_a;
    }
}

// ============================================================================
// Scaling by N^{-1}
// ============================================================================

@compute @workgroup_size(256)
fn four_step_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.N * params.batch_size;
    if (idx >= total) { return; }

    let Q = vec2<u32>(params.Q_lo, params.Q_hi);
    let N_inv = vec2<u32>(params.N_inv_lo, params.N_inv_hi);
    let pre = vec2<u32>(params.N_inv_precon_lo, params.N_inv_precon_hi);
    data[idx] = barrett_mul(data[idx], N_inv, Q, pre);
}

// ============================================================================
// Pointwise multiplication
// ============================================================================

@compute @workgroup_size(256)
fn four_step_pointwise_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.N * params.batch_size;
    if (idx >= total) { return; }

    let Q = vec2<u32>(params.Q_lo, params.Q_hi);
    let mu = vec2<u32>(params.mu_lo, params.mu_hi);
    let a = data[idx];
    let b = twiddles[idx]; // reuse binding 1 as second polynomial
    let prod = u64_mul_lo(a, b);
    let q = u64_mulhi(prod, mu);
    var r = u64_sub(prod, u64_mul_lo(q, Q));
    if (u64_gte(r, Q)) { r = u64_sub(r, Q); }
    data[idx] = r;
}
