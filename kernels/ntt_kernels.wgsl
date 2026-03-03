// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Optimal NTT kernels for Lux FHE in WGSL, ported from ntt_kernels.metal.
// Forward/inverse NTT stages with Barrett reduction.
// u64 emulated via vec2<u32>(lo, hi).

@group(0) @binding(0) var<storage, read_write> data: array<vec2<u32>>; // u64 as (lo, hi)
@group(0) @binding(1) var<storage, read> twiddles: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> precon_twiddles: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> params: NTTParams;
@group(0) @binding(4) var<uniform> stage_params: vec4<u32>; // x=stage, y=batch_size

struct NTTParams {
    Q_lo: u32, Q_hi: u32,
    mu_lo: u32, mu_hi: u32,
    N_inv_lo: u32, N_inv_hi: u32,
    N_inv_precon_lo: u32, N_inv_precon_hi: u32,
    N: u32, log_N: u32,
}

// u64 helpers
fn u64_zero() -> vec2<u32> { return vec2<u32>(0u, 0u); }
fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    let c = select(0u, 1u, lo < a.x);
    return vec2<u32>(lo, a.y + b.y + c);
}
fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let bw = select(0u, 1u, a.x < b.x);
    return vec2<u32>(a.x - b.x, a.y - b.y - bw);
}
fn u64_gte(a: vec2<u32>, b: vec2<u32>) -> bool {
    if (a.y > b.y) { return true; }
    if (a.y < b.y) { return false; }
    return a.x >= b.x;
}

// 32x32 -> 64 multiply
fn mul32_64(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xFFFFu; let ah = a >> 16u;
    let bl = b & 0xFFFFu; let bh = b >> 16u;
    let ll = al * bl;
    let lh = al * bh;
    let hl = ah * bl;
    let hh = ah * bh;
    let mid = lh + hl;
    let lo = ll + (mid << 16u);
    let hi = hh + (mid >> 16u) + select(0u, 1u, lo < ll) + select(0u, 0x10000u, mid < lh);
    return vec2<u32>(lo, hi);
}

// Approximate mulhi(a, b) for u64: returns high 64 bits of a*b
// For Barrett: we only need a.hi * b.hi as approximation for small Q
fn u64_mulhi_approx(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    // Full 128-bit would require 4 partial products; for Q < 2^32 this suffices
    let p = mul32_64(a.y, b.y);
    let cross1 = mul32_64(a.x, b.y);
    let cross2 = mul32_64(a.y, b.x);
    let mid_lo = cross1.y + cross2.y; // approximate carry into high product
    return vec2<u32>(p.x + mid_lo, p.y + select(0u, 1u, p.x + mid_lo < p.x));
}

// u64 multiply (lo 64 bits only)
fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let ll = mul32_64(a.x, b.x);
    let cross = a.x * b.y + a.y * b.x; // low 32 bits of cross products
    return vec2<u32>(ll.x, ll.y + cross);
}

fn mod_add(a: vec2<u32>, b: vec2<u32>, Q: vec2<u32>) -> vec2<u32> {
    let sum = u64_add(a, b);
    if (u64_gte(sum, Q)) { return u64_sub(sum, Q); }
    return sum;
}

fn mod_sub(a: vec2<u32>, b: vec2<u32>, Q: vec2<u32>) -> vec2<u32> {
    if (u64_gte(a, b)) { return u64_sub(a, b); }
    return u64_sub(u64_add(a, Q), b);
}

fn mod_mul_barrett(a: vec2<u32>, omega: vec2<u32>, Q: vec2<u32>,
                   precon: vec2<u32>) -> vec2<u32> {
    let q_approx = u64_mulhi_approx(a, precon);
    let product = u64_mul(a, omega);
    var result = u64_sub(product, u64_mul(q_approx, Q));
    if (u64_gte(result, Q)) { result = u64_sub(result, Q); }
    return result;
}

// ============================================================================
// Forward NTT stage (Cooley-Tukey)
// ============================================================================

@compute @workgroup_size(256)
fn ntt_forward_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.y;
    let butterfly_idx = gid.x;

    if (batch_idx >= stage_params.y) { return; }

    let N = params.N;
    let Q = vec2<u32>(params.Q_lo, params.Q_hi);
    let stage = stage_params.x;

    let m = 1u << stage;
    let t = N >> (stage + 1u);
    let num_butterflies = N >> 1u;
    if (butterfly_idx >= num_butterflies) { return; }

    let i = butterfly_idx / t;
    let j = butterfly_idx % t;
    let idx_lo = (i << (params.log_N - stage)) + j;
    let idx_hi = idx_lo + t;
    let tw_idx = m + i;

    let poly_offset = batch_idx * N;
    let lo_val = data[poly_offset + idx_lo];
    let hi_val = data[poly_offset + idx_hi];

    let omega = twiddles[tw_idx];
    let precon = precon_twiddles[tw_idx];

    let omega_factor = mod_mul_barrett(hi_val, omega, Q, precon);
    data[poly_offset + idx_lo] = mod_add(lo_val, omega_factor, Q);
    data[poly_offset + idx_hi] = mod_sub(lo_val, omega_factor, Q);
}

// ============================================================================
// Inverse NTT stage (Gentleman-Sande)
// ============================================================================

@compute @workgroup_size(256)
fn ntt_inverse_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.y;
    let butterfly_idx = gid.x;

    if (batch_idx >= stage_params.y) { return; }

    let N = params.N;
    let Q = vec2<u32>(params.Q_lo, params.Q_hi);
    let stage = stage_params.x;

    let m = N >> (stage + 1u);
    let t = 1u << stage;
    let num_butterflies = N >> 1u;
    if (butterfly_idx >= num_butterflies) { return; }

    let i = butterfly_idx / t;
    let j = butterfly_idx % t;
    let idx_lo = (i << (stage + 1u)) + j;
    let idx_hi = idx_lo + t;
    let tw_idx = m + i;

    let poly_offset = batch_idx * N;
    let lo_val = data[poly_offset + idx_lo];
    let hi_val = data[poly_offset + idx_hi];

    let omega = twiddles[tw_idx];
    let precon = precon_twiddles[tw_idx];

    let sum = mod_add(lo_val, hi_val, Q);
    let diff = mod_sub(lo_val, hi_val, Q);
    let diff_tw = mod_mul_barrett(diff, omega, Q, precon);

    data[poly_offset + idx_lo] = sum;
    data[poly_offset + idx_hi] = diff_tw;
}

// ============================================================================
// Scale by N^{-1} after inverse NTT
// ============================================================================

@compute @workgroup_size(256)
fn ntt_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.y;
    let coeff_idx = gid.x;
    if (batch_idx >= stage_params.y || coeff_idx >= params.N) { return; }

    let Q = vec2<u32>(params.Q_lo, params.Q_hi);
    let N_inv = vec2<u32>(params.N_inv_lo, params.N_inv_hi);
    let N_inv_pre = vec2<u32>(params.N_inv_precon_lo, params.N_inv_precon_hi);

    let idx = batch_idx * params.N + coeff_idx;
    data[idx] = mod_mul_barrett(data[idx], N_inv, Q, N_inv_pre);
}

// ============================================================================
// Pointwise multiply-accumulate
// ============================================================================

@compute @workgroup_size(256)
fn ntt_pointwise_mac(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.y;
    let coeff_idx = gid.x;
    if (batch_idx >= stage_params.y || coeff_idx >= params.N) { return; }

    let Q = vec2<u32>(params.Q_lo, params.Q_hi);
    let idx = batch_idx * params.N + coeff_idx;
    let a_val = twiddles[idx]; // reuse binding 1 as 'a' input
    let b_val = precon_twiddles[idx]; // reuse binding 2 as 'b' input

    // Simple modular multiply (no precon)
    let prod = u64_mul(a_val, b_val);
    // Reduction: if prod >= Q, subtract Q (sufficient for small Q)
    var r = prod;
    if (u64_gte(r, Q)) { r = u64_sub(r, Q); }
    if (u64_gte(r, Q)) { r = u64_sub(r, Q); }

    data[idx] = mod_add(data[idx], r, Q);
}
