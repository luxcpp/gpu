// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Unified memory NTT kernels in WGSL, ported from ntt_unified_memory.metal.
// Zero-copy NTT with branch-free modular arithmetic.
// u64 emulated as vec2<u32>(lo, hi).

@group(0) @binding(0) var<storage, read_write> data: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> twiddles: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: NTTUnifiedParams;

struct NTTUnifiedParams {
    Q_lo: u32, Q_hi: u32,
    mu_lo: u32, mu_hi: u32,
    N_inv_lo: u32, N_inv_hi: u32,
    N_inv_precon_lo: u32, N_inv_precon_hi: u32,
    N: u32, log_N: u32,
    stage: u32, batch: u32,
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
    // Approximate high 64 bits
    let p = a.y * b.y;
    let c1 = (a.x >> 16u) * b.y + a.y * (b.x >> 16u);
    return vec2<u32>(p + (c1 >> 16u), 0u);
}

// Branch-free modular ops (matching unified memory Metal style)
fn mod_add_bf(a: vec2<u32>, b: vec2<u32>, Q: vec2<u32>) -> vec2<u32> {
    let sum = u64_add(a, b);
    let mask_hi = select(0u, 0xFFFFFFFFu, u64_gte(sum, Q));
    let mask_lo = mask_hi;
    return u64_sub(sum, vec2<u32>(Q.x & mask_lo, Q.y & mask_hi));
}
fn mod_sub_bf(a: vec2<u32>, b: vec2<u32>, Q: vec2<u32>) -> vec2<u32> {
    let diff = u64_sub(a, b);
    let mask = select(0u, 0xFFFFFFFFu, !u64_gte(a, b));
    return u64_add(diff, vec2<u32>(Q.x & mask, Q.y & mask));
}

fn barrett_mul_unified(a: vec2<u32>, b: vec2<u32>, Q: vec2<u32>,
                       mu: vec2<u32>) -> vec2<u32> {
    let lo = u64_mul_lo(a, b);
    let q = u64_mulhi(lo, mu);
    var r = u64_sub(lo, u64_mul_lo(q, Q));
    let mask = select(0u, 0xFFFFFFFFu, u64_gte(r, Q));
    r = u64_sub(r, vec2<u32>(Q.x & mask, Q.y & mask));
    return r;
}

@compute @workgroup_size(256)
fn ntt_unified_forward_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.y;
    let butterfly_idx = gid.x;
    if (batch_idx >= params.batch) { return; }

    let N = params.N;
    let Q = vec2<u32>(params.Q_lo, params.Q_hi);
    let mu = vec2<u32>(params.mu_lo, params.mu_hi);
    let stage = params.stage;
    let m = 1u << stage;
    let t = N >> (stage + 1u);
    if (butterfly_idx >= N >> 1u) { return; }

    let i = butterfly_idx / t;
    let j = butterfly_idx % t;
    let idx_lo = (i << (params.log_N - stage)) + j;
    let idx_hi = idx_lo + t;
    let poly = batch_idx * N;

    let lo_val = data[poly + idx_lo];
    let hi_val = data[poly + idx_hi];
    let tw = twiddles[m + i];
    let hi_tw = barrett_mul_unified(hi_val, tw, Q, mu);

    data[poly + idx_lo] = mod_add_bf(lo_val, hi_tw, Q);
    data[poly + idx_hi] = mod_sub_bf(lo_val, hi_tw, Q);
}

@compute @workgroup_size(256)
fn ntt_unified_inverse_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.y;
    let butterfly_idx = gid.x;
    if (batch_idx >= params.batch) { return; }

    let N = params.N;
    let Q = vec2<u32>(params.Q_lo, params.Q_hi);
    let mu = vec2<u32>(params.mu_lo, params.mu_hi);
    let stage = params.stage;
    let m = N >> (stage + 1u);
    let t = 1u << stage;
    if (butterfly_idx >= N >> 1u) { return; }

    let i = butterfly_idx / t;
    let j = butterfly_idx % t;
    let idx_lo = (i << (stage + 1u)) + j;
    let idx_hi = idx_lo + t;
    let poly = batch_idx * N;

    let lo_val = data[poly + idx_lo];
    let hi_val = data[poly + idx_hi];
    let tw = twiddles[m + i];

    let sum = mod_add_bf(lo_val, hi_val, Q);
    let diff = mod_sub_bf(lo_val, hi_val, Q);
    let diff_tw = barrett_mul_unified(diff, tw, Q, mu);

    data[poly + idx_lo] = sum;
    data[poly + idx_hi] = diff_tw;
}

@compute @workgroup_size(256)
fn ntt_unified_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.y;
    let coeff_idx = gid.x;
    if (batch_idx >= params.batch || coeff_idx >= params.N) { return; }

    let Q = vec2<u32>(params.Q_lo, params.Q_hi);
    let mu = vec2<u32>(params.mu_lo, params.mu_hi);
    let N_inv = vec2<u32>(params.N_inv_lo, params.N_inv_hi);
    let idx = batch_idx * params.N + coeff_idx;
    data[idx] = barrett_mul_unified(data[idx], N_inv, Q, mu);
}
