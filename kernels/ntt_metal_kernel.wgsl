// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// NTT with shared memory twiddle prefetch in WGSL.
// Ported from ntt_metal_kernel.metal.
// Single NTT stage with workgroup-local twiddle cache.
// u64 emulated as vec2<u32>(lo, hi).

@group(0) @binding(0) var<storage, read_write> data: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> twiddles: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: NTTStageParams;

struct NTTStageParams {
    Q_lo: u32, Q_hi: u32,
    mu_lo: u32, mu_hi: u32,
    N: u32, log_N: u32,
    stage: u32, batch: u32,
}

var<workgroup> shared_twiddles: array<vec2<u32>, 4096>;

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
    let ll = al * bl;
    let mid = al * bh + ah * bl;
    let lo = ll + (mid << 16u);
    let hi = ah * bh + (mid >> 16u) + select(0u, 1u, lo < ll) + a.x * b.y + a.y * b.x;
    return vec2<u32>(lo, hi);
}
fn u64_mulhi_approx(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let al = a.x & 0xFFFFu; let ah = a.x >> 16u;
    let bh_lo = b.y & 0xFFFFu; let bh_hi = b.y >> 16u;
    let p = ah * bh_hi;
    let cross = ah * bh_lo + al * bh_hi + (a.y & 0xFFFFu) * (b.x >> 16u);
    return vec2<u32>(a.y * b.y + (cross >> 16u), p);
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
fn barrett_mul(a: vec2<u32>, b: vec2<u32>, Q: vec2<u32>, mu: vec2<u32>) -> vec2<u32> {
    let prod = u64_mul_lo(a, b);
    let q = u64_mulhi_approx(prod, mu);
    var r = u64_sub(prod, u64_mul_lo(q, Q));
    if (u64_gte(r, Q)) { r = u64_sub(r, Q); }
    return r;
}

@compute @workgroup_size(256)
fn ntt_forward_stage_shared(@builtin(global_invocation_id) gid: vec3<u32>,
                            @builtin(local_invocation_id) lid: vec3<u32>) {
    let batch_idx = gid.y;
    let thread_idx = lid.x;
    if (batch_idx >= params.batch) { return; }

    let N = params.N;
    let Q = vec2<u32>(params.Q_lo, params.Q_hi);
    let mu = vec2<u32>(params.mu_lo, params.mu_hi);
    let stage = params.stage;
    let m = 1u << stage;

    // Prefetch twiddles into workgroup memory
    if (thread_idx < m) {
        shared_twiddles[thread_idx] = twiddles[m + thread_idx];
    }
    workgroupBarrier();

    let t = N >> (stage + 1u);
    let num_butterflies = N >> 1u;
    let butterfly_idx = gid.x;
    if (butterfly_idx >= num_butterflies) { return; }

    let i = butterfly_idx / t;
    let j = butterfly_idx % t;
    let idx_lo = (i << (params.log_N - stage)) + j;
    let idx_hi = idx_lo + t;

    let poly_offset = batch_idx * N;
    let lo_val = data[poly_offset + idx_lo];
    let hi_val = data[poly_offset + idx_hi];

    let tw = shared_twiddles[i % m];
    let omega_factor = barrett_mul(hi_val, tw, Q, mu);

    data[poly_offset + idx_lo] = mod_add(lo_val, omega_factor, Q);
    data[poly_offset + idx_hi] = mod_sub(lo_val, omega_factor, Q);
}

@compute @workgroup_size(256)
fn ntt_inverse_stage_shared(@builtin(global_invocation_id) gid: vec3<u32>,
                            @builtin(local_invocation_id) lid: vec3<u32>) {
    let batch_idx = gid.y;
    let thread_idx = lid.x;
    if (batch_idx >= params.batch) { return; }

    let N = params.N;
    let Q = vec2<u32>(params.Q_lo, params.Q_hi);
    let mu = vec2<u32>(params.mu_lo, params.mu_hi);
    let stage = params.stage;
    let m = N >> (stage + 1u);

    if (thread_idx < m) {
        shared_twiddles[thread_idx] = twiddles[m + thread_idx];
    }
    workgroupBarrier();

    let t = 1u << stage;
    let num_butterflies = N >> 1u;
    let butterfly_idx = gid.x;
    if (butterfly_idx >= num_butterflies) { return; }

    let i = butterfly_idx / t;
    let j = butterfly_idx % t;
    let idx_lo = (i << (stage + 1u)) + j;
    let idx_hi = idx_lo + t;

    let poly_offset = batch_idx * N;
    let lo_val = data[poly_offset + idx_lo];
    let hi_val = data[poly_offset + idx_hi];

    let tw = shared_twiddles[i % m];
    let sum = mod_add(lo_val, hi_val, Q);
    let diff = mod_sub(lo_val, hi_val, Q);
    let diff_tw = barrett_mul(diff, tw, Q, mu);

    data[poly_offset + idx_lo] = sum;
    data[poly_offset + idx_hi] = diff_tw;
}
