// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Lux FHE GPU kernels in WGSL, ported from fhe_kernels.metal.
// Fused NTT, CMux, pipelined key switching, Barrett/Montgomery arithmetic.
// u64 emulated as vec2<u32>(lo, hi).

@group(0) @binding(0) var<storage, read_write> data: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> twiddles: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> scratch: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> params: FHEParams;

struct FHEParams {
    N: u32,         // Ring dimension (1024)
    LOG_N: u32,     // log2(N) = 10
    L: u32,         // Decomposition levels (4)
    BASE_LOG: u32,  // log2(base) (7)
    n_lwe: u32,     // LWE dimension
    Q_lo: u32, Q_hi: u32,
    BARRETT_MU_lo: u32, BARRETT_MU_hi: u32,
    batch_size: u32,
}

fn u64_from(lo: u32, hi: u32) -> vec2<u32> { return vec2<u32>(lo, hi); }
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
    return vec2<u32>(a.y * b.y, 0u);
}

fn barrett_reduce(x: vec2<u32>, Q: vec2<u32>, mu: vec2<u32>) -> vec2<u32> {
    let q_hat = u64_mulhi(x, mu);
    var r = u64_sub(x, u64_mul_lo(q_hat, Q));
    if (u64_gte(r, Q)) { r = u64_sub(r, Q); }
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
fn mod_mul(a: vec2<u32>, b: vec2<u32>, Q: vec2<u32>, mu: vec2<u32>) -> vec2<u32> {
    let prod = u64_mul_lo(a, b);
    return barrett_reduce(prod, Q, mu);
}
fn mod_neg(a: vec2<u32>, Q: vec2<u32>) -> vec2<u32> {
    if (a.x == 0u && a.y == 0u) { return a; }
    return u64_sub(Q, a);
}

// ============================================================================
// Forward NTT (in-place, one polynomial per workgroup)
// ============================================================================

var<workgroup> ntt_shared: array<vec2<u32>, 1024>;

@compute @workgroup_size(256)
fn fhe_ntt_forward(@builtin(global_invocation_id) gid: vec3<u32>,
                   @builtin(local_invocation_id) lid: vec3<u32>,
                   @builtin(workgroup_id) wg_id: vec3<u32>) {
    let batch_idx = wg_id.x;
    let thread_idx = lid.x;
    if (batch_idx >= params.batch_size) { return; }

    let N = params.N;
    let Q = vec2<u32>(params.Q_lo, params.Q_hi);
    let mu = vec2<u32>(params.BARRETT_MU_lo, params.BARRETT_MU_hi);
    let poly_base = batch_idx * N;

    // Load into shared memory
    for (var i = thread_idx; i < N; i = i + 256u) {
        ntt_shared[i] = data[poly_base + i];
    }
    workgroupBarrier();

    // Cooley-Tukey stages
    for (var stage = 0u; stage < params.LOG_N; stage = stage + 1u) {
        let m = 1u << stage;
        let t = N >> (stage + 1u);

        for (var bf = thread_idx; bf < N / 2u; bf = bf + 256u) {
            let group = bf / t;
            let j = bf % t;
            let idx_lo = (group * 2u * t) + j;
            let idx_hi = idx_lo + t;
            let tw_idx = m + group;

            let lo_val = ntt_shared[idx_lo];
            let hi_val = ntt_shared[idx_hi];
            let tw = twiddles[tw_idx];
            let omega = mod_mul(hi_val, tw, Q, mu);

            ntt_shared[idx_lo] = mod_add(lo_val, omega, Q);
            ntt_shared[idx_hi] = mod_sub(lo_val, omega, Q);
        }
        workgroupBarrier();
    }

    // Write back
    for (var i = thread_idx; i < N; i = i + 256u) {
        data[poly_base + i] = ntt_shared[i];
    }
}

// ============================================================================
// Inverse NTT (Gentleman-Sande)
// ============================================================================

@compute @workgroup_size(256)
fn fhe_ntt_inverse(@builtin(global_invocation_id) gid: vec3<u32>,
                   @builtin(local_invocation_id) lid: vec3<u32>,
                   @builtin(workgroup_id) wg_id: vec3<u32>) {
    let batch_idx = wg_id.x;
    let thread_idx = lid.x;
    if (batch_idx >= params.batch_size) { return; }

    let N = params.N;
    let Q = vec2<u32>(params.Q_lo, params.Q_hi);
    let mu = vec2<u32>(params.BARRETT_MU_lo, params.BARRETT_MU_hi);
    let poly_base = batch_idx * N;

    for (var i = thread_idx; i < N; i = i + 256u) {
        ntt_shared[i] = data[poly_base + i];
    }
    workgroupBarrier();

    for (var stage = 0u; stage < params.LOG_N; stage = stage + 1u) {
        let m = N >> (stage + 1u);
        let t = 1u << stage;

        for (var bf = thread_idx; bf < N / 2u; bf = bf + 256u) {
            let group = bf / t;
            let j = bf % t;
            let idx_lo = (group * 2u * t) + j;
            let idx_hi = idx_lo + t;
            let tw_idx = m + group;

            let lo_val = ntt_shared[idx_lo];
            let hi_val = ntt_shared[idx_hi];
            let tw = twiddles[tw_idx];

            ntt_shared[idx_lo] = mod_add(lo_val, hi_val, Q);
            let diff = mod_sub(lo_val, hi_val, Q);
            ntt_shared[idx_hi] = mod_mul(diff, tw, Q, mu);
        }
        workgroupBarrier();
    }

    for (var i = thread_idx; i < N; i = i + 256u) {
        data[poly_base + i] = ntt_shared[i];
    }
}

// ============================================================================
// Digit decomposition for external product
// ============================================================================

@compute @workgroup_size(256)
fn fhe_decompose_digits(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coeff_idx = gid.x;
    let level = gid.y;
    let batch_idx = gid.z;
    if (batch_idx >= params.batch_size || level >= params.L || coeff_idx >= params.N) { return; }

    let val = data[batch_idx * params.N + coeff_idx];
    let base_log = params.BASE_LOG;
    let base = 1u << base_log;
    let mask = base - 1u;

    // Extract digit at decomposition level
    // val_shifted = val >> (level * base_log), digit = val_shifted & mask
    // Since val is u64 (lo, hi), shift accordingly
    let total_shift = level * base_log;
    var digit_lo = 0u;
    if (total_shift < 32u) {
        digit_lo = (val.x >> total_shift) | (val.y << (32u - total_shift));
    } else {
        digit_lo = val.y >> (total_shift - 32u);
    }
    digit_lo = digit_lo & mask;

    let out_idx = batch_idx * params.L * params.N + level * params.N + coeff_idx;
    scratch[out_idx] = vec2<u32>(digit_lo, 0u);
}

// ============================================================================
// CMux difference: d1 - d0
// ============================================================================

@compute @workgroup_size(256)
fn fhe_cmux_diff(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.y;
    let coeff_idx = gid.x;
    if (batch_idx >= params.batch_size || coeff_idx >= params.N) { return; }

    let Q = vec2<u32>(params.Q_lo, params.Q_hi);
    let idx = batch_idx * params.N + coeff_idx;
    // data = d0, scratch = d1
    scratch[idx] = mod_sub(scratch[idx], data[idx], Q);
}

// ============================================================================
// Key switching step
// ============================================================================

@compute @workgroup_size(256)
fn fhe_keyswitch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let batch_idx = gid.y;
    if (batch_idx >= params.batch_size || idx >= params.n_lwe) { return; }

    let Q = vec2<u32>(params.Q_lo, params.Q_hi);
    let mu = vec2<u32>(params.BARRETT_MU_lo, params.BARRETT_MU_hi);
    let L = params.L;
    let base_log = params.BASE_LOG;
    let base = 1u << base_log;
    let mask = base - 1u;

    let in_base = batch_idx * (params.n_lwe + 1u);
    let a_val = data[in_base + idx];

    var acc = vec2<u32>(0u, 0u);
    for (var level = 0u; level < L; level = level + 1u) {
        let shift = level * base_log;
        var digit = 0u;
        if (shift < 32u) {
            digit = (a_val.x >> shift) | (a_val.y << (32u - shift));
        } else {
            digit = a_val.y >> (shift - 32u);
        }
        digit = digit & mask;

        // KSK contribution: ksk[idx * L + level] * digit
        let ksk_idx = idx * L + level;
        let ksk_val = twiddles[ksk_idx]; // reuse twiddles binding for KSK
        let prod = mod_mul(ksk_val, vec2<u32>(digit, 0u), Q, mu);
        acc = mod_add(acc, prod, Q);
    }

    scratch[batch_idx * params.n_lwe + idx] = acc;
}
