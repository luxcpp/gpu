// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused External Product — WGSL compute shaders
// Ported from external_product_fused.metal.
// Single kernel fusing decompose + NTT + multiply + INTT + accumulate.
// u64 emulated as vec2<u32>(lo, hi). No native u64 in WGSL.

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<storage, read_write> result: array<vec2<u32>>;    // [B, 2, N]
@group(0) @binding(1) var<storage, read>       rlwe: array<vec2<u32>>;      // [B, 2, N]
@group(0) @binding(2) var<storage, read>       rgsw_buf: array<vec2<u32>>;  // [B, 2, L, 2, N]
@group(0) @binding(3) var<storage, read>       fwd_twiddles: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read>       fwd_precon: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read>       inv_twiddles: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read>       inv_precon: array<vec2<u32>>;
@group(0) @binding(7) var<uniform>             params: FusedExtProdParams;

struct FusedExtProdParams {
    Q_lo: u32, Q_hi: u32,
    mu_lo: u32, mu_hi: u32,
    N_inv_lo: u32, N_inv_hi: u32,
    N_inv_precon_lo: u32, N_inv_precon_hi: u32,
    N: u32, log_N: u32,
    L: u32, base_log: u32,
    base_mask_lo: u32, base_mask_hi: u32,
    batch_size: u32,
    _pad0: u32,
}

// ---------------------------------------------------------------------------
// u64 emulation
// ---------------------------------------------------------------------------

fn u64_from(lo: u32, hi: u32) -> vec2<u32> { return vec2<u32>(lo, hi); }
fn u64_zero() -> vec2<u32> { return vec2<u32>(0u, 0u); }
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
fn mul32_64(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xFFFFu; let ah = a >> 16u;
    let bl = b & 0xFFFFu; let bh = b >> 16u;
    let ll = al * bl;
    let mid = al * bh + ah * bl;
    let lo = ll + (mid << 16u);
    let hi = ah * bh + (mid >> 16u) + select(0u, 1u, lo < ll) + select(0u, 0x10000u, mid < (al * bh));
    return vec2<u32>(lo, hi);
}
fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let ll = mul32_64(a.x, b.x);
    let cross = a.x * b.y + a.y * b.x;
    return vec2<u32>(ll.x, ll.y + cross);
}
fn u64_mulhi(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let p = mul32_64(a.y, b.y);
    let c1 = mul32_64(a.x, b.y);
    let c2 = mul32_64(a.y, b.x);
    let mid_lo = c1.y + c2.y;
    return vec2<u32>(p.x + mid_lo, p.y + select(0u, 1u, p.x + mid_lo < p.x));
}
fn u64_shr(a: vec2<u32>, n: u32) -> vec2<u32> {
    if (n == 0u) { return a; }
    if (n >= 32u) { return vec2<u32>(a.y >> (n - 32u), 0u); }
    return vec2<u32>((a.x >> n) | (a.y << (32u - n)), a.y >> n);
}
fn u64_and(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x & b.x, a.y & b.y);
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
fn barrett_mul(a: vec2<u32>, omega: vec2<u32>, Q: vec2<u32>, precon: vec2<u32>) -> vec2<u32> {
    let q_hat = u64_mulhi(a, precon);
    let product = u64_mul(a, omega);
    var r = u64_sub(product, u64_mul(q_hat, Q));
    if (u64_gte(r, Q)) { r = u64_sub(r, Q); }
    return r;
}
fn mod_mul_general(a: vec2<u32>, b: vec2<u32>, Q: vec2<u32>, mu: vec2<u32>) -> vec2<u32> {
    let prod = u64_mul(a, b);
    let q_hat = u64_mulhi(prod, mu);
    var r = u64_sub(prod, u64_mul(q_hat, Q));
    if (u64_gte(r, Q)) { r = u64_sub(r, Q); }
    return r;
}

// ---------------------------------------------------------------------------
// Workgroup shared: work[N] + tw_fwd[N] + pre_fwd[N] + tw_inv[N] + pre_inv[N]
// For N=1024: 5*1024 = 5120 entries
// ---------------------------------------------------------------------------

var<workgroup> work: array<vec2<u32>, 1024>;
var<workgroup> s_tw_fwd: array<vec2<u32>, 1024>;
var<workgroup> s_pre_fwd: array<vec2<u32>, 1024>;
var<workgroup> s_tw_inv: array<vec2<u32>, 1024>;
var<workgroup> s_pre_inv: array<vec2<u32>, 1024>;

// ---------------------------------------------------------------------------
// NTT helpers operating on `work` array
// ---------------------------------------------------------------------------

fn ntt_forward_work(N: u32, log_N: u32, Q: vec2<u32>, lid: u32) {
    for (var stage = 0u; stage < log_N; stage++) {
        let m = 1u << stage;
        let t = N >> (stage + 1u);
        for (var bf = lid; bf < N / 2u; bf += 256u) {
            let ii = bf / t;
            let jj = bf % t;
            let idx_lo = (ii << (log_N - stage)) + jj;
            let idx_hi = idx_lo + t;
            let tw_idx = m + ii;
            let lo = work[idx_lo];
            let hi = work[idx_hi];
            let hi_tw = barrett_mul(hi, s_tw_fwd[tw_idx], Q, s_pre_fwd[tw_idx]);
            work[idx_lo] = mod_add(lo, hi_tw, Q);
            work[idx_hi] = mod_sub(lo, hi_tw, Q);
        }
        workgroupBarrier();
    }
}

fn ntt_inverse_work(N: u32, log_N: u32, Q: vec2<u32>, lid: u32) {
    for (var stage = 0u; stage < log_N; stage++) {
        let m = N >> (stage + 1u);
        let t = 1u << stage;
        for (var bf = lid; bf < N / 2u; bf += 256u) {
            let ii = bf / t;
            let jj = bf % t;
            let idx_lo = (ii << (stage + 1u)) + jj;
            let idx_hi = idx_lo + t;
            let tw_idx = m + ii;
            let lo = work[idx_lo];
            let hi = work[idx_hi];
            let sum = mod_add(lo, hi, Q);
            let diff = mod_sub(lo, hi, Q);
            let diff_tw = barrett_mul(diff, s_tw_inv[tw_idx], Q, s_pre_inv[tw_idx]);
            work[idx_lo] = sum;
            work[idx_hi] = diff_tw;
        }
        workgroupBarrier();
    }
}

// ===========================================================================
// Fused external product kernel
// ===========================================================================

@compute @workgroup_size(256)
fn fused_external_product(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let batch_idx = wgid.y;
    let lid = lid_v.x;
    let N = params.N;
    let log_N = params.log_N;
    let L = params.L;
    let Q = u64_from(params.Q_lo, params.Q_hi);
    let mu = u64_from(params.mu_lo, params.mu_hi);
    let base_mask = u64_from(params.base_mask_lo, params.base_mask_hi);
    let base_log = params.base_log;
    let N_inv = u64_from(params.N_inv_lo, params.N_inv_hi);
    let N_inv_pre = u64_from(params.N_inv_precon_lo, params.N_inv_precon_hi);

    if (batch_idx >= params.batch_size) { return; }

    // Prefetch twiddles to workgroup memory
    for (var i = lid; i < N; i += 256u) {
        s_tw_fwd[i] = fwd_twiddles[i];
        s_pre_fwd[i] = fwd_precon[i];
        s_tw_inv[i] = inv_twiddles[i];
        s_pre_inv[i] = inv_precon[i];
    }
    workgroupBarrier();

    let rlwe_b = batch_idx * 2u * N;
    let rgsw_b = batch_idx * 2u * L * 2u * N;
    let result_b = batch_idx * 2u * N;

    for (var in_c = 0u; in_c < 2u; in_c++) {
        for (var l = 0u; l < L; l++) {
            // Load and decompose
            for (var i = lid; i < N; i += 256u) {
                let val = rlwe[rlwe_b + in_c * N + i];
                work[i] = u64_and(u64_shr(val, l * base_log), base_mask);
            }
            workgroupBarrier();

            // Forward NTT
            ntt_forward_work(N, log_N, Q, lid);

            // For each output component
            for (var out_c = 0u; out_c < 2u; out_c++) {
                // Multiply with RGSW
                for (var i = lid; i < N; i += 256u) {
                    let rgsw_idx = rgsw_b + in_c * L * 2u * N + l * 2u * N + out_c * N + i;
                    work[i] = mod_mul_general(work[i], rgsw_buf[rgsw_idx], Q, mu);
                }
                workgroupBarrier();

                // Inverse NTT
                ntt_inverse_work(N, log_N, Q, lid);

                // Scale by N^{-1} and accumulate
                for (var i = lid; i < N; i += 256u) {
                    let val = barrett_mul(work[i], N_inv, Q, N_inv_pre);
                    if (in_c == 0u && l == 0u) {
                        result[result_b + out_c * N + i] = val;
                    } else {
                        let acc = result[result_b + out_c * N + i];
                        result[result_b + out_c * N + i] = mod_add(acc, val, Q);
                    }
                }
                workgroupBarrier();

                // Reload NTT digit for next output component
                if (out_c == 0u) {
                    for (var i = lid; i < N; i += 256u) {
                        let val = rlwe[rlwe_b + in_c * N + i];
                        work[i] = u64_and(u64_shr(val, l * base_log), base_mask);
                    }
                    workgroupBarrier();
                    ntt_forward_work(N, log_N, Q, lid);
                }
            }
        }
    }
}

// ===========================================================================
// Accumulating variant: accumulator += ExternalProduct(rlwe, rgsw)
// ===========================================================================

@group(0) @binding(9) var<storage, read_write> accumulator: array<vec2<u32>>; // [B, 2, N]

@compute @workgroup_size(256)
fn fused_external_product_accumulate(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let batch_idx = wgid.y;
    let lid = lid_v.x;
    let N = params.N;
    let log_N = params.log_N;
    let L = params.L;
    let Q = u64_from(params.Q_lo, params.Q_hi);
    let mu = u64_from(params.mu_lo, params.mu_hi);
    let base_mask = u64_from(params.base_mask_lo, params.base_mask_hi);
    let base_log = params.base_log;
    let N_inv = u64_from(params.N_inv_lo, params.N_inv_hi);
    let N_inv_pre = u64_from(params.N_inv_precon_lo, params.N_inv_precon_hi);

    if (batch_idx >= params.batch_size) { return; }

    // Prefetch twiddles
    for (var i = lid; i < N; i += 256u) {
        s_tw_fwd[i] = fwd_twiddles[i];
        s_pre_fwd[i] = fwd_precon[i];
        s_tw_inv[i] = inv_twiddles[i];
        s_pre_inv[i] = inv_precon[i];
    }
    workgroupBarrier();

    let rlwe_b = batch_idx * 2u * N;
    let rgsw_b = batch_idx * 2u * L * 2u * N;
    let acc_b = batch_idx * 2u * N;

    for (var in_c = 0u; in_c < 2u; in_c++) {
        for (var l = 0u; l < L; l++) {
            for (var i = lid; i < N; i += 256u) {
                let val = rlwe[rlwe_b + in_c * N + i];
                work[i] = u64_and(u64_shr(val, l * base_log), base_mask);
            }
            workgroupBarrier();

            ntt_forward_work(N, log_N, Q, lid);

            for (var out_c = 0u; out_c < 2u; out_c++) {
                for (var i = lid; i < N; i += 256u) {
                    let rgsw_idx = rgsw_b + in_c * L * 2u * N + l * 2u * N + out_c * N + i;
                    work[i] = mod_mul_general(work[i], rgsw_buf[rgsw_idx], Q, mu);
                }
                workgroupBarrier();

                ntt_inverse_work(N, log_N, Q, lid);

                for (var i = lid; i < N; i += 256u) {
                    let val = barrett_mul(work[i], N_inv, Q, N_inv_pre);
                    accumulator[acc_b + out_c * N + i] = mod_add(accumulator[acc_b + out_c * N + i], val, Q);
                }
                workgroupBarrier();

                if (out_c == 0u) {
                    for (var i = lid; i < N; i += 256u) {
                        let val = rlwe[rlwe_b + in_c * N + i];
                        work[i] = u64_and(u64_shr(val, l * base_log), base_mask);
                    }
                    workgroupBarrier();
                    ntt_forward_work(N, log_N, Q, lid);
                }
            }
        }
    }
}
