// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Batched External Product — WGSL compute shaders
// Ported from external_product_batch.metal.
// Fully batched: multiple RLWE ciphertexts against a single RGSW.
// u64 emulated as vec2<u32>(lo, hi). No native u64 in WGSL.

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<storage, read_write> out_c0: array<vec2<u32>>;   // [batch, N]
@group(0) @binding(1) var<storage, read_write> out_c1: array<vec2<u32>>;   // [batch, N]
@group(0) @binding(2) var<storage, read>       rlwe_c0: array<vec2<u32>>;  // [batch, N]
@group(0) @binding(3) var<storage, read>       rlwe_c1: array<vec2<u32>>;  // [batch, N]
@group(0) @binding(4) var<storage, read>       rgsw: array<vec2<u32>>;     // [2, L, 2, N]
@group(0) @binding(5) var<uniform>             params: BatchedExtProdParams;

struct BatchedExtProdParams {
    Q_lo: u32, Q_hi: u32,
    barrett_mu_lo: u32, barrett_mu_hi: u32,
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

// ---------------------------------------------------------------------------
// Workgroup shared memory
// Max layout: decomp_c0[L*N] + decomp_c1[L*N] + acc_c0[N] + acc_c1[N]
// For L=4, N=1024: 4*1024*2 + 2*1024 = 10240 entries (too large for most GPUs)
// Use smaller approach: process levels sequentially, shared = acc_c0[N] + acc_c1[N]
// ---------------------------------------------------------------------------

var<workgroup> shared_acc: array<vec2<u32>, 2048>;  // acc_c0[N] + acc_c1[N]

// ===========================================================================
// Main batched external product kernel
// ===========================================================================

@compute @workgroup_size(256)
fn external_product_batched(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let batch_idx = wgid.x;
    let lid = lid_v.x;
    let N = params.N;
    let L = params.L;
    let Q = u64_from(params.Q_lo, params.Q_hi);
    let mu = u64_from(params.barrett_mu_lo, params.barrett_mu_hi);
    let base_log = params.base_log;
    let base_mask = u64_from(params.base_mask_lo, params.base_mask_hi);

    if (batch_idx >= params.batch_size) { return; }

    let off_a0 = 0u;
    let off_a1 = N;

    // Initialize accumulators to zero
    for (var i = lid; i < N; i += 256u) {
        shared_acc[off_a0 + i] = u64_zero();
        shared_acc[off_a1 + i] = u64_zero();
    }
    workgroupBarrier();

    // Process each level
    for (var l = 0u; l < L; l++) {
        // Contribution from RLWE component 0
        let rgsw_c0_l_0_base = 0u * L * 2u * N + l * 2u * N + 0u * N;
        let rgsw_c0_l_1_base = 0u * L * 2u * N + l * 2u * N + 1u * N;

        for (var i = lid; i < N; i += 256u) {
            let val_c0 = rlwe_c0[batch_idx * N + i];
            let d0 = u64_and(u64_shr(val_c0, l * base_log), base_mask);

            let prod_0 = barrett_mul(d0, rgsw[rgsw_c0_l_0_base + i], Q, mu);
            let prod_1 = barrett_mul(d0, rgsw[rgsw_c0_l_1_base + i], Q, mu);

            shared_acc[off_a0 + i] = mod_add(shared_acc[off_a0 + i], prod_0, Q);
            shared_acc[off_a1 + i] = mod_add(shared_acc[off_a1 + i], prod_1, Q);
        }
        workgroupBarrier();

        // Contribution from RLWE component 1
        let rgsw_c1_l_0_base = 1u * L * 2u * N + l * 2u * N + 0u * N;
        let rgsw_c1_l_1_base = 1u * L * 2u * N + l * 2u * N + 1u * N;

        for (var i = lid; i < N; i += 256u) {
            let val_c1 = rlwe_c1[batch_idx * N + i];
            let d1 = u64_and(u64_shr(val_c1, l * base_log), base_mask);

            let prod_0 = barrett_mul(d1, rgsw[rgsw_c1_l_0_base + i], Q, mu);
            let prod_1 = barrett_mul(d1, rgsw[rgsw_c1_l_1_base + i], Q, mu);

            shared_acc[off_a0 + i] = mod_add(shared_acc[off_a0 + i], prod_0, Q);
            shared_acc[off_a1 + i] = mod_add(shared_acc[off_a1 + i], prod_1, Q);
        }
        workgroupBarrier();
    }

    // Write output
    for (var i = lid; i < N; i += 256u) {
        out_c0[batch_idx * N + i] = shared_acc[off_a0 + i];
        out_c1[batch_idx * N + i] = shared_acc[off_a1 + i];
    }
}

// ===========================================================================
// Accumulating variant: acc += ExternalProduct(rlwe, rgsw)
// ===========================================================================

@compute @workgroup_size(256)
fn external_product_batched_accumulate(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let batch_idx = wgid.x;
    let lid = lid_v.x;
    let N = params.N;
    let L = params.L;
    let Q = u64_from(params.Q_lo, params.Q_hi);
    let mu = u64_from(params.barrett_mu_lo, params.barrett_mu_hi);
    let base_log = params.base_log;
    let base_mask = u64_from(params.base_mask_lo, params.base_mask_hi);

    if (batch_idx >= params.batch_size) { return; }

    for (var l = 0u; l < L; l++) {
        let rgsw_c0_l_0 = 0u * L * 2u * N + l * 2u * N;
        let rgsw_c0_l_1 = 0u * L * 2u * N + l * 2u * N + N;
        let rgsw_c1_l_0 = 1u * L * 2u * N + l * 2u * N;
        let rgsw_c1_l_1 = 1u * L * 2u * N + l * 2u * N + N;

        for (var i = lid; i < N; i += 256u) {
            let val0 = rlwe_c0[batch_idx * N + i];
            let val1 = rlwe_c1[batch_idx * N + i];
            let d0 = u64_and(u64_shr(val0, l * base_log), base_mask);
            let d1 = u64_and(u64_shr(val1, l * base_log), base_mask);

            let p00 = barrett_mul(d0, rgsw[rgsw_c0_l_0 + i], Q, mu);
            let p01 = barrett_mul(d0, rgsw[rgsw_c0_l_1 + i], Q, mu);
            let p10 = barrett_mul(d1, rgsw[rgsw_c1_l_0 + i], Q, mu);
            let p11 = barrett_mul(d1, rgsw[rgsw_c1_l_1 + i], Q, mu);

            out_c0[batch_idx * N + i] = mod_add(out_c0[batch_idx * N + i], mod_add(p00, p10, Q), Q);
            out_c1[batch_idx * N + i] = mod_add(out_c1[batch_idx * N + i], mod_add(p01, p11, Q), Q);
        }
        workgroupBarrier();
    }
}

// ===========================================================================
// NTT-Domain variant
// ===========================================================================

@group(0) @binding(6) var<storage, read> fwd_twiddles: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read> fwd_precon: array<vec2<u32>>;
@group(0) @binding(8) var<storage, read> rgsw_ntt: array<vec2<u32>>;

var<workgroup> ntt_work: array<vec2<u32>, 1024>;
var<workgroup> ntt_acc_c0: array<vec2<u32>, 1024>;
var<workgroup> ntt_acc_c1: array<vec2<u32>, 1024>;

@compute @workgroup_size(256)
fn external_product_batched_ntt(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let batch_idx = wgid.x;
    let lid = lid_v.x;
    let N = params.N;
    let log_N = params.log_N;
    let L = params.L;
    let Q = u64_from(params.Q_lo, params.Q_hi);
    let mu = u64_from(params.barrett_mu_lo, params.barrett_mu_hi);
    let base_log = params.base_log;
    let base_mask = u64_from(params.base_mask_lo, params.base_mask_hi);

    if (batch_idx >= params.batch_size) { return; }

    // Initialize accumulators
    for (var i = lid; i < N; i += 256u) {
        ntt_acc_c0[i] = u64_zero();
        ntt_acc_c1[i] = u64_zero();
    }
    workgroupBarrier();

    for (var in_c = 0u; in_c < 2u; in_c++) {
        for (var l = 0u; l < L; l++) {
            // Load and decompose
            for (var i = lid; i < N; i += 256u) {
                var val: vec2<u32>;
                if (in_c == 0u) {
                    val = rlwe_c0[batch_idx * N + i];
                } else {
                    val = rlwe_c1[batch_idx * N + i];
                }
                ntt_work[i] = u64_and(u64_shr(val, l * base_log), base_mask);
            }
            workgroupBarrier();

            // Forward NTT (Cooley-Tukey)
            for (var stage = 0u; stage < log_N; stage++) {
                let m = 1u << stage;
                let t = N >> (stage + 1u);
                for (var bf = lid; bf < N / 2u; bf += 256u) {
                    let ii = bf / t;
                    let jj = bf % t;
                    let idx_lo = (ii << (log_N - stage)) + jj;
                    let idx_hi = idx_lo + t;
                    let tw_idx = m + ii;
                    let lo = ntt_work[idx_lo];
                    let hi = ntt_work[idx_hi];
                    let hi_tw = barrett_mul(hi, fwd_twiddles[tw_idx], Q, fwd_precon[tw_idx]);
                    ntt_work[idx_lo] = mod_add(lo, hi_tw, Q);
                    ntt_work[idx_hi] = mod_sub(lo, hi_tw, Q);
                }
                workgroupBarrier();
            }

            // Pointwise multiply and accumulate
            let rgsw_l_0 = in_c * L * 2u * N + l * 2u * N;
            let rgsw_l_1 = in_c * L * 2u * N + l * 2u * N + N;

            for (var i = lid; i < N; i += 256u) {
                let digit_ntt = ntt_work[i];
                let p0 = barrett_mul(digit_ntt, rgsw_ntt[rgsw_l_0 + i], Q, mu);
                let p1 = barrett_mul(digit_ntt, rgsw_ntt[rgsw_l_1 + i], Q, mu);
                ntt_acc_c0[i] = mod_add(ntt_acc_c0[i], p0, Q);
                ntt_acc_c1[i] = mod_add(ntt_acc_c1[i], p1, Q);
            }
            workgroupBarrier();
        }
    }

    // Write accumulated NTT-domain result
    for (var i = lid; i < N; i += 256u) {
        out_c0[batch_idx * N + i] = ntt_acc_c0[i];
        out_c1[batch_idx * N + i] = ntt_acc_c1[i];
    }
}
