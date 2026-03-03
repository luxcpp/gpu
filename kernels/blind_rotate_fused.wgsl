// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused Blind Rotation — WGSL compute shader
// Ported from blind_rotate_fused.metal.
// u64 emulated as vec2<u32>(lo, hi). No native u64 in WGSL.
//
// Fuses the entire blind rotation loop (n iterations) into a single dispatch.
// Each workgroup handles one bootstrap operation.
//
// Shared memory layout (for N=1024, each vec2<u32> = 8 bytes):
//   acc_c0[N], acc_c1[N], rot_c0[N], rot_c1[N], work[N] = 5*N entries

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<storage, read_write> acc_out: array<vec2<u32>>;   // [B, 2, N]
@group(0) @binding(1) var<storage, read>       lwe_in: array<vec2<u32>>;    // [B, n+1]
@group(0) @binding(2) var<storage, read>       bsk: array<vec2<u32>>;       // [n, 2, L, 2, N]
@group(0) @binding(3) var<storage, read>       test_poly: array<vec2<u32>>; // [N]
@group(0) @binding(4) var<storage, read>       tw_fwd: array<vec2<u32>>;    // [N]
@group(0) @binding(5) var<storage, read>       tw_fwd_pre: array<vec2<u32>>; // [N]
@group(0) @binding(6) var<storage, read>       tw_inv: array<vec2<u32>>;    // [N]
@group(0) @binding(7) var<storage, read>       tw_inv_pre: array<vec2<u32>>; // [N]
@group(0) @binding(8) var<uniform>             params: BlindRotateParams;

struct BlindRotateParams {
    Q_lo: u32, Q_hi: u32,
    mu_lo: u32, mu_hi: u32,
    N_inv_lo: u32, N_inv_hi: u32,
    N_inv_precon_lo: u32, N_inv_precon_hi: u32,
    N: u32, log_N: u32,
    n: u32, L: u32,
    base_log: u32, base_mask_lo: u32, base_mask_hi: u32,
    batch_size: u32,
}

// ---------------------------------------------------------------------------
// u64 emulation (vec2<u32>, lo=x, hi=y)
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
fn u64_eq_zero(a: vec2<u32>) -> bool { return a.x == 0u && a.y == 0u; }

fn mul32_64(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xFFFFu; let ah = a >> 16u;
    let bl = b & 0xFFFFu; let bh = b >> 16u;
    let ll = al * bl;
    let lh = al * bh; let hl = ah * bl; let hh = ah * bh;
    let mid = lh + hl;
    let lo = ll + (mid << 16u);
    let hi = hh + (mid >> 16u) + select(0u, 1u, lo < ll) + select(0u, 0x10000u, mid < lh);
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

// ---------------------------------------------------------------------------
// Modular arithmetic
// ---------------------------------------------------------------------------

fn barrett_mul(a: vec2<u32>, omega: vec2<u32>, Q: vec2<u32>, precon: vec2<u32>) -> vec2<u32> {
    let q_hat = u64_mulhi(a, precon);
    let product = u64_mul(a, omega);
    var r = u64_sub(product, u64_mul(q_hat, Q));
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
    let prod = u64_mul(a, b);
    let q_hat = u64_mulhi(prod, mu);
    var r = u64_sub(prod, u64_mul(q_hat, Q));
    if (u64_gte(r, Q)) { r = u64_sub(r, Q); }
    return r;
}

// ---------------------------------------------------------------------------
// Workgroup shared memory (max 5 * 1024 = 5120 entries)
// ---------------------------------------------------------------------------

const WG_SIZE: u32 = 256u;
const MAX_N: u32 = 1024u;

// Flat shared array; partitioned in kernel: acc_c0[N], acc_c1[N], rot_c0[N], rot_c1[N], work[N]
var<workgroup> wg_data: array<vec2<u32>, 5120>;

// ---------------------------------------------------------------------------
// Forward NTT on workgroup-memory slice starting at `base`
// ---------------------------------------------------------------------------

fn ntt_forward(base: u32, N: u32, log_N: u32, Q: vec2<u32>, lid: u32, threads: u32) {
    for (var stage = 0u; stage < log_N; stage++) {
        let m = 1u << stage;
        let t = N >> (stage + 1u);
        for (var bf = lid; bf < N / 2u; bf += threads) {
            let ii = bf / t;
            let jj = bf % t;
            let idx_lo = (ii << (log_N - stage)) + jj;
            let idx_hi = idx_lo + t;
            let tw_idx = m + ii;
            let lo = wg_data[base + idx_lo];
            let hi = wg_data[base + idx_hi];
            let hi_tw = barrett_mul(hi, tw_fwd[tw_idx], Q, tw_fwd_pre[tw_idx]);
            wg_data[base + idx_lo] = mod_add(lo, hi_tw, Q);
            wg_data[base + idx_hi] = mod_sub(lo, hi_tw, Q);
        }
        workgroupBarrier();
    }
}

// ---------------------------------------------------------------------------
// Inverse NTT on workgroup-memory slice starting at `base`
// ---------------------------------------------------------------------------

fn ntt_inverse(base: u32, N: u32, log_N: u32, Q: vec2<u32>, lid: u32, threads: u32) {
    for (var stage = 0u; stage < log_N; stage++) {
        let m = N >> (stage + 1u);
        let t = 1u << stage;
        for (var bf = lid; bf < N / 2u; bf += threads) {
            let ii = bf / t;
            let jj = bf % t;
            let idx_lo = (ii << (stage + 1u)) + jj;
            let idx_hi = idx_lo + t;
            let tw_idx = m + ii;
            let lo = wg_data[base + idx_lo];
            let hi = wg_data[base + idx_hi];
            let sum = mod_add(lo, hi, Q);
            let diff = mod_sub(lo, hi, Q);
            let diff_tw = barrett_mul(diff, tw_inv[tw_idx], Q, tw_inv_pre[tw_idx]);
            wg_data[base + idx_lo] = sum;
            wg_data[base + idx_hi] = diff_tw;
        }
        workgroupBarrier();
    }
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------

@compute @workgroup_size(256)
fn blind_rotate_fused(
    @builtin(workgroup_id)         wgid: vec3<u32>,
    @builtin(local_invocation_id)  lid_v: vec3<u32>,
    @builtin(num_workgroups)       nwg: vec3<u32>,
) {
    let batch_idx = wgid.x;
    let lid = lid_v.x;
    let threads = WG_SIZE;

    let N = params.N;
    let log_N = params.log_N;
    let n = params.n;
    let L = params.L;
    let base_log = params.base_log;
    let base_mask = u64_from(params.base_mask_lo, params.base_mask_hi);
    let Q = u64_from(params.Q_lo, params.Q_hi);
    let N_inv = u64_from(params.N_inv_lo, params.N_inv_hi);
    let N_inv_pre = u64_from(params.N_inv_precon_lo, params.N_inv_precon_hi);
    let mu = u64_from(params.mu_lo, params.mu_hi);

    if (batch_idx >= params.batch_size) { return; }

    // Shared memory partition offsets
    let off_c0 = 0u;
    let off_c1 = N;
    let off_r0 = 2u * N;
    let off_r1 = 3u * N;
    let off_work = 4u * N;

    // Pointers into global arrays
    let lwe_base = batch_idx * (n + 1u);
    let out_base = batch_idx * 2u * N;

    // -------------------------------------------------------------------------
    // Stage 1: Initialize accumulator
    // -------------------------------------------------------------------------

    // Get b value (last LWE element), reduce mod 2N
    let b_val = lwe_in[lwe_base + n];
    let two_N = 2u * N;
    // Compute b mod 2N from lo word (sufficient for typical LWE dimensions)
    let b_mod = b_val.x % two_N;
    let init_rot = select(two_N - b_mod, 0u, b_mod == 0u);

    for (var i = lid; i < N; i += threads) {
        wg_data[off_c0 + i] = u64_zero();

        // Negacyclic rotation of test polynomial by init_rot
        var src_idx = i32(i) - i32(init_rot);
        var negate = false;
        if (src_idx < 0) { src_idx += i32(N); negate = !negate; }
        if (src_idx < 0) { src_idx += i32(N); negate = !negate; }
        if (src_idx >= i32(N)) { src_idx -= i32(N); negate = !negate; }
        let val = test_poly[u32(src_idx)];
        wg_data[off_c1 + i] = select(val, mod_sub(u64_zero(), val, Q), negate);
    }
    workgroupBarrier();

    // -------------------------------------------------------------------------
    // Stage 2: Main blind rotation loop
    // -------------------------------------------------------------------------

    for (var j = 0u; j < n; j++) {
        let a_j = lwe_in[lwe_base + j];
        let rot = a_j.x % two_N;

        if (rot == 0u) { continue; }

        // Step 2a: Negacyclic rotation of accumulator
        for (var i = lid; i < N; i += threads) {
            var s0 = i32(i) - i32(rot);
            var neg0 = false;
            if (s0 < 0) { s0 += i32(N); neg0 = !neg0; }
            if (s0 < 0) { s0 += i32(N); neg0 = !neg0; }
            if (s0 >= i32(N)) { s0 -= i32(N); neg0 = !neg0; }
            let v0 = wg_data[off_c0 + u32(s0)];
            wg_data[off_r0 + i] = select(v0, mod_sub(u64_zero(), v0, Q), neg0);

            var s1 = i32(i) - i32(rot);
            var neg1 = false;
            if (s1 < 0) { s1 += i32(N); neg1 = !neg1; }
            if (s1 < 0) { s1 += i32(N); neg1 = !neg1; }
            if (s1 >= i32(N)) { s1 -= i32(N); neg1 = !neg1; }
            let v1 = wg_data[off_c1 + u32(s1)];
            wg_data[off_r1 + i] = select(v1, mod_sub(u64_zero(), v1, Q), neg1);
        }
        workgroupBarrier();

        // Step 2b: diff = rotated - acc (in-place in rot)
        for (var i = lid; i < N; i += threads) {
            wg_data[off_r0 + i] = mod_sub(wg_data[off_r0 + i], wg_data[off_c0 + i], Q);
            wg_data[off_r1 + i] = mod_sub(wg_data[off_r1 + i], wg_data[off_c1 + i], Q);
        }
        workgroupBarrier();

        // Step 2c: External product
        let bsk_j_base = j * 2u * L * 2u * N;

        for (var in_c = 0u; in_c < 2u; in_c++) {
            let diff_off = select(off_r0, off_r1, in_c == 1u);

            for (var l = 0u; l < L; l++) {
                // Decompose: extract digit l from diff_comp
                for (var i = lid; i < N; i += threads) {
                    wg_data[off_work + i] = u64_and(u64_shr(wg_data[diff_off + i], l * base_log), base_mask);
                }
                workgroupBarrier();

                // Forward NTT on work buffer
                ntt_forward(off_work, N, log_N, Q, lid, threads);

                // Multiply with RGSW and accumulate
                let rgsw_base = bsk_j_base + in_c * L * 2u * N + l * 2u * N;

                for (var out_c = 0u; out_c < 2u; out_c++) {
                    let rot_off = select(off_r0, off_r1, out_c == 1u);
                    for (var i = lid; i < N; i += threads) {
                        let digit_ntt = wg_data[off_work + i];
                        let rgsw_val = bsk[rgsw_base + out_c * N + i];
                        let prod = mod_mul(digit_ntt, rgsw_val, Q, mu);
                        if (in_c == 0u && l == 0u) {
                            wg_data[rot_off + i] = prod;
                        } else {
                            wg_data[rot_off + i] = mod_add(wg_data[rot_off + i], prod, Q);
                        }
                    }
                    workgroupBarrier();
                }
            }
        }

        // Inverse NTT on accumulated products (rot_c0 and rot_c1)
        ntt_inverse(off_r0, N, log_N, Q, lid, threads);
        ntt_inverse(off_r1, N, log_N, Q, lid, threads);

        // Scale by N^{-1} and add to accumulator
        for (var i = lid; i < N; i += threads) {
            let p0 = barrett_mul(wg_data[off_r0 + i], N_inv, Q, N_inv_pre);
            let p1 = barrett_mul(wg_data[off_r1 + i], N_inv, Q, N_inv_pre);
            wg_data[off_c0 + i] = mod_add(wg_data[off_c0 + i], p0, Q);
            wg_data[off_c1 + i] = mod_add(wg_data[off_c1 + i], p1, Q);
        }
        workgroupBarrier();
    }

    // -------------------------------------------------------------------------
    // Stage 3: Write output
    // -------------------------------------------------------------------------

    for (var i = lid; i < N; i += threads) {
        acc_out[out_base + i] = wg_data[off_c0 + i];
        acc_out[out_base + N + i] = wg_data[off_c1 + i];
    }
}
