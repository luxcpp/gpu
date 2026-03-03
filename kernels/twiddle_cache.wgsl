// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-2-Clause
//
// Twiddle Hotset Caching — WGSL compute shaders
// Ported from twiddle_cache.metal.
// NTT kernels with intelligent twiddle caching for different memory tiers.
// u64 emulated as vec2<u32>(lo, hi). No native u64 in WGSL.
//
// Memory hierarchy:
//   Uniform buffer: First-level twiddles (8 values), modular constants
//   Workgroup memory: Stage-specific twiddles with prefetch
//   Registers: Current butterfly operands

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<storage, read_write> data: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read>       twiddles_buf: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read>        cache: ConstantCache;
@group(0) @binding(3) var<uniform>             ntt_params: NTTParams;

// Constant cache with first-level twiddles per prime
struct PrimeConstants {
    q_lo: u32, q_hi: u32,
    q_inv_lo: u32, q_inv_hi: u32,
    mu_hi_lo: u32, mu_hi_hi: u32,
    mu_lo_lo: u32, mu_lo_hi: u32,
    r_squared_lo: u32, r_squared_hi: u32,
    root_lo: u32, root_hi: u32,
    root_inv_lo: u32, root_inv_hi: u32,
    n_inv_lo: u32, n_inv_hi: u32,
}

struct ConstantCache {
    num_primes: u32,
    ring_dim: u32,
    _pad0: u32, _pad1: u32,
    // First-level twiddles: 8 per prime, up to 16 primes
    // Flattened: first_level_twiddles[prime_idx * 8 + i]
    first_level_twiddles: array<vec2<u32>, 128>,     // [16][8]
    first_level_inv_twiddles: array<vec2<u32>, 128>,  // [16][8]
    // Prime constants (up to 16 primes)
    primes: array<PrimeConstants, 16>,
}

struct NTTParams {
    Q_lo: u32, Q_hi: u32,
    mu_lo: u32, mu_hi: u32,
    N_inv_lo: u32, N_inv_hi: u32,
    N_inv_precon_lo: u32, N_inv_precon_hi: u32,
    N: u32, log_N: u32,
    stage: u32, prime_idx: u32,
    batch: u32, prefetch_stage: u32,
    _pad0: u32, _pad1: u32,
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

fn mod_add(a: vec2<u32>, b: vec2<u32>, Q: vec2<u32>) -> vec2<u32> {
    let s = u64_add(a, b);
    if (u64_gte(s, Q)) { return u64_sub(s, Q); }
    return s;
}
fn mod_sub(a: vec2<u32>, b: vec2<u32>, Q: vec2<u32>) -> vec2<u32> {
    if (u64_gte(a, b)) { return u64_sub(a, b); }
    return u64_sub(u64_add(a, Q), b);
}
fn barrett_mul(a: vec2<u32>, omega: vec2<u32>, Q: vec2<u32>, mu: vec2<u32>) -> vec2<u32> {
    let q_hat = u64_mulhi(a, mu);
    let product = u64_mul(a, omega);
    var r = u64_sub(product, u64_mul(q_hat, Q));
    if (u64_gte(r, Q)) { r = u64_sub(r, Q); }
    return r;
}

// ---------------------------------------------------------------------------
// Bank conflict avoidance
// ---------------------------------------------------------------------------

const BANK_WIDTH: u32 = 32u;
const BANK_PADDING: u32 = 1u;

fn padded_index(idx: u32) -> u32 {
    return idx + (idx / BANK_WIDTH) * BANK_PADDING;
}

// ---------------------------------------------------------------------------
// Workgroup shared memory
// ---------------------------------------------------------------------------

// Max twiddles in workgroup = 4096 + padding
var<workgroup> twiddles_shared: array<vec2<u32>, 4224>;   // 4096 + 4096/32
var<workgroup> twiddles_prefetch: array<vec2<u32>, 4224>;

// For fused kernel: all twiddles
var<workgroup> all_twiddles: array<vec2<u32>, 4096>;

// ===========================================================================
// Forward NTT stage with hotset caching
// ===========================================================================

@compute @workgroup_size(256)
fn ntt_hotset_forward_stage(
    @builtin(local_invocation_id) lid_v: vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(num_workgroups)      nwg: vec3<u32>,
) {
    let lid = lid_v.x;
    let batch_idx = wgid.x;
    let N = ntt_params.N;
    let Q = u64_from(ntt_params.Q_lo, ntt_params.Q_hi);
    let mu = u64_from(ntt_params.mu_lo, ntt_params.mu_hi);
    let stage = ntt_params.stage;
    let prime_idx = ntt_params.prime_idx;

    let m = 1u << stage;
    let t = N >> (stage + 1u);

    let batch_data_offset = batch_idx * N;

    // Determine twiddle source
    let use_constant = (stage < 4u && m <= 8u);

    if (!use_constant) {
        // Cooperative load into workgroup memory with padding
        let loads = (m + 255u) / 256u;
        for (var i = 0u; i < loads; i++) {
            let tw_idx = lid + i * 256u;
            if (tw_idx < m) {
                let padded = padded_index(tw_idx);
                twiddles_shared[padded] = twiddles_buf[m + tw_idx];
            }
        }

        // Prefetch next stage twiddles if enabled
        if (ntt_params.prefetch_stage < ntt_params.log_N && ntt_params.prefetch_stage > stage) {
            let next_m = 1u << ntt_params.prefetch_stage;
            let pf_loads = (next_m + 255u) / 256u;
            for (var i = 0u; i < pf_loads; i++) {
                let tw_idx = lid + i * 256u;
                if (tw_idx < next_m && tw_idx < 4096u) {
                    let padded = padded_index(tw_idx);
                    twiddles_prefetch[padded] = twiddles_buf[next_m + tw_idx];
                }
            }
        }

        workgroupBarrier();
    }

    // Butterfly computation
    let butterflies_per_thread = (N / 2u + 255u) / 256u;

    for (var b = 0u; b < butterflies_per_thread; b++) {
        let bf_idx = lid + b * 256u;
        if (bf_idx >= N / 2u) { break; }

        let group = bf_idx / t;
        let elem = bf_idx % t;
        let idx_lo = (group << (ntt_params.log_N - stage)) + elem;
        let idx_hi = idx_lo + t;

        let lo = data[batch_data_offset + idx_lo];
        let hi = data[batch_data_offset + idx_hi];

        // Get twiddle from appropriate cache tier
        var tw: vec2<u32>;
        if (use_constant) {
            tw = cache.first_level_twiddles[prime_idx * 8u + group];
        } else {
            let padded = padded_index(group);
            tw = twiddles_shared[padded];
        }

        let hi_tw = barrett_mul(hi, tw, Q, mu);
        data[batch_data_offset + idx_lo] = mod_add(lo, hi_tw, Q);
        data[batch_data_offset + idx_hi] = mod_sub(lo, hi_tw, Q);
    }
}

// ===========================================================================
// Inverse NTT stage with hotset caching
// ===========================================================================

@compute @workgroup_size(256)
fn ntt_hotset_inverse_stage(
    @builtin(local_invocation_id) lid_v: vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {
    let lid = lid_v.x;
    let batch_idx = wgid.x;
    let N = ntt_params.N;
    let Q = u64_from(ntt_params.Q_lo, ntt_params.Q_hi);
    let mu = u64_from(ntt_params.mu_lo, ntt_params.mu_hi);
    let stage = ntt_params.stage;
    let prime_idx = ntt_params.prime_idx;

    let m = N >> (stage + 1u);
    let t = 1u << stage;

    let batch_data_offset = batch_idx * N;

    let use_constant = (stage >= ntt_params.log_N - 4u && m <= 8u);

    if (!use_constant) {
        let loads = (m + 255u) / 256u;
        for (var i = 0u; i < loads; i++) {
            let tw_idx = lid + i * 256u;
            if (tw_idx < m) {
                let padded = padded_index(tw_idx);
                twiddles_shared[padded] = twiddles_buf[m + tw_idx];
            }
        }
        workgroupBarrier();
    }

    let butterflies_per_thread = (N / 2u + 255u) / 256u;

    for (var b = 0u; b < butterflies_per_thread; b++) {
        let bf_idx = lid + b * 256u;
        if (bf_idx >= N / 2u) { break; }

        let group = bf_idx / t;
        let elem = bf_idx % t;
        let idx_lo = (group << (stage + 1u)) + elem;
        let idx_hi = idx_lo + t;

        let lo = data[batch_data_offset + idx_lo];
        let hi = data[batch_data_offset + idx_hi];

        var tw: vec2<u32>;
        if (use_constant) {
            tw = cache.first_level_inv_twiddles[prime_idx * 8u + group];
        } else {
            let padded = padded_index(group);
            tw = twiddles_shared[padded];
        }

        // Gentleman-Sande butterfly
        let sum = mod_add(lo, hi, Q);
        let diff = mod_sub(lo, hi, Q);
        data[batch_data_offset + idx_lo] = sum;
        data[batch_data_offset + idx_hi] = barrett_mul(diff, tw, Q, mu);
    }
}

// ===========================================================================
// Multi-stage fused NTT (all stages in one dispatch for N <= 4096)
// ===========================================================================

@compute @workgroup_size(256)
fn ntt_hotset_fused(
    @builtin(local_invocation_id) lid_v: vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {
    let lid = lid_v.x;
    let batch_idx = wgid.x;
    let N = ntt_params.N;
    let log_N = ntt_params.log_N;
    let Q = u64_from(ntt_params.Q_lo, ntt_params.Q_hi);
    let mu = u64_from(ntt_params.mu_lo, ntt_params.mu_hi);

    let batch_data_offset = batch_idx * N;

    // Load ALL twiddles into workgroup memory (N-1 total)
    let total_twiddles = N - 1u;
    let loads = (total_twiddles + 255u) / 256u;
    for (var i = 0u; i < loads; i++) {
        let tw_idx = lid + i * 256u;
        if (tw_idx < total_twiddles) {
            all_twiddles[tw_idx] = twiddles_buf[tw_idx];
        }
    }
    workgroupBarrier();

    // Process all stages
    for (var stage = 0u; stage < log_N; stage++) {
        let m = 1u << stage;
        let t = N >> (stage + 1u);
        let tw_base = m;

        let bpt = (N / 2u + 255u) / 256u;
        for (var b = 0u; b < bpt; b++) {
            let bf_idx = lid + b * 256u;
            if (bf_idx >= N / 2u) { break; }

            let group = bf_idx / t;
            let elem = bf_idx % t;
            let idx_lo = (group << (log_N - stage)) + elem;
            let idx_hi = idx_lo + t;

            let lo = data[batch_data_offset + idx_lo];
            let hi = data[batch_data_offset + idx_hi];
            let tw = all_twiddles[tw_base + group];

            let hi_tw = barrett_mul(hi, tw, Q, mu);
            data[batch_data_offset + idx_lo] = mod_add(lo, hi_tw, Q);
            data[batch_data_offset + idx_hi] = mod_sub(lo, hi_tw, Q);
        }

        // Device memory barrier between stages
        storageBarrier();
    }
}

// ===========================================================================
// N^{-1} scaling for INTT
// ===========================================================================

@compute @workgroup_size(256)
fn ntt_hotset_scale_ninv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = ntt_params.N * ntt_params.batch;
    let idx = gid.x;
    if (idx >= total) { return; }

    let Q = u64_from(ntt_params.Q_lo, ntt_params.Q_hi);
    let mu = u64_from(ntt_params.mu_lo, ntt_params.mu_hi);
    let N_inv = u64_from(ntt_params.N_inv_lo, ntt_params.N_inv_hi);

    data[idx] = barrett_mul(data[idx], N_inv, Q, mu);
}
