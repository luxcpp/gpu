// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// DAG Executor — WGSL compute shaders
// Ported from dag_executor.metal.
// Batched FHE operations scheduled by the DAG scheduler.
// u64 emulated as vec2<u32>(lo, hi). No native u64 in WGSL.

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<storage, read_write> ciphertexts: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read>       ops: array<vec4<u32>>;  // BatchOp: input1, input2, output, level
@group(0) @binding(2) var<storage, read>       twiddles: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read>       precons: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read>       inv_twiddles: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read>       inv_precons: array<vec2<u32>>;
@group(0) @binding(6) var<uniform>             params: DAGBatchParams;

struct DAGBatchParams {
    Q_lo: u32, Q_hi: u32,
    barrett_mu_lo: u32, barrett_mu_hi: u32,
    N_inv_lo: u32, N_inv_hi: u32,
    N_inv_precon_lo: u32, N_inv_precon_hi: u32,
    N: u32, log_N: u32,
    poly_degree: u32, batch_size: u32,
    op_type: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
}

// Operation type constants (must match Go dag.OpType)
const DAG_OP_ADD: u32 = 0u;
const DAG_OP_SUB: u32 = 1u;
const DAG_OP_NEGATE: u32 = 6u;
const DAG_OP_NTT: u32 = 7u;
const DAG_OP_INTT: u32 = 8u;
const DAG_OP_COPY: u32 = 15u;

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

// ---------------------------------------------------------------------------
// Modular arithmetic
// ---------------------------------------------------------------------------

fn mod_add(a: vec2<u32>, b: vec2<u32>, Q: vec2<u32>) -> vec2<u32> {
    let s = u64_add(a, b);
    if (u64_gte(s, Q)) { return u64_sub(s, Q); }
    return s;
}
fn mod_sub(a: vec2<u32>, b: vec2<u32>, Q: vec2<u32>) -> vec2<u32> {
    if (u64_gte(a, b)) { return u64_sub(a, b); }
    return u64_sub(u64_add(a, Q), b);
}
fn mod_neg(a: vec2<u32>, Q: vec2<u32>) -> vec2<u32> {
    if (a.x == 0u && a.y == 0u) { return a; }
    return u64_sub(Q, a);
}
fn barrett_mul(a: vec2<u32>, omega: vec2<u32>, Q: vec2<u32>, precon: vec2<u32>) -> vec2<u32> {
    let q_hat = u64_mulhi(a, precon);
    let product = u64_mul(a, omega);
    var r = u64_sub(product, u64_mul(q_hat, Q));
    if (u64_gte(r, Q)) { r = u64_sub(r, Q); }
    return r;
}

// ---------------------------------------------------------------------------
// Workgroup shared memory for NTT
// ---------------------------------------------------------------------------

var<workgroup> ntt_shared: array<vec2<u32>, 1024>;

// ===========================================================================
// Batched Add
// ===========================================================================

@compute @workgroup_size(256)
fn dag_batch_add(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let op_idx = wgid.x;
    let poly_idx = wgid.y;
    let lid = lid_v.x;
    let N = params.N;
    let poly_degree = params.poly_degree;
    let Q = u64_from(params.Q_lo, params.Q_hi);

    if (op_idx >= params.batch_size || poly_idx > poly_degree) { return; }

    let op = ops[op_idx];
    let ct_stride = (poly_degree + 1u) * N;
    let in1_base = op.x * ct_stride + poly_idx * N;
    let in2_base = op.y * ct_stride + poly_idx * N;
    let out_base = op.z * ct_stride + poly_idx * N;

    for (var i = lid; i < N; i += 256u) {
        ciphertexts[out_base + i] = mod_add(ciphertexts[in1_base + i], ciphertexts[in2_base + i], Q);
    }
}

// ===========================================================================
// Batched Sub
// ===========================================================================

@compute @workgroup_size(256)
fn dag_batch_sub(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let op_idx = wgid.x;
    let poly_idx = wgid.y;
    let lid = lid_v.x;
    let N = params.N;
    let poly_degree = params.poly_degree;
    let Q = u64_from(params.Q_lo, params.Q_hi);

    if (op_idx >= params.batch_size || poly_idx > poly_degree) { return; }

    let op = ops[op_idx];
    let ct_stride = (poly_degree + 1u) * N;
    let in1_base = op.x * ct_stride + poly_idx * N;
    let in2_base = op.y * ct_stride + poly_idx * N;
    let out_base = op.z * ct_stride + poly_idx * N;

    for (var i = lid; i < N; i += 256u) {
        ciphertexts[out_base + i] = mod_sub(ciphertexts[in1_base + i], ciphertexts[in2_base + i], Q);
    }
}

// ===========================================================================
// Batched Negate
// ===========================================================================

@compute @workgroup_size(256)
fn dag_batch_negate(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let op_idx = wgid.x;
    let poly_idx = wgid.y;
    let lid = lid_v.x;
    let N = params.N;
    let poly_degree = params.poly_degree;
    let Q = u64_from(params.Q_lo, params.Q_hi);

    if (op_idx >= params.batch_size || poly_idx > poly_degree) { return; }

    let op = ops[op_idx];
    let ct_stride = (poly_degree + 1u) * N;
    let in_base = op.x * ct_stride + poly_idx * N;
    let out_base = op.z * ct_stride + poly_idx * N;

    for (var i = lid; i < N; i += 256u) {
        ciphertexts[out_base + i] = mod_neg(ciphertexts[in_base + i], Q);
    }
}

// ===========================================================================
// Batched Copy
// ===========================================================================

@compute @workgroup_size(256)
fn dag_batch_copy(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let op_idx = wgid.x;
    let poly_idx = wgid.y;
    let lid = lid_v.x;
    let N = params.N;
    let poly_degree = params.poly_degree;

    if (op_idx >= params.batch_size || poly_idx > poly_degree) { return; }

    let op = ops[op_idx];
    let ct_stride = (poly_degree + 1u) * N;
    let in_base = op.x * ct_stride + poly_idx * N;
    let out_base = op.z * ct_stride + poly_idx * N;

    for (var i = lid; i < N; i += 256u) {
        ciphertexts[out_base + i] = ciphertexts[in_base + i];
    }
}

// ===========================================================================
// Batched NTT (Forward, Cooley-Tukey)
// ===========================================================================

@compute @workgroup_size(256)
fn dag_batch_ntt(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let op_idx = wgid.x;
    let poly_idx = wgid.y;
    let lid = lid_v.x;
    let N = params.N;
    let log_N = params.log_N;
    let poly_degree = params.poly_degree;
    let Q = u64_from(params.Q_lo, params.Q_hi);

    if (op_idx >= params.batch_size || poly_idx > poly_degree) { return; }

    let op = ops[op_idx];
    let ct_stride = (poly_degree + 1u) * N;
    let in_base = op.x * ct_stride + poly_idx * N;
    let out_base = op.z * ct_stride + poly_idx * N;

    // Load to shared memory
    for (var i = lid; i < N; i += 256u) {
        ntt_shared[i] = ciphertexts[in_base + i];
    }
    workgroupBarrier();

    // NTT stages (Cooley-Tukey)
    for (var stage = 0u; stage < log_N; stage++) {
        let m = 1u << stage;
        let t = N >> (stage + 1u);

        for (var bf = lid; bf < N / 2u; bf += 256u) {
            let group = bf / t;
            let j = bf % t;
            let idx_lo = group * (2u * t) + j;
            let idx_hi = idx_lo + t;
            let tw_idx = m + group;

            let lo = ntt_shared[idx_lo];
            let hi = ntt_shared[idx_hi];
            let hi_tw = barrett_mul(hi, twiddles[tw_idx], Q, precons[tw_idx]);

            ntt_shared[idx_lo] = mod_add(lo, hi_tw, Q);
            ntt_shared[idx_hi] = mod_sub(lo, hi_tw, Q);
        }
        workgroupBarrier();
    }

    // Write back
    for (var i = lid; i < N; i += 256u) {
        ciphertexts[out_base + i] = ntt_shared[i];
    }
}

// ===========================================================================
// Batched INTT (Inverse, Gentleman-Sande)
// ===========================================================================

@compute @workgroup_size(256)
fn dag_batch_intt(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let op_idx = wgid.x;
    let poly_idx = wgid.y;
    let lid = lid_v.x;
    let N = params.N;
    let log_N = params.log_N;
    let poly_degree = params.poly_degree;
    let Q = u64_from(params.Q_lo, params.Q_hi);
    let N_inv = u64_from(params.N_inv_lo, params.N_inv_hi);
    let N_inv_pre = u64_from(params.N_inv_precon_lo, params.N_inv_precon_hi);

    if (op_idx >= params.batch_size || poly_idx > poly_degree) { return; }

    let op = ops[op_idx];
    let ct_stride = (poly_degree + 1u) * N;
    let in_base = op.x * ct_stride + poly_idx * N;
    let out_base = op.z * ct_stride + poly_idx * N;

    // Load to shared
    for (var i = lid; i < N; i += 256u) {
        ntt_shared[i] = ciphertexts[in_base + i];
    }
    workgroupBarrier();

    // INTT stages (Gentleman-Sande)
    for (var stage = 0u; stage < log_N; stage++) {
        let m = N >> (stage + 1u);
        let t = 1u << stage;

        for (var bf = lid; bf < N / 2u; bf += 256u) {
            let group = bf / t;
            let j = bf % t;
            let idx_lo = group * (2u * t) + j;
            let idx_hi = idx_lo + t;
            let tw_idx = m + group;

            let lo = ntt_shared[idx_lo];
            let hi = ntt_shared[idx_hi];

            let sum = mod_add(lo, hi, Q);
            let diff = mod_sub(lo, hi, Q);
            let diff_tw = barrett_mul(diff, inv_twiddles[tw_idx], Q, inv_precons[tw_idx]);

            ntt_shared[idx_lo] = sum;
            ntt_shared[idx_hi] = diff_tw;
        }
        workgroupBarrier();
    }

    // Scale by N^{-1} and write back
    for (var i = lid; i < N; i += 256u) {
        ciphertexts[out_base + i] = barrett_mul(ntt_shared[i], N_inv, Q, N_inv_pre);
    }
}

// ===========================================================================
// Unified Dispatch (for small batches or heterogeneous ops)
// ===========================================================================

@compute @workgroup_size(256)
fn dag_dispatch(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let op_idx = wgid.x;
    let poly_idx = wgid.y;
    let lid = lid_v.x;
    let N = params.N;
    let poly_degree = params.poly_degree;
    let Q = u64_from(params.Q_lo, params.Q_hi);

    if (op_idx >= params.batch_size || poly_idx > poly_degree) { return; }

    let op = ops[op_idx];
    let ct_stride = (poly_degree + 1u) * N;

    switch (params.op_type) {
        case DAG_OP_ADD: {
            let in1_base = op.x * ct_stride + poly_idx * N;
            let in2_base = op.y * ct_stride + poly_idx * N;
            let out_base = op.z * ct_stride + poly_idx * N;
            for (var i = lid; i < N; i += 256u) {
                ciphertexts[out_base + i] = mod_add(ciphertexts[in1_base + i], ciphertexts[in2_base + i], Q);
            }
        }
        case DAG_OP_SUB: {
            let in1_base = op.x * ct_stride + poly_idx * N;
            let in2_base = op.y * ct_stride + poly_idx * N;
            let out_base = op.z * ct_stride + poly_idx * N;
            for (var i = lid; i < N; i += 256u) {
                ciphertexts[out_base + i] = mod_sub(ciphertexts[in1_base + i], ciphertexts[in2_base + i], Q);
            }
        }
        case DAG_OP_NEGATE: {
            let in_base = op.x * ct_stride + poly_idx * N;
            let out_base = op.z * ct_stride + poly_idx * N;
            for (var i = lid; i < N; i += 256u) {
                ciphertexts[out_base + i] = mod_neg(ciphertexts[in_base + i], Q);
            }
        }
        case DAG_OP_COPY: {
            let in_base = op.x * ct_stride + poly_idx * N;
            let out_base = op.z * ct_stride + poly_idx * N;
            for (var i = lid; i < N; i += 256u) {
                ciphertexts[out_base + i] = ciphertexts[in_base + i];
            }
        }
        default: {}
    }
}

// ===========================================================================
// Multi-Operation Batch (heterogeneous ops per-entry)
// ===========================================================================

@group(0) @binding(7) var<storage, read> multi_ops: array<vec4<u32>>; // input1, input2, output, op_type

@compute @workgroup_size(256)
fn dag_multi_dispatch(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let op_idx = wgid.x;
    let poly_idx = wgid.y;
    let lid = lid_v.x;
    let N = params.N;
    let poly_degree = params.poly_degree;
    let Q = u64_from(params.Q_lo, params.Q_hi);

    if (op_idx >= params.batch_size || poly_idx > poly_degree) { return; }

    let op = multi_ops[op_idx];
    let ct_stride = (poly_degree + 1u) * N;
    let in1_base = op.x * ct_stride + poly_idx * N;
    let in2_base = op.y * ct_stride + poly_idx * N;
    let out_base = op.z * ct_stride + poly_idx * N;

    switch (op.w) {
        case DAG_OP_ADD: {
            for (var i = lid; i < N; i += 256u) {
                ciphertexts[out_base + i] = mod_add(ciphertexts[in1_base + i], ciphertexts[in2_base + i], Q);
            }
        }
        case DAG_OP_SUB: {
            for (var i = lid; i < N; i += 256u) {
                ciphertexts[out_base + i] = mod_sub(ciphertexts[in1_base + i], ciphertexts[in2_base + i], Q);
            }
        }
        case DAG_OP_NEGATE: {
            for (var i = lid; i < N; i += 256u) {
                ciphertexts[out_base + i] = mod_neg(ciphertexts[in1_base + i], Q);
            }
        }
        case DAG_OP_COPY: {
            for (var i = lid; i < N; i += 256u) {
                ciphertexts[out_base + i] = ciphertexts[in1_base + i];
            }
        }
        default: {}
    }
}

// ===========================================================================
// Fused Add-Add: out = (in1 + in2) + in3
// ===========================================================================

@group(0) @binding(8) var<storage, read> fused_ops: array<vec4<u32>>; // input1, input2, input3, output

@compute @workgroup_size(256)
fn dag_fused_add_add(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
) {
    let op_idx = wgid.x;
    let poly_idx = wgid.y;
    let lid = lid_v.x;
    let N = params.N;
    let poly_degree = params.poly_degree;
    let Q = u64_from(params.Q_lo, params.Q_hi);

    if (op_idx >= params.batch_size || poly_idx > poly_degree) { return; }

    let op = fused_ops[op_idx];
    let ct_stride = (poly_degree + 1u) * N;
    let in1_base = op.x * ct_stride + poly_idx * N;
    let in2_base = op.y * ct_stride + poly_idx * N;
    let in3_base = op.z * ct_stride + poly_idx * N;
    let out_base = op.w * ct_stride + poly_idx * N;

    for (var i = lid; i < N; i += 256u) {
        let tmp = mod_add(ciphertexts[in1_base + i], ciphertexts[in2_base + i], Q);
        ciphertexts[out_base + i] = mod_add(tmp, ciphertexts[in3_base + i], Q);
    }
}
