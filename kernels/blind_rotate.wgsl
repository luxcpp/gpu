// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// TFHE blind rotation (programmable bootstrapping) in WGSL.
// Ported from blind_rotate.metal.
// Includes external product, polynomial rotation, and sample extraction.

@group(0) @binding(0) var<storage, read_write> acc_poly: array<u32>;
@group(0) @binding(1) var<storage, read> ggsw: array<u32>;
@group(0) @binding(2) var<storage, read_write> temp_poly: array<u32>;
@group(0) @binding(3) var<uniform> params: BlindRotateParams;

struct BlindRotateParams {
    N: u32,
    k: u32,
    n: u32,
    l: u32,
    base_log: u32,
    num_samples: u32,
}

// Signed gadget decomposition
fn signed_decompose(x: u32, level: u32, base_log: u32) -> i32 {
    let Bg = 1u << base_log;
    let half_Bg = Bg >> 1u;
    let mask = Bg - 1u;
    let shift = 32u - (level + 1u) * base_log;
    let digit = (x >> shift) & mask;
    if (digit >= half_Bg) {
        return i32(digit) - i32(Bg);
    }
    return i32(digit);
}

// ============================================================================
// External product: GLWE x GGSW -> accumulated in temp_poly
// Each thread computes one coefficient of the output
// ============================================================================

@compute @workgroup_size(256)
fn external_product(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let N = params.N;
    let l = params.l;
    if (idx >= N) { return; }

    var result = 0u;
    for (var level = 0u; level < l; level = level + 1u) {
        let decomp_val = signed_decompose(acc_poly[idx], level, params.base_log);
        let ggsw_offset = level * N + idx;
        let ggsw_coeff = ggsw[ggsw_offset];
        result = result + u32(decomp_val) * ggsw_coeff;
    }
    temp_poly[idx] = result;
}

// ============================================================================
// Polynomial rotation in negacyclic ring Z_Q[X]/(X^N + 1)
// ============================================================================

var<workgroup> shared_rot: array<u32, 1024>;

@compute @workgroup_size(256)
fn rotate_polynomial(@builtin(global_invocation_id) gid: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>) {
    let sample_idx = gid.y;
    let tid = lid.x;
    let N = params.N;

    if (sample_idx >= params.num_samples || tid >= N) { return; }

    let acc_offset = sample_idx * N;
    let rotation = (params.n) % (2u * N); // rotation amount passed via params.n field

    var dst_idx: u32;
    var negate = false;

    if (rotation < N) {
        if (tid >= rotation) {
            dst_idx = tid - rotation;
            negate = false;
        } else {
            dst_idx = N - rotation + tid;
            negate = true;
        }
    } else {
        let r = rotation - N;
        if (tid >= r) {
            dst_idx = tid - r;
            negate = true;
        } else {
            dst_idx = N - r + tid;
            negate = false;
        }
    }

    let val = acc_poly[acc_offset + tid];
    shared_rot[dst_idx] = select(val, 0u - val, negate);

    workgroupBarrier();

    if (tid < N) {
        acc_poly[acc_offset + tid] = shared_rot[tid];
    }
}

// ============================================================================
// Sample extraction: extract LWE from GLWE at position 0
// ============================================================================

@compute @workgroup_size(256)
fn sample_extract(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let sample_idx = gid.y;
    let N = params.N;
    let k = params.k;

    if (idx >= N * k || sample_idx >= params.num_samples) { return; }

    let acc_offset = sample_idx * N * (k + 1u);
    let poly_idx = idx / N;
    let coeff_idx = idx % N;

    // LWE 'a' coefficients: a[i] = acc[0] for i=0, -acc[N-i] for i>0
    if (coeff_idx == 0u) {
        temp_poly[sample_idx * N * k + idx] = acc_poly[acc_offset + poly_idx * N];
    } else {
        temp_poly[sample_idx * N * k + idx] = 0u - acc_poly[acc_offset + poly_idx * N + N - coeff_idx];
    }
}

// ============================================================================
// Extract LWE body (b component)
// ============================================================================

@compute @workgroup_size(256)
fn extract_body(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sample_idx = gid.x;
    if (sample_idx >= params.num_samples) { return; }

    let acc_offset = sample_idx * params.N * (params.k + 1u);
    let body_offset = params.k * params.N;
    temp_poly[sample_idx] = acc_poly[acc_offset + body_offset];
}
