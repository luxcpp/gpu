// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Speculative Bootstrap Key Prefetching — WGSL compute shaders
// Ported from bsk_prefetch.metal.
// u64 emulated as vec2<u32>(lo, hi). No native u64 in WGSL.
//
// Double-buffered BSK storage with async memory copy:
//   Fetch BSK[i+1] while computing CMux with BSK[i]
//   Hides memory latency for all but first iteration

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<storage, read_write> dst: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read>       bsk: array<vec2<u32>>;
@group(0) @binding(2) var<uniform>             params: BSKPrefetchParams;

struct BSKPrefetchParams {
    N: u32,
    L: u32,
    n: u32,
    entry_size: u32,
    current_entry: u32,
    prefetch_entry: u32,
    Q_lo: u32, Q_hi: u32,
    mu_lo: u32, mu_hi: u32,
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
fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let al = a.x & 0xFFFFu; let ah = a.x >> 16u;
    let bl = b.x & 0xFFFFu; let bh = b.x >> 16u;
    let ll = al * bl;
    let mid = al * bh + ah * bl;
    let lo = ll + (mid << 16u);
    let hi = ah * bh + (mid >> 16u) + select(0u, 1u, lo < ll) + a.x * b.y + a.y * b.x;
    return vec2<u32>(lo, hi);
}
fn u64_mulhi(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let al = a.x & 0xFFFFu; let ah = a.x >> 16u;
    let bl = b.x & 0xFFFFu; let bh = b.x >> 16u;
    let p = vec2<u32>(ah * bh, 0u);
    let c1y = (al * bh + ah * bl) >> 16u;
    let c2y = (a.y * b.x);
    let c3y = (a.x * b.y);
    _ = c2y; _ = c3y;
    let hh = a.y * b.y;
    let cross1_hi = (al * b.y + a.y * bl) >> 0u;
    _ = cross1_hi;
    return vec2<u32>(p.x + c1y, p.y + hh);
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

// ===========================================================================
// Kernel 1: Async BSK Copy
// ===========================================================================
// Copies one BSK entry from source to destination buffer.
// Dispatch: grid = ceil(entry_size / 256), threads = 256

@compute @workgroup_size(256)
fn async_bsk_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let entry_size = params.entry_size;
    let entry_idx = params.prefetch_entry;
    let src_offset = entry_idx * entry_size;

    if (tid < entry_size) {
        dst[tid] = bsk[src_offset + tid];
    }
}

// ===========================================================================
// Kernel 2: CMux with Prefetch
// ===========================================================================
// CMux(selector, d0, d1) = d0 + ExternalProduct(d1 - d0, RGSW(selector))

@group(0) @binding(3) var<storage, read_write> acc: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read>       bsk_active: array<vec2<u32>>;
@group(0) @binding(5) var<uniform>             rotation_val: vec4<u32>;  // x = rotation

var<workgroup> cmux_shared: array<vec2<u32>, 4096>;  // 8*N for N=512

@compute @workgroup_size(256)
fn cmux_with_prefetch(
    @builtin(global_invocation_id)  gid: vec3<u32>,
    @builtin(local_invocation_id)   lid_v: vec3<u32>,
    @builtin(num_workgroups)        nwg: vec3<u32>,
) {
    let lid = lid_v.x;
    let grid_size = nwg.x * 256u;
    let N = params.N;
    let L = params.L;
    let Q = u64_from(params.Q_lo, params.Q_hi);
    let mu = u64_from(params.mu_lo, params.mu_hi);
    let rotation = i32(rotation_val.x);

    if (rotation == 0) { return; }

    let two_N = 2u * N;
    var rot = ((rotation % i32(two_N)) + i32(two_N)) % i32(two_N);

    // Shared memory layout: acc_snap[2*N], rotated[2*N], diff[2*N], prod[2*N]
    let off_snap = 0u;
    let off_rot = 2u * N;
    let off_diff = 4u * N;
    let off_prod = 6u * N;

    // Load accumulator snapshot
    for (var i = lid; i < 2u * N; i += 256u) {
        cmux_shared[off_snap + i] = acc[i];
    }
    workgroupBarrier();

    // Negacyclic rotation
    for (var c = 0u; c < 2u; c++) {
        for (var i = lid; i < N; i += 256u) {
            var src_idx = i32(i) - rot;
            var neg = false;
            if (src_idx < 0) { src_idx += i32(N); neg = !neg; }
            if (src_idx < 0) { src_idx += i32(N); neg = !neg; }
            if (src_idx >= i32(N)) { src_idx -= i32(N); neg = !neg; }
            let val = cmux_shared[off_snap + c * N + u32(src_idx)];
            cmux_shared[off_rot + c * N + i] = select(val, mod_sub(u64_zero(), val, Q), neg);
        }
    }
    workgroupBarrier();

    // Difference: rotated - acc
    for (var i = lid; i < 2u * N; i += 256u) {
        cmux_shared[off_diff + i] = mod_sub(cmux_shared[off_rot + i], cmux_shared[off_snap + i], Q);
    }
    workgroupBarrier();

    // Initialize product
    for (var i = lid; i < 2u * N; i += 256u) {
        cmux_shared[off_prod + i] = u64_zero();
    }
    workgroupBarrier();

    // External product: diff * RGSW (simplified digit extraction)
    let mask = u64_from(127u, 0u);  // base_log = 7

    for (var comp = 0u; comp < 2u; comp++) {
        for (var l = 0u; l < L; l++) {
            let rgsw_row_base = comp * L * 2u * N + l * 2u * N;

            for (var j_idx = lid; j_idx < N; j_idx += 256u) {
                let val = cmux_shared[off_diff + comp * N + j_idx];
                let shift_amt = l * 7u;
                let digit_lo = select((val.x >> shift_amt), ((val.x >> shift_amt) | (val.y << (32u - shift_amt))), shift_amt > 0u && shift_amt < 32u);
                let digit = u64_from(digit_lo & 127u, 0u);

                for (var out_c = 0u; out_c < 2u; out_c++) {
                    let rgsw_val = bsk_active[rgsw_row_base + out_c * N + j_idx];
                    let term = u64_mul(digit, rgsw_val);
                    cmux_shared[off_prod + out_c * N + j_idx] = mod_add(
                        cmux_shared[off_prod + out_c * N + j_idx], term, Q
                    );
                }
            }
            workgroupBarrier();
        }
    }

    // Update accumulator: acc = acc + prod
    for (var i = lid; i < 2u * N; i += 256u) {
        acc[i] = mod_add(cmux_shared[off_snap + i], cmux_shared[off_prod + i], Q);
    }
}

// ===========================================================================
// Kernel 3: Batch BSK Prefetch
// ===========================================================================

@group(0) @binding(6) var<storage, read> entry_indices: array<u32>;
@group(0) @binding(7) var<uniform>       batch_params: vec4<u32>; // x = batch_size

@compute @workgroup_size(256)
fn batch_bsk_prefetch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_x = gid.x;
    let batch_idx = gid.y;
    let batch_size = batch_params.x;

    if (batch_idx >= batch_size) { return; }

    let entry_size = params.entry_size;
    let entry_idx = entry_indices[batch_idx];
    if (entry_idx >= params.n) { return; }

    let src_offset = entry_idx * entry_size;
    let dst_offset = batch_idx * entry_size;

    for (var i = thread_x; i < entry_size; i += 256u) {
        dst[dst_offset + i] = bsk[src_offset + i];
    }
}

// ===========================================================================
// Kernel 4: Streaming Prefetch
// ===========================================================================

@group(0) @binding(8) var<storage, read> stream_src: array<vec2<u32>>;
@group(0) @binding(9) var<uniform>       stream_params: vec4<u32>; // x = num_elements

@compute @workgroup_size(256)
fn streaming_prefetch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let num_elements = stream_params.x;
    if (i < num_elements) {
        dst[i] = stream_src[i];
    }
}

// ===========================================================================
// Kernel 5: Signal Prefetch Complete
// ===========================================================================

@group(0) @binding(10) var<storage, read_write> completion_flag: array<atomic<u32>>;
@group(0) @binding(11) var<uniform>             signal_params: vec4<u32>; // x = expected_value

@compute @workgroup_size(1)
fn signal_prefetch_complete(@builtin(global_invocation_id) gid: vec3<u32>) {
    let expected = signal_params.x;
    atomicStore(&completion_flag[0], expected + 1u);
}
