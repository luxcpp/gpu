// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Tensor operations compute shaders in WGSL.
// Matches tensor_ops.metal output byte-for-byte.
// Elementwise, unary, matmul, transpose, reduce, softmax, normalization.

@group(0) @binding(0) var<storage, read> in_a: array<f32>;
@group(0) @binding(1) var<storage, read> in_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;
// params.x = n (or M for matmul, batch for softmax/norm)
// params.y = K (matmul) or dim (softmax/norm) or cols (transpose) or inner (reduce_axis)
// params.z = N (matmul)
// params.w = mode (reduce: 0=sum 1=max 2=min 3=mean) or eps (bitcast f32 for norms)

// ============================================================================
// Elementwise binary ops
// ============================================================================

@compute @workgroup_size(256)
fn lux_add_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    out[idx] = in_a[idx] + in_b[idx];
}

@compute @workgroup_size(256)
fn lux_sub_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    out[idx] = in_a[idx] - in_b[idx];
}

@compute @workgroup_size(256)
fn lux_mul_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    out[idx] = in_a[idx] * in_b[idx];
}

@compute @workgroup_size(256)
fn lux_div_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    out[idx] = in_a[idx] / in_b[idx];
}

// ============================================================================
// Unary ops
// ============================================================================

@compute @workgroup_size(256)
fn lux_exp_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    out[idx] = exp(in_a[idx]);
}

@compute @workgroup_size(256)
fn lux_log_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    out[idx] = log(in_a[idx]);
}

@compute @workgroup_size(256)
fn lux_sqrt_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    out[idx] = sqrt(in_a[idx]);
}

@compute @workgroup_size(256)
fn lux_neg_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    out[idx] = -in_a[idx];
}

@compute @workgroup_size(256)
fn lux_abs_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    out[idx] = abs(in_a[idx]);
}

@compute @workgroup_size(256)
fn lux_tanh_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    out[idx] = tanh(in_a[idx]);
}

@compute @workgroup_size(256)
fn lux_sigmoid_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    out[idx] = 1.0 / (1.0 + exp(-in_a[idx]));
}

@compute @workgroup_size(256)
fn lux_relu_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    out[idx] = max(in_a[idx], 0.0);
}

@compute @workgroup_size(256)
fn lux_gelu_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    let x = in_a[idx];
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    out[idx] = 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(256)
fn lux_copy_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    out[idx] = in_a[idx];
}

// ============================================================================
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
// params.x = M, params.y = K, params.z = N
// ============================================================================

@compute @workgroup_size(16, 16)
fn lux_matmul_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    let M = params.x;
    let K = params.y;
    let N = params.z;

    if (row >= M || col >= N) { return; }

    var sum = 0.0f;
    for (var k = 0u; k < K; k = k + 1u) {
        sum = sum + in_a[row * K + k] * in_b[k * N + col];
    }
    out[row * N + col] = sum;
}

// ============================================================================
// Transpose: out[j,i] = in[i,j]
// params.x = rows, params.y = cols
// ============================================================================

@compute @workgroup_size(256)
fn lux_transpose_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let rows = params.x;
    let cols = params.y;
    if (idx >= rows * cols) { return; }

    let row = idx / cols;
    let col = idx % cols;
    out[col * rows + row] = in_a[row * cols + col];
}

// ============================================================================
// Full reduction: in_a[n] -> out[0]
// params.x = n, params.w = mode (0=sum, 1=max, 2=min, 3=mean)
// ============================================================================

@compute @workgroup_size(1)
fn lux_reduce_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.x;
    let mode = params.w;

    var acc = in_a[0];
    for (var i = 1u; i < n; i = i + 1u) {
        let v = in_a[i];
        if (mode == 0u || mode == 3u) {
            acc = acc + v;
        } else if (mode == 1u) {
            acc = max(acc, v);
        } else {
            acc = min(acc, v);
        }
    }
    if (mode == 3u) {
        acc = acc / f32(n);
    }
    out[0] = acc;
}

// ============================================================================
// Axis reduction: [outer, inner] -> [outer]
// params.x = outer_size, params.y = inner_size, params.w = mode (0=sum, 1=max)
// ============================================================================

@compute @workgroup_size(256)
fn lux_reduce_axis_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let outer = gid.x;
    if (outer >= params.x) { return; }

    let inner = params.y;
    let mode = params.w;
    let base = outer * inner;

    var acc = in_a[base];
    for (var i = 1u; i < inner; i = i + 1u) {
        let v = in_a[base + i];
        if (mode == 0u) {
            acc = acc + v;
        } else {
            acc = max(acc, v);
        }
    }
    out[outer] = acc;
}

// ============================================================================
// Softmax: out[b,i] = exp(in[b,i] - max) / sum(exp(...))
// params.x = batch_size, params.y = dim
// ============================================================================

@compute @workgroup_size(256)
fn lux_softmax_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = gid.x;
    if (batch >= params.x) { return; }

    let dim = params.y;
    let base = batch * dim;

    var m = in_a[base];
    for (var i = 1u; i < dim; i = i + 1u) {
        m = max(m, in_a[base + i]);
    }

    var s = 0.0f;
    for (var i = 0u; i < dim; i = i + 1u) {
        s = s + exp(in_a[base + i] - m);
    }

    for (var i = 0u; i < dim; i = i + 1u) {
        out[base + i] = exp(in_a[base + i] - m) / s;
    }
}

@compute @workgroup_size(256)
fn lux_log_softmax_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = gid.x;
    if (batch >= params.x) { return; }

    let dim = params.y;
    let base = batch * dim;

    var m = in_a[base];
    for (var i = 1u; i < dim; i = i + 1u) {
        m = max(m, in_a[base + i]);
    }

    var s = 0.0f;
    for (var i = 0u; i < dim; i = i + 1u) {
        s = s + exp(in_a[base + i] - m);
    }
    let log_s = log(s);

    for (var i = 0u; i < dim; i = i + 1u) {
        out[base + i] = (in_a[base + i] - m) - log_s;
    }
}

// ============================================================================
// Layer normalization
// params.x = batch_size, params.y = dim, params.w = eps (bitcast f32)
// in_a = input, in_b[0..dim-1] = gamma, in_b[dim..2dim-1] = beta
// ============================================================================

@compute @workgroup_size(256)
fn lux_layer_norm_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = gid.x;
    if (batch >= params.x) { return; }

    let dim = params.y;
    let eps = bitcast<f32>(params.w);
    let base = batch * dim;

    var mean = 0.0f;
    for (var i = 0u; i < dim; i = i + 1u) {
        mean = mean + in_a[base + i];
    }
    mean = mean / f32(dim);

    var variance = 0.0f;
    for (var i = 0u; i < dim; i = i + 1u) {
        let d = in_a[base + i] - mean;
        variance = variance + d * d;
    }
    variance = variance / f32(dim);

    let inv_std = 1.0 / sqrt(variance + eps);

    for (var i = 0u; i < dim; i = i + 1u) {
        let norm = (in_a[base + i] - mean) * inv_std;
        out[base + i] = norm * in_b[i] + in_b[dim + i];
    }
}

// ============================================================================
// RMS normalization
// params.x = batch_size, params.y = dim, params.w = eps (bitcast f32)
// in_a = input, in_b = weight
// ============================================================================

@compute @workgroup_size(256)
fn lux_rms_norm_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = gid.x;
    if (batch >= params.x) { return; }

    let dim = params.y;
    let eps = bitcast<f32>(params.w);
    let base = batch * dim;

    var sum_sq = 0.0f;
    for (var i = 0u; i < dim; i = i + 1u) {
        let v = in_a[base + i];
        sum_sq = sum_sq + v * v;
    }

    let inv_rms = 1.0 / sqrt(sum_sq / f32(dim) + eps);

    for (var i = 0u; i < dim; i = i + 1u) {
        out[base + i] = in_a[base + i] * inv_rms * in_b[i];
    }
}
