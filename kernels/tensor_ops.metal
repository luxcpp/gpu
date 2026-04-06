// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Metal Tensor Operations
// Optimized GPU kernels for ML tensor operations on Apple Silicon

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Configuration
// =============================================================================

constant uint BLOCK_SIZE = 256;
constant uint WARP_SIZE = 32;  // Simdgroup width
constant uint TILE_SIZE = 16;

// =============================================================================
// Elementwise Binary Operations
// =============================================================================

kernel void lux_add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

kernel void lux_sub_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

kernel void lux_mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

kernel void lux_div_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

// Vectorized versions for better memory throughput
kernel void lux_add_f32_vec4(
    device const float4* a [[buffer(0)]],
    device const float4* b [[buffer(1)]],
    device float4* out [[buffer(2)]],
    constant uint& n4 [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n4) {
        out[idx] = a[idx] + b[idx];
    }
}

kernel void lux_mul_f32_vec4(
    device const float4* a [[buffer(0)]],
    device const float4* b [[buffer(1)]],
    device float4* out [[buffer(2)]],
    constant uint& n4 [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n4) {
        out[idx] = a[idx] * b[idx];
    }
}

// =============================================================================
// Unary Operations
// =============================================================================

kernel void lux_exp_f32(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n) out[idx] = exp(in[idx]);
}

kernel void lux_log_f32(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n) out[idx] = log(in[idx]);
}

kernel void lux_sqrt_f32(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n) out[idx] = sqrt(in[idx]);
}

kernel void lux_neg_f32(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n) out[idx] = -in[idx];
}

kernel void lux_abs_f32(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n) out[idx] = abs(in[idx]);
}

kernel void lux_tanh_f32(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n) out[idx] = tanh(in[idx]);
}

kernel void lux_sigmoid_f32(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n) out[idx] = 1.0f / (1.0f + exp(-in[idx]));
}

kernel void lux_relu_f32(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n) out[idx] = max(0.0f, in[idx]);
}

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
kernel void lux_gelu_f32(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n) {
        float x = in[idx];
        float x3 = x * x * x;
        // sqrt(2/pi) = 0.7978845608
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        out[idx] = 0.5f * x * (1.0f + tanh(inner));
    }
}

// =============================================================================
// Copy
// =============================================================================

kernel void lux_copy_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < n) dst[idx] = src[idx];
}

// =============================================================================
// Tiled Matrix Multiplication
// =============================================================================

// GEMM with shared memory tiling
// C[M,N] = A[M,K] @ B[K,N]
kernel void lux_matmul_tiled_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& N [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    int row = gid.y * TILE_SIZE + tid.y;
    int col = gid.x * TILE_SIZE + tid.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A into shared memory
        int a_col = t * TILE_SIZE + tid.x;
        if (row < M && a_col < K) {
            As[tid.y][tid.x] = A[row * K + a_col];
        } else {
            As[tid.y][tid.x] = 0.0f;
        }

        // Load tile of B into shared memory
        int b_row = t * TILE_SIZE + tid.y;
        if (b_row < K && col < N) {
            Bs[tid.y][tid.x] = B[b_row * N + col];
        } else {
            Bs[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[tid.y][k] * Bs[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// =============================================================================
// Matrix Transpose
// =============================================================================

// Tiled transpose with bank conflict avoidance
kernel void lux_transpose_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& rows [[buffer(2)]],
    constant int& cols [[buffer(3)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    threadgroup float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts

    int x = gid.x * TILE_SIZE + tid.x;
    int y = gid.y * TILE_SIZE + tid.y;

    // Read into shared memory (coalesced read)
    if (x < cols && y < rows) {
        tile[tid.y][tid.x] = input[y * cols + x];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Transposed coordinates
    x = gid.y * TILE_SIZE + tid.x;
    y = gid.x * TILE_SIZE + tid.y;

    // Write from shared memory (coalesced write)
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[tid.x][tid.y];
    }
}

// =============================================================================
// Reduction Operations - Full Array
// =============================================================================

// Sum reduction using simdgroup operations
kernel void lux_reduce_sum_f32(
    device const float* input [[buffer(0)]],
    device atomic_float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial_sums[32];

    // Grid-stride loop to accumulate
    float sum = 0.0f;
    for (uint i = idx; i < n; i += BLOCK_SIZE * 1024) {
        sum += input[i];
    }

    // Simdgroup reduction
    sum = simd_sum(sum);

    // First lane writes to shared memory
    if (simd_lane == 0) {
        partial_sums[simd_group] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by first simdgroup
    if (simd_group == 0) {
        sum = (simd_lane < (BLOCK_SIZE / 32)) ? partial_sums[simd_lane] : 0.0f;
        sum = simd_sum(sum);

        if (simd_lane == 0) {
            atomic_fetch_add_explicit(output, sum, memory_order_relaxed);
        }
    }
}

kernel void lux_reduce_max_f32(
    device const float* input [[buffer(0)]],
    device atomic_float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial_max[32];

    float local_max = -INFINITY;
    for (uint i = idx; i < n; i += BLOCK_SIZE * 1024) {
        local_max = max(local_max, input[i]);
    }

    local_max = simd_max(local_max);

    if (simd_lane == 0) {
        partial_max[simd_group] = local_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        local_max = (simd_lane < (BLOCK_SIZE / 32)) ? partial_max[simd_lane] : -INFINITY;
        local_max = simd_max(local_max);

        if (simd_lane == 0) {
            // Atomic max using compare-and-swap loop
            float old_val = atomic_load_explicit(output, memory_order_relaxed);
            while (old_val < local_max) {
                if (atomic_compare_exchange_weak_explicit(output, &old_val, local_max,
                    memory_order_relaxed, memory_order_relaxed)) {
                    break;
                }
            }
        }
    }
}

kernel void lux_reduce_min_f32(
    device const float* input [[buffer(0)]],
    device atomic_float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial_min[32];

    float local_min = INFINITY;
    for (uint i = idx; i < n; i += BLOCK_SIZE * 1024) {
        local_min = min(local_min, input[i]);
    }

    local_min = simd_min(local_min);

    if (simd_lane == 0) {
        partial_min[simd_group] = local_min;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        local_min = (simd_lane < (BLOCK_SIZE / 32)) ? partial_min[simd_lane] : INFINITY;
        local_min = simd_min(local_min);

        if (simd_lane == 0) {
            float old_val = atomic_load_explicit(output, memory_order_relaxed);
            while (old_val > local_min) {
                if (atomic_compare_exchange_weak_explicit(output, &old_val, local_min,
                    memory_order_relaxed, memory_order_relaxed)) {
                    break;
                }
            }
        }
    }
}

// =============================================================================
// Reduction Operations - Axis Reduction (Last Axis)
// =============================================================================

// Reduce along last axis: [outer_size, inner_size] -> [outer_size]
kernel void lux_reduce_sum_axis_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& inner_size [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (gid >= outer_size) return;

    threadgroup float partial_sums[32];

    device const float* row = input + gid * inner_size;

    // Accumulate
    float sum = 0.0f;
    for (uint i = tid; i < inner_size; i += BLOCK_SIZE) {
        sum += row[i];
    }

    // Simdgroup reduction
    sum = simd_sum(sum);

    if (simd_lane == 0) {
        partial_sums[simd_group] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction
    if (simd_group == 0) {
        sum = (simd_lane < (BLOCK_SIZE / 32)) ? partial_sums[simd_lane] : 0.0f;
        sum = simd_sum(sum);

        if (simd_lane == 0) {
            output[gid] = sum;
        }
    }
}

kernel void lux_reduce_max_axis_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& inner_size [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (gid >= outer_size) return;

    threadgroup float partial_max[32];

    device const float* row = input + gid * inner_size;

    float local_max = -INFINITY;
    for (uint i = tid; i < inner_size; i += BLOCK_SIZE) {
        local_max = max(local_max, row[i]);
    }

    local_max = simd_max(local_max);

    if (simd_lane == 0) {
        partial_max[simd_group] = local_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        local_max = (simd_lane < (BLOCK_SIZE / 32)) ? partial_max[simd_lane] : -INFINITY;
        local_max = simd_max(local_max);

        if (simd_lane == 0) {
            output[gid] = local_max;
        }
    }
}

kernel void lux_reduce_mean_axis_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& inner_size [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (gid >= outer_size) return;

    threadgroup float partial_sums[32];

    device const float* row = input + gid * inner_size;

    float sum = 0.0f;
    for (uint i = tid; i < inner_size; i += BLOCK_SIZE) {
        sum += row[i];
    }

    sum = simd_sum(sum);

    if (simd_lane == 0) {
        partial_sums[simd_group] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        sum = (simd_lane < (BLOCK_SIZE / 32)) ? partial_sums[simd_lane] : 0.0f;
        sum = simd_sum(sum);

        if (simd_lane == 0) {
            output[gid] = sum / float(inner_size);
        }
    }
}

// =============================================================================
// Softmax (Numerically Stable)
// =============================================================================

// Softmax along last dimension: [batch_size, dim]
kernel void lux_softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (gid >= batch_size) return;

    threadgroup float shared_max[32];
    threadgroup float shared_sum[32];
    threadgroup float s_max;
    threadgroup float s_sum;

    device const float* x = input + gid * dim;
    device float* y = output + gid * dim;

    // Pass 1: Find max
    float local_max = -INFINITY;
    for (uint i = tid; i < dim; i += BLOCK_SIZE) {
        local_max = max(local_max, x[i]);
    }

    local_max = simd_max(local_max);

    if (simd_lane == 0) {
        shared_max[simd_group] = local_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        local_max = (simd_lane < (BLOCK_SIZE / 32)) ? shared_max[simd_lane] : -INFINITY;
        local_max = simd_max(local_max);
        if (simd_lane == 0) s_max = local_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float max_val = s_max;

    // Pass 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < dim; i += BLOCK_SIZE) {
        float exp_val = exp(x[i] - max_val);
        y[i] = exp_val;
        local_sum += exp_val;
    }

    local_sum = simd_sum(local_sum);

    if (simd_lane == 0) {
        shared_sum[simd_group] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        local_sum = (simd_lane < (BLOCK_SIZE / 32)) ? shared_sum[simd_lane] : 0.0f;
        local_sum = simd_sum(local_sum);
        if (simd_lane == 0) s_sum = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sum = 1.0f / s_sum;

    // Normalize
    for (uint i = tid; i < dim; i += BLOCK_SIZE) {
        y[i] *= inv_sum;
    }
}

// Log-softmax: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
kernel void lux_log_softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (gid >= batch_size) return;

    threadgroup float shared_max[32];
    threadgroup float shared_sum[32];
    threadgroup float s_max;
    threadgroup float s_log_sum;

    device const float* x = input + gid * dim;
    device float* y = output + gid * dim;

    // Find max
    float local_max = -INFINITY;
    for (uint i = tid; i < dim; i += BLOCK_SIZE) {
        local_max = max(local_max, x[i]);
    }

    local_max = simd_max(local_max);

    if (simd_lane == 0) {
        shared_max[simd_group] = local_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        local_max = (simd_lane < (BLOCK_SIZE / 32)) ? shared_max[simd_lane] : -INFINITY;
        local_max = simd_max(local_max);
        if (simd_lane == 0) s_max = local_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float max_val = s_max;

    // Compute sum(exp(x - max))
    float local_sum = 0.0f;
    for (uint i = tid; i < dim; i += BLOCK_SIZE) {
        local_sum += exp(x[i] - max_val);
    }

    local_sum = simd_sum(local_sum);

    if (simd_lane == 0) {
        shared_sum[simd_group] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        local_sum = (simd_lane < (BLOCK_SIZE / 32)) ? shared_sum[simd_lane] : 0.0f;
        local_sum = simd_sum(local_sum);
        if (simd_lane == 0) s_log_sum = log(local_sum);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float log_sum = s_log_sum;

    // Compute log softmax
    for (uint i = tid; i < dim; i += BLOCK_SIZE) {
        y[i] = x[i] - max_val - log_sum;
    }
}

// =============================================================================
// Layer Normalization
// =============================================================================

kernel void lux_layer_norm_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (gid >= batch_size) return;

    threadgroup float shared_data[32];
    threadgroup float s_mean;
    threadgroup float s_var;

    device const float* x = input + gid * dim;
    device float* y = output + gid * dim;

    // Compute mean
    float local_sum = 0.0f;
    for (uint i = tid; i < dim; i += BLOCK_SIZE) {
        local_sum += x[i];
    }

    local_sum = simd_sum(local_sum);

    if (simd_lane == 0) {
        shared_data[simd_group] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        local_sum = (simd_lane < (BLOCK_SIZE / 32)) ? shared_data[simd_lane] : 0.0f;
        local_sum = simd_sum(local_sum);
        if (simd_lane == 0) s_mean = local_sum / float(dim);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = s_mean;

    // Compute variance
    float local_var = 0.0f;
    for (uint i = tid; i < dim; i += BLOCK_SIZE) {
        float diff = x[i] - mean;
        local_var += diff * diff;
    }

    local_var = simd_sum(local_var);

    if (simd_lane == 0) {
        shared_data[simd_group] = local_var;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        local_var = (simd_lane < (BLOCK_SIZE / 32)) ? shared_data[simd_lane] : 0.0f;
        local_var = simd_sum(local_var);
        if (simd_lane == 0) s_var = local_var / float(dim);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_std = rsqrt(s_var + eps);

    // Normalize and scale
    for (uint i = tid; i < dim; i += BLOCK_SIZE) {
        float normalized = (x[i] - mean) * inv_std;
        y[i] = normalized * gamma[i] + beta[i];
    }
}

// =============================================================================
// RMS Normalization
// =============================================================================

kernel void lux_rms_norm_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (gid >= batch_size) return;

    threadgroup float shared_data[32];
    threadgroup float s_rms;

    device const float* x = input + gid * dim;
    device float* y = output + gid * dim;

    // Compute sum of squares
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += BLOCK_SIZE) {
        float val = x[i];
        local_sum_sq += val * val;
    }

    local_sum_sq = simd_sum(local_sum_sq);

    if (simd_lane == 0) {
        shared_data[simd_group] = local_sum_sq;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        local_sum_sq = (simd_lane < (BLOCK_SIZE / 32)) ? shared_data[simd_lane] : 0.0f;
        local_sum_sq = simd_sum(local_sum_sq);
        if (simd_lane == 0) s_rms = rsqrt(local_sum_sq / float(dim) + eps);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_scale = s_rms;

    // Scale
    for (uint i = tid; i < dim; i += BLOCK_SIZE) {
        y[i] = x[i] * rms_scale * weight[i];
    }
}
