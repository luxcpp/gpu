// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// CUDA Tensor Operations - Matches tensor_ops.metal output byte-for-byte
// GPU kernels for ML tensor operations on NVIDIA GPUs

#include <cstdint>
#include <cfloat>
#include <cmath>

#ifdef __CUDA_ARCH__

static const uint32_t BLOCK_SIZE = 256;
static const uint32_t WARP_SIZE = 32;
static const uint32_t TILE_SIZE = 16;

// =============================================================================
// Warp-level reduction primitives
// =============================================================================

__device__ __forceinline__
float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__
float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__
float warp_reduce_min(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// =============================================================================
// Elementwise Binary Operations
// =============================================================================

extern "C" __global__
void lux_add_f32(const float* a, const float* b, float* out, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

extern "C" __global__
void lux_sub_f32(const float* a, const float* b, float* out, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] - b[idx];
}

extern "C" __global__
void lux_mul_f32(const float* a, const float* b, float* out, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}

extern "C" __global__
void lux_div_f32(const float* a, const float* b, float* out, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] / b[idx];
}

// Vectorized versions for better memory throughput
extern "C" __global__
void lux_add_f32_vec4(const float4* a, const float4* b, float4* out, uint32_t n4) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 va = a[idx];
        float4 vb = b[idx];
        out[idx] = make_float4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w);
    }
}

extern "C" __global__
void lux_mul_f32_vec4(const float4* a, const float4* b, float4* out, uint32_t n4) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 va = a[idx];
        float4 vb = b[idx];
        out[idx] = make_float4(va.x * vb.x, va.y * vb.y, va.z * vb.z, va.w * vb.w);
    }
}

// =============================================================================
// Unary Operations
// =============================================================================

extern "C" __global__
void lux_exp_f32(const float* in, float* out, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = expf(in[idx]);
}

extern "C" __global__
void lux_log_f32(const float* in, float* out, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = logf(in[idx]);
}

extern "C" __global__
void lux_sqrt_f32(const float* in, float* out, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = sqrtf(in[idx]);
}

extern "C" __global__
void lux_neg_f32(const float* in, float* out, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = -in[idx];
}

extern "C" __global__
void lux_abs_f32(const float* in, float* out, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fabsf(in[idx]);
}

extern "C" __global__
void lux_tanh_f32(const float* in, float* out, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = tanhf(in[idx]);
}

extern "C" __global__
void lux_sigmoid_f32(const float* in, float* out, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = 1.0f / (1.0f + expf(-in[idx]));
}

extern "C" __global__
void lux_relu_f32(const float* in, float* out, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fmaxf(0.0f, in[idx]);
}

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
extern "C" __global__
void lux_gelu_f32(const float* in, float* out, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        out[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// =============================================================================
// Copy
// =============================================================================

extern "C" __global__
void lux_copy_f32(const float* src, float* dst, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

// =============================================================================
// Tiled Matrix Multiplication: C[M,N] = A[M,K] @ B[K,N]
// =============================================================================

extern "C" __global__
void lux_matmul_tiled_f32(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// =============================================================================
// Matrix Transpose (tiled, bank-conflict free)
// =============================================================================

extern "C" __global__
void lux_transpose_f32(
    const float* input, float* output,
    int rows, int cols
) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 avoids bank conflicts

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Coalesced read
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    // Transposed coordinates
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    // Coalesced write
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// =============================================================================
// Reduction Operations - Full Array
// =============================================================================

extern "C" __global__
void lux_reduce_sum_f32(
    const float* input, float* output, uint32_t n
) {
    __shared__ float partial_sums[32];

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t lane = tid % WARP_SIZE;
    uint32_t warp_id = tid / WARP_SIZE;

    // Grid-stride loop
    float sum = 0.0f;
    for (uint32_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // Warp reduction
    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        partial_sums[warp_id] = sum;
    }

    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        sum = (lane < (BLOCK_SIZE / 32)) ? partial_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);

        if (lane == 0) {
            atomicAdd(output, sum);
        }
    }
}

extern "C" __global__
void lux_reduce_max_f32(
    const float* input, float* output, uint32_t n
) {
    __shared__ float partial_max[32];

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t lane = tid % WARP_SIZE;
    uint32_t warp_id = tid / WARP_SIZE;

    float local_max = -INFINITY;
    for (uint32_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }

    local_max = warp_reduce_max(local_max);

    if (lane == 0) {
        partial_max[warp_id] = local_max;
    }

    __syncthreads();

    if (warp_id == 0) {
        local_max = (lane < (BLOCK_SIZE / 32)) ? partial_max[lane] : -INFINITY;
        local_max = warp_reduce_max(local_max);

        if (lane == 0) {
            // Atomic max via CAS loop (matches Metal atomic_compare_exchange pattern)
            int* addr = (int*)output;
            int expected = __float_as_int(*output);
            while (local_max > __int_as_float(expected)) {
                int old = atomicCAS(addr, expected, __float_as_int(local_max));
                if (old == expected) break;
                expected = old;
            }
        }
    }
}

extern "C" __global__
void lux_reduce_min_f32(
    const float* input, float* output, uint32_t n
) {
    __shared__ float partial_min[32];

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t lane = tid % WARP_SIZE;
    uint32_t warp_id = tid / WARP_SIZE;

    float local_min = INFINITY;
    for (uint32_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        local_min = fminf(local_min, input[i]);
    }

    local_min = warp_reduce_min(local_min);

    if (lane == 0) {
        partial_min[warp_id] = local_min;
    }

    __syncthreads();

    if (warp_id == 0) {
        local_min = (lane < (BLOCK_SIZE / 32)) ? partial_min[lane] : INFINITY;
        local_min = warp_reduce_min(local_min);

        if (lane == 0) {
            // Atomic min via CAS loop
            int* addr = (int*)output;
            int expected = __float_as_int(*output);
            while (local_min < __int_as_float(expected)) {
                int old = atomicCAS(addr, expected, __float_as_int(local_min));
                if (old == expected) break;
                expected = old;
            }
        }
    }
}

// =============================================================================
// Reduction Operations - Axis Reduction (Last Axis)
// =============================================================================

extern "C" __global__
void lux_reduce_sum_axis_f32(
    const float* input, float* output,
    uint32_t outer_size, uint32_t inner_size
) {
    uint32_t gid = blockIdx.x;
    if (gid >= outer_size) return;

    __shared__ float partial_sums[32];

    uint32_t tid = threadIdx.x;
    uint32_t lane = tid % WARP_SIZE;
    uint32_t warp_id = tid / WARP_SIZE;

    const float* row = input + gid * inner_size;

    float sum = 0.0f;
    for (uint32_t i = tid; i < inner_size; i += BLOCK_SIZE) {
        sum += row[i];
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        partial_sums[warp_id] = sum;
    }

    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < (BLOCK_SIZE / 32)) ? partial_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);

        if (lane == 0) {
            output[gid] = sum;
        }
    }
}

extern "C" __global__
void lux_reduce_max_axis_f32(
    const float* input, float* output,
    uint32_t outer_size, uint32_t inner_size
) {
    uint32_t gid = blockIdx.x;
    if (gid >= outer_size) return;

    __shared__ float partial_max[32];

    uint32_t tid = threadIdx.x;
    uint32_t lane = tid % WARP_SIZE;
    uint32_t warp_id = tid / WARP_SIZE;

    const float* row = input + gid * inner_size;

    float local_max = -INFINITY;
    for (uint32_t i = tid; i < inner_size; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, row[i]);
    }

    local_max = warp_reduce_max(local_max);

    if (lane == 0) {
        partial_max[warp_id] = local_max;
    }

    __syncthreads();

    if (warp_id == 0) {
        local_max = (lane < (BLOCK_SIZE / 32)) ? partial_max[lane] : -INFINITY;
        local_max = warp_reduce_max(local_max);

        if (lane == 0) {
            output[gid] = local_max;
        }
    }
}

extern "C" __global__
void lux_reduce_mean_axis_f32(
    const float* input, float* output,
    uint32_t outer_size, uint32_t inner_size
) {
    uint32_t gid = blockIdx.x;
    if (gid >= outer_size) return;

    __shared__ float partial_sums[32];

    uint32_t tid = threadIdx.x;
    uint32_t lane = tid % WARP_SIZE;
    uint32_t warp_id = tid / WARP_SIZE;

    const float* row = input + gid * inner_size;

    float sum = 0.0f;
    for (uint32_t i = tid; i < inner_size; i += BLOCK_SIZE) {
        sum += row[i];
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        partial_sums[warp_id] = sum;
    }

    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < (BLOCK_SIZE / 32)) ? partial_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);

        if (lane == 0) {
            output[gid] = sum / float(inner_size);
        }
    }
}

// =============================================================================
// Softmax (Numerically Stable)
// =============================================================================

extern "C" __global__
void lux_softmax_f32(
    const float* input, float* output,
    uint32_t batch_size, uint32_t dim
) {
    uint32_t gid = blockIdx.x;
    if (gid >= batch_size) return;

    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];
    __shared__ float s_max;
    __shared__ float s_sum;

    uint32_t tid = threadIdx.x;
    uint32_t lane = tid % WARP_SIZE;
    uint32_t warp_id = tid / WARP_SIZE;

    const float* x = input + gid * dim;
    float* y = output + gid * dim;

    // Pass 1: Find max
    float local_max = -INFINITY;
    for (uint32_t i = tid; i < dim; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, x[i]);
    }

    local_max = warp_reduce_max(local_max);

    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        local_max = (lane < (BLOCK_SIZE / 32)) ? shared_max[lane] : -INFINITY;
        local_max = warp_reduce_max(local_max);
        if (lane == 0) s_max = local_max;
    }
    __syncthreads();

    float max_val = s_max;

    // Pass 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint32_t i = tid; i < dim; i += BLOCK_SIZE) {
        float exp_val = expf(x[i] - max_val);
        y[i] = exp_val;
        local_sum += exp_val;
    }

    local_sum = warp_reduce_sum(local_sum);

    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (lane < (BLOCK_SIZE / 32)) ? shared_sum[lane] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        if (lane == 0) s_sum = local_sum;
    }
    __syncthreads();

    float inv_sum = 1.0f / s_sum;

    // Normalize
    for (uint32_t i = tid; i < dim; i += BLOCK_SIZE) {
        y[i] *= inv_sum;
    }
}

// Log-softmax: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
extern "C" __global__
void lux_log_softmax_f32(
    const float* input, float* output,
    uint32_t batch_size, uint32_t dim
) {
    uint32_t gid = blockIdx.x;
    if (gid >= batch_size) return;

    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];
    __shared__ float s_max;
    __shared__ float s_log_sum;

    uint32_t tid = threadIdx.x;
    uint32_t lane = tid % WARP_SIZE;
    uint32_t warp_id = tid / WARP_SIZE;

    const float* x = input + gid * dim;
    float* y = output + gid * dim;

    // Find max
    float local_max = -INFINITY;
    for (uint32_t i = tid; i < dim; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, x[i]);
    }

    local_max = warp_reduce_max(local_max);

    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        local_max = (lane < (BLOCK_SIZE / 32)) ? shared_max[lane] : -INFINITY;
        local_max = warp_reduce_max(local_max);
        if (lane == 0) s_max = local_max;
    }
    __syncthreads();

    float max_val = s_max;

    // Compute sum(exp(x - max))
    float local_sum = 0.0f;
    for (uint32_t i = tid; i < dim; i += BLOCK_SIZE) {
        local_sum += expf(x[i] - max_val);
    }

    local_sum = warp_reduce_sum(local_sum);

    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (lane < (BLOCK_SIZE / 32)) ? shared_sum[lane] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        if (lane == 0) s_log_sum = logf(local_sum);
    }
    __syncthreads();

    float log_sum = s_log_sum;

    // Compute log softmax
    for (uint32_t i = tid; i < dim; i += BLOCK_SIZE) {
        y[i] = x[i] - max_val - log_sum;
    }
}

// =============================================================================
// Layer Normalization
// =============================================================================

extern "C" __global__
void lux_layer_norm_f32(
    const float* input, float* output,
    const float* gamma, const float* beta,
    uint32_t batch_size, uint32_t dim, float eps
) {
    uint32_t gid = blockIdx.x;
    if (gid >= batch_size) return;

    __shared__ float shared_data[32];
    __shared__ float s_mean;
    __shared__ float s_var;

    uint32_t tid = threadIdx.x;
    uint32_t lane = tid % WARP_SIZE;
    uint32_t warp_id = tid / WARP_SIZE;

    const float* x = input + gid * dim;
    float* y = output + gid * dim;

    // Compute mean
    float local_sum = 0.0f;
    for (uint32_t i = tid; i < dim; i += BLOCK_SIZE) {
        local_sum += x[i];
    }

    local_sum = warp_reduce_sum(local_sum);

    if (lane == 0) shared_data[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (lane < (BLOCK_SIZE / 32)) ? shared_data[lane] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        if (lane == 0) s_mean = local_sum / float(dim);
    }
    __syncthreads();

    float mean = s_mean;

    // Compute variance
    float local_var = 0.0f;
    for (uint32_t i = tid; i < dim; i += BLOCK_SIZE) {
        float diff = x[i] - mean;
        local_var += diff * diff;
    }

    local_var = warp_reduce_sum(local_var);

    if (lane == 0) shared_data[warp_id] = local_var;
    __syncthreads();

    if (warp_id == 0) {
        local_var = (lane < (BLOCK_SIZE / 32)) ? shared_data[lane] : 0.0f;
        local_var = warp_reduce_sum(local_var);
        if (lane == 0) s_var = local_var / float(dim);
    }
    __syncthreads();

    float inv_std = rsqrtf(s_var + eps);

    // Normalize and scale
    for (uint32_t i = tid; i < dim; i += BLOCK_SIZE) {
        float normalized = (x[i] - mean) * inv_std;
        y[i] = normalized * gamma[i] + beta[i];
    }
}

// =============================================================================
// RMS Normalization
// =============================================================================

extern "C" __global__
void lux_rms_norm_f32(
    const float* input, float* output,
    const float* weight,
    uint32_t batch_size, uint32_t dim, float eps
) {
    uint32_t gid = blockIdx.x;
    if (gid >= batch_size) return;

    __shared__ float shared_data[32];
    __shared__ float s_rms;

    uint32_t tid = threadIdx.x;
    uint32_t lane = tid % WARP_SIZE;
    uint32_t warp_id = tid / WARP_SIZE;

    const float* x = input + gid * dim;
    float* y = output + gid * dim;

    // Compute sum of squares
    float local_sum_sq = 0.0f;
    for (uint32_t i = tid; i < dim; i += BLOCK_SIZE) {
        float val = x[i];
        local_sum_sq += val * val;
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);

    if (lane == 0) shared_data[warp_id] = local_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        local_sum_sq = (lane < (BLOCK_SIZE / 32)) ? shared_data[lane] : 0.0f;
        local_sum_sq = warp_reduce_sum(local_sum_sq);
        if (lane == 0) s_rms = rsqrtf(local_sum_sq / float(dim) + eps);
    }
    __syncthreads();

    float rms_scale = s_rms;

    // Scale
    for (uint32_t i = tid; i < dim; i += BLOCK_SIZE) {
        y[i] = x[i] * rms_scale * weight[i];
    }
}

#endif // __CUDA_ARCH__
