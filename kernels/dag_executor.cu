// =============================================================================
// DAG Executor CUDA Kernels - Lux FHE GPU Acceleration
// =============================================================================
//
// CUDA port of dag_executor.metal. Batched FHE operations scheduled by the DAG
// scheduler. Operations at the same depth level execute in parallel.
//
// Copyright (C) 2024-2026 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>

#ifdef __CUDA_ARCH__

// =============================================================================
// Operation Types (must match Go dag.OpType)
// =============================================================================

enum DAGOpType : uint32_t {
    DAG_OP_ADD       = 0,
    DAG_OP_SUB       = 1,
    DAG_OP_MUL       = 2,
    DAG_OP_MUL_PLAIN = 3,
    DAG_OP_ADD_PLAIN = 4,
    DAG_OP_SUB_PLAIN = 5,
    DAG_OP_NEGATE    = 6,
    DAG_OP_NTT       = 7,
    DAG_OP_INTT      = 8,
    DAG_OP_COPY      = 15,
};

// =============================================================================
// Structures
// =============================================================================

struct BatchOp {
    uint32_t input1_idx;
    uint32_t input2_idx;
    uint32_t output_idx;
    uint32_t level;
};

struct DAGBatchParams {
    uint64_t Q;
    uint64_t barrett_mu;
    uint64_t N_inv;
    uint64_t N_inv_precon;
    uint32_t N;
    uint32_t log_N;
    uint32_t poly_degree;
    uint32_t batch_size;
    uint32_t op_type;
};

struct FusedAddOp {
    uint32_t input1_idx;
    uint32_t input2_idx;
    uint32_t input3_idx;
    uint32_t output_idx;
};

struct MultiOp {
    uint32_t input1_idx;
    uint32_t input2_idx;
    uint32_t output_idx;
    uint32_t op_type;
};

// =============================================================================
// Modular Arithmetic
// =============================================================================

__device__ __forceinline__
uint64_t mulhi64(uint64_t a, uint64_t b) {
    return __umul64hi(a, b);
}

__device__ __forceinline__
uint64_t barrett_reduce(uint64_t x, uint64_t Q, uint64_t mu) {
    uint64_t q_approx = mulhi64(x, mu);
    uint64_t result = x - q_approx * Q;
    return (result >= Q) ? result - Q : result;
}

__device__ __forceinline__
uint64_t barrett_mul(uint64_t a, uint64_t b, uint64_t Q, uint64_t mu) {
    uint64_t lo = a * b;
    uint64_t hi = mulhi64(a, b);

    if (hi == 0) {
        return barrett_reduce(lo, Q, mu);
    }

    uint64_t two32 = uint64_t(1) << 32;
    uint64_t two32_mod = two32 % Q;
    uint64_t two64_mod = barrett_reduce(two32_mod * two32_mod, Q, mu);

    uint64_t hi_contrib = barrett_reduce(hi * two64_mod, Q, mu);
    uint64_t lo_mod = barrett_reduce(lo, Q, mu);

    return barrett_reduce(lo_mod + hi_contrib, Q, mu);
}

__device__ __forceinline__
uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    return (sum >= Q) ? sum - Q : sum;
}

__device__ __forceinline__
uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    return (a >= b) ? a - b : a + Q - b;
}

__device__ __forceinline__
uint64_t mod_neg(uint64_t a, uint64_t Q) {
    return (a == 0) ? 0 : Q - a;
}

// =============================================================================
// NTT Butterfly Operations
// =============================================================================

__device__ __forceinline__
void ct_butterfly(uint64_t* lo, uint64_t* hi,
                  uint64_t omega, uint64_t Q, uint64_t mu) {
    uint64_t u = *lo;
    uint64_t v = barrett_mul(*hi, omega, Q, mu);
    *lo = mod_add(u, v, Q);
    *hi = mod_sub(u, v, Q);
}

__device__ __forceinline__
void gs_butterfly(uint64_t* lo, uint64_t* hi,
                  uint64_t omega, uint64_t Q, uint64_t mu) {
    uint64_t u = *lo;
    uint64_t v = *hi;
    *lo = mod_add(u, v, Q);
    *hi = barrett_mul(mod_sub(u, v, Q), omega, Q, mu);
}

// =============================================================================
// Batched Add Kernel
// =============================================================================

extern "C" __global__
void dag_batch_add(
    uint64_t* ciphertexts,
    const BatchOp* ops,
    const DAGBatchParams params
) {
    uint32_t N = params.N;
    uint32_t poly_degree = params.poly_degree;
    uint64_t Q = params.Q;

    uint32_t op_idx = blockIdx.x;
    if (op_idx >= params.batch_size) return;

    uint32_t poly_idx = blockIdx.y;
    if (poly_idx > poly_degree) return;

    BatchOp op = ops[op_idx];

    uint32_t ct_stride = (poly_degree + 1) * N;
    const uint64_t* in1 = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
    const uint64_t* in2 = ciphertexts + op.input2_idx * ct_stride + poly_idx * N;
    uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;

    for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) {
        out[i] = mod_add(in1[i], in2[i], Q);
    }
}

// =============================================================================
// Batched Sub Kernel
// =============================================================================

extern "C" __global__
void dag_batch_sub(
    uint64_t* ciphertexts,
    const BatchOp* ops,
    const DAGBatchParams params
) {
    uint32_t N = params.N;
    uint32_t poly_degree = params.poly_degree;
    uint64_t Q = params.Q;

    uint32_t op_idx = blockIdx.x;
    if (op_idx >= params.batch_size) return;

    uint32_t poly_idx = blockIdx.y;
    if (poly_idx > poly_degree) return;

    BatchOp op = ops[op_idx];

    uint32_t ct_stride = (poly_degree + 1) * N;
    const uint64_t* in1 = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
    const uint64_t* in2 = ciphertexts + op.input2_idx * ct_stride + poly_idx * N;
    uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;

    for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) {
        out[i] = mod_sub(in1[i], in2[i], Q);
    }
}

// =============================================================================
// Batched Negate Kernel
// =============================================================================

extern "C" __global__
void dag_batch_negate(
    uint64_t* ciphertexts,
    const BatchOp* ops,
    const DAGBatchParams params
) {
    uint32_t N = params.N;
    uint32_t poly_degree = params.poly_degree;
    uint64_t Q = params.Q;

    uint32_t op_idx = blockIdx.x;
    if (op_idx >= params.batch_size) return;

    uint32_t poly_idx = blockIdx.y;
    if (poly_idx > poly_degree) return;

    BatchOp op = ops[op_idx];

    uint32_t ct_stride = (poly_degree + 1) * N;
    const uint64_t* in = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
    uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;

    for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) {
        out[i] = mod_neg(in[i], Q);
    }
}

// =============================================================================
// Batched NTT Kernel (Forward, Cooley-Tukey)
// =============================================================================

extern "C" __global__
void dag_batch_ntt(
    uint64_t* ciphertexts,
    const BatchOp* ops,
    const uint64_t* twiddles,
    const uint64_t* precons,
    const DAGBatchParams params
) {
    extern __shared__ uint64_t shared[];

    uint32_t N = params.N;
    uint32_t log_N = params.log_N;
    uint32_t poly_degree = params.poly_degree;
    uint64_t Q = params.Q;
    uint64_t mu = params.barrett_mu;

    uint32_t op_idx = blockIdx.x;
    if (op_idx >= params.batch_size) return;

    uint32_t poly_idx = blockIdx.y;
    if (poly_idx > poly_degree) return;

    BatchOp op = ops[op_idx];
    uint32_t local_id = threadIdx.x;
    uint32_t threads = blockDim.x;

    uint32_t ct_stride = (poly_degree + 1) * N;
    const uint64_t* in = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
    uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;

    // Load to shared memory
    for (uint32_t i = local_id; i < N; i += threads) {
        shared[i] = in[i];
    }
    __syncthreads();

    // NTT stages (Cooley-Tukey)
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = 1u << stage;
        uint32_t t = N >> (stage + 1);

        for (uint32_t bf = local_id; bf < N / 2; bf += threads) {
            uint32_t group = bf / t;
            uint32_t j = bf % t;
            uint32_t idx_lo = group * (2 * t) + j;
            uint32_t idx_hi = idx_lo + t;

            uint32_t tw_idx = m + group;
            uint64_t omega = twiddles[tw_idx];
            uint64_t precon = precons[tw_idx];

            uint64_t lo = shared[idx_lo];
            uint64_t hi = shared[idx_hi];

            // Barrett multiplication with precomputed constant
            uint64_t q_approx = mulhi64(hi, precon);
            uint64_t hi_tw = hi * omega - q_approx * Q;
            if (hi_tw >= Q) hi_tw -= Q;

            shared[idx_lo] = mod_add(lo, hi_tw, Q);
            shared[idx_hi] = mod_sub(lo, hi_tw, Q);
        }
        __syncthreads();
    }

    // Write back
    for (uint32_t i = local_id; i < N; i += threads) {
        out[i] = shared[i];
    }
}

// =============================================================================
// Batched INTT Kernel (Inverse, Gentleman-Sande)
// =============================================================================

extern "C" __global__
void dag_batch_intt(
    uint64_t* ciphertexts,
    const BatchOp* ops,
    const uint64_t* inv_twiddles,
    const uint64_t* inv_precons,
    const DAGBatchParams params
) {
    extern __shared__ uint64_t shared[];

    uint32_t N = params.N;
    uint32_t log_N = params.log_N;
    uint32_t poly_degree = params.poly_degree;
    uint64_t Q = params.Q;
    uint64_t N_inv = params.N_inv;
    uint64_t N_inv_precon = params.N_inv_precon;

    uint32_t op_idx = blockIdx.x;
    if (op_idx >= params.batch_size) return;

    uint32_t poly_idx = blockIdx.y;
    if (poly_idx > poly_degree) return;

    BatchOp op = ops[op_idx];
    uint32_t local_id = threadIdx.x;
    uint32_t threads = blockDim.x;

    uint32_t ct_stride = (poly_degree + 1) * N;
    const uint64_t* in = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
    uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;

    // Load to shared memory
    for (uint32_t i = local_id; i < N; i += threads) {
        shared[i] = in[i];
    }
    __syncthreads();

    // INTT stages (Gentleman-Sande)
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = N >> (stage + 1);
        uint32_t t = 1u << stage;

        for (uint32_t bf = local_id; bf < N / 2; bf += threads) {
            uint32_t group = bf / t;
            uint32_t j = bf % t;
            uint32_t idx_lo = group * (2 * t) + j;
            uint32_t idx_hi = idx_lo + t;

            uint32_t tw_idx = m + group;
            uint64_t omega = inv_twiddles[tw_idx];
            uint64_t precon = inv_precons[tw_idx];

            uint64_t lo = shared[idx_lo];
            uint64_t hi = shared[idx_hi];

            uint64_t sum = mod_add(lo, hi, Q);
            uint64_t diff = mod_sub(lo, hi, Q);

            // Barrett multiplication
            uint64_t q_approx = mulhi64(diff, precon);
            uint64_t diff_tw = diff * omega - q_approx * Q;
            if (diff_tw >= Q) diff_tw -= Q;

            shared[idx_lo] = sum;
            shared[idx_hi] = diff_tw;
        }
        __syncthreads();
    }

    // Scale by N^{-1} and write back
    for (uint32_t i = local_id; i < N; i += threads) {
        uint64_t val = shared[i];
        uint64_t q_approx = mulhi64(val, N_inv_precon);
        uint64_t scaled = val * N_inv - q_approx * Q;
        out[i] = (scaled >= Q) ? scaled - Q : scaled;
    }
}

// =============================================================================
// Batched Copy Kernel
// =============================================================================

extern "C" __global__
void dag_batch_copy(
    uint64_t* ciphertexts,
    const BatchOp* ops,
    const DAGBatchParams params
) {
    uint32_t N = params.N;
    uint32_t poly_degree = params.poly_degree;

    uint32_t op_idx = blockIdx.x;
    if (op_idx >= params.batch_size) return;

    uint32_t poly_idx = blockIdx.y;
    if (poly_idx > poly_degree) return;

    BatchOp op = ops[op_idx];

    uint32_t ct_stride = (poly_degree + 1) * N;
    const uint64_t* in = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
    uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;

    for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) {
        out[i] = in[i];
    }
}

// =============================================================================
// Fused Add-Add Kernel: out = (in1 + in2) + in3
// =============================================================================

extern "C" __global__
void dag_fused_add_add(
    uint64_t* ciphertexts,
    const FusedAddOp* ops,
    const DAGBatchParams params
) {
    uint32_t N = params.N;
    uint32_t poly_degree = params.poly_degree;
    uint64_t Q = params.Q;

    uint32_t op_idx = blockIdx.x;
    if (op_idx >= params.batch_size) return;

    uint32_t poly_idx = blockIdx.y;
    if (poly_idx > poly_degree) return;

    FusedAddOp op = ops[op_idx];

    uint32_t ct_stride = (poly_degree + 1) * N;
    const uint64_t* in1 = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
    const uint64_t* in2 = ciphertexts + op.input2_idx * ct_stride + poly_idx * N;
    const uint64_t* in3 = ciphertexts + op.input3_idx * ct_stride + poly_idx * N;
    uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;

    for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) {
        uint64_t tmp = mod_add(in1[i], in2[i], Q);
        out[i] = mod_add(tmp, in3[i], Q);
    }
}

// =============================================================================
// Unified Dispatch Kernel (for small/heterogeneous batches)
// =============================================================================

extern "C" __global__
void dag_dispatch(
    uint64_t* ciphertexts,
    const BatchOp* ops,
    const uint64_t* twiddles,
    const uint64_t* precons,
    const uint64_t* inv_twiddles,
    const uint64_t* inv_precons,
    const DAGBatchParams params
) {
    extern __shared__ uint64_t shared[];

    uint32_t N = params.N;
    uint32_t poly_degree = params.poly_degree;
    uint64_t Q = params.Q;

    uint32_t op_idx = blockIdx.x;
    if (op_idx >= params.batch_size) return;

    uint32_t poly_idx = blockIdx.y;
    if (poly_idx > poly_degree) return;

    BatchOp op = ops[op_idx];
    uint32_t ct_stride = (poly_degree + 1) * N;

    switch (params.op_type) {
        case DAG_OP_ADD: {
            const uint64_t* in1 = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
            const uint64_t* in2 = ciphertexts + op.input2_idx * ct_stride + poly_idx * N;
            uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;
            for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) {
                out[i] = mod_add(in1[i], in2[i], Q);
            }
            break;
        }

        case DAG_OP_SUB: {
            const uint64_t* in1 = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
            const uint64_t* in2 = ciphertexts + op.input2_idx * ct_stride + poly_idx * N;
            uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;
            for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) {
                out[i] = mod_sub(in1[i], in2[i], Q);
            }
            break;
        }

        case DAG_OP_NEGATE: {
            const uint64_t* in = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
            uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;
            for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) {
                out[i] = mod_neg(in[i], Q);
            }
            break;
        }

        case DAG_OP_COPY: {
            const uint64_t* in = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
            uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;
            for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) {
                out[i] = in[i];
            }
            break;
        }

        default:
            break;
    }
}

// =============================================================================
// Multi-Operation Batch Kernel (different ops in one dispatch)
// =============================================================================

extern "C" __global__
void dag_multi_dispatch(
    uint64_t* ciphertexts,
    const MultiOp* ops,
    const DAGBatchParams params
) {
    uint32_t N = params.N;
    uint32_t poly_degree = params.poly_degree;
    uint64_t Q = params.Q;

    uint32_t op_idx = blockIdx.x;
    if (op_idx >= params.batch_size) return;

    uint32_t poly_idx = blockIdx.y;
    if (poly_idx > poly_degree) return;

    MultiOp op = ops[op_idx];
    uint32_t ct_stride = (poly_degree + 1) * N;

    const uint64_t* in1 = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
    const uint64_t* in2 = ciphertexts + op.input2_idx * ct_stride + poly_idx * N;
    uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;

    switch (op.op_type) {
        case DAG_OP_ADD:
            for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) {
                out[i] = mod_add(in1[i], in2[i], Q);
            }
            break;

        case DAG_OP_SUB:
            for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) {
                out[i] = mod_sub(in1[i], in2[i], Q);
            }
            break;

        case DAG_OP_NEGATE:
            for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) {
                out[i] = mod_neg(in1[i], Q);
            }
            break;

        case DAG_OP_COPY:
            for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) {
                out[i] = in1[i];
            }
            break;

        default:
            break;
    }
}

#endif // __CUDA_ARCH__
