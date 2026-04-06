// =============================================================================
// DAG Executor Metal Kernels - Lux FHE GPU Acceleration
// =============================================================================
//
// Metal kernels for executing batched FHE operations scheduled by the DAG
// scheduler. Operations at the same depth level can be executed in parallel
// with a single kernel dispatch.
//
// Key Features:
// - Batched execution of homogeneous operations (Add, Sub, Negate, NTT, INTT)
// - Configurable batch size per dispatch
// - Optimized for Apple Silicon (M1/M2/M3/M4)
// - Minimal synchronization via depth-level batching
//
// Copyright (C) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Operation Types (must match Go dag.OpType)
// =============================================================================

enum DAGOpType : uint32_t {
    DAG_OP_ADD = 0,
    DAG_OP_SUB = 1,
    DAG_OP_MUL = 2,
    DAG_OP_MUL_PLAIN = 3,
    DAG_OP_ADD_PLAIN = 4,
    DAG_OP_SUB_PLAIN = 5,
    DAG_OP_NEGATE = 6,
    DAG_OP_NTT = 7,
    DAG_OP_INTT = 8,
    DAG_OP_COPY = 15,
};

// =============================================================================
// Structures
// =============================================================================

// Batch operation descriptor
struct BatchOp {
    uint32_t input1_idx;  // Index of first input ciphertext
    uint32_t input2_idx;  // Index of second input (for binary ops)
    uint32_t output_idx;  // Index of output ciphertext
    uint32_t level;       // Ciphertext level
};

// Batch dispatch parameters
struct DAGBatchParams {
    uint64_t Q;              // Prime modulus
    uint64_t barrett_mu;     // Barrett constant
    uint64_t N_inv;          // N^{-1} mod Q for INTT
    uint64_t N_inv_precon;   // Barrett precomputation for N_inv
    uint32_t N;              // Ring dimension
    uint32_t log_N;          // log2(N)
    uint32_t poly_degree;    // Polynomial degree (k for RLWE degree k)
    uint32_t batch_size;     // Number of operations in batch
    uint32_t op_type;        // Operation type
};

// =============================================================================
// Modular Arithmetic
// =============================================================================

// High 64 bits of 64x64 multiplication
inline uint64_t mulhi64(uint64_t a, uint64_t b) {
    return metal::mulhi(a, b);
}

// Barrett reduction: x mod Q
inline uint64_t barrett_reduce(uint64_t x, uint64_t Q, uint64_t mu) {
    uint64_t q_approx = mulhi64(x, mu);
    uint64_t result = x - q_approx * Q;
    return (result >= Q) ? result - Q : result;
}

// Barrett multiplication: (a * b) mod Q
inline uint64_t barrett_mul(uint64_t a, uint64_t b, uint64_t Q, uint64_t mu) {
    uint64_t lo = a * b;
    uint64_t hi = mulhi64(a, b);

    if (hi == 0) {
        return barrett_reduce(lo, Q, mu);
    }

    // Full 128-bit reduction
    uint64_t two32 = uint64_t(1) << 32;
    uint64_t two32_mod = two32 % Q;
    uint64_t two64_mod = barrett_reduce(two32_mod * two32_mod, Q, mu);

    uint64_t hi_contrib = barrett_reduce(hi * two64_mod, Q, mu);
    uint64_t lo_mod = barrett_reduce(lo, Q, mu);

    return barrett_reduce(lo_mod + hi_contrib, Q, mu);
}

// Modular addition: (a + b) mod Q
inline uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    return (sum >= Q) ? sum - Q : sum;
}

// Modular subtraction: (a - b) mod Q
inline uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    return (a >= b) ? a - b : a + Q - b;
}

// Modular negation: -a mod Q
inline uint64_t mod_neg(uint64_t a, uint64_t Q) {
    return (a == 0) ? 0 : Q - a;
}

// =============================================================================
// NTT Butterfly Operations
// =============================================================================

// Cooley-Tukey butterfly for forward NTT
inline void ct_butterfly(
    threadgroup uint64_t* lo,
    threadgroup uint64_t* hi,
    uint64_t omega,
    uint64_t Q,
    uint64_t mu
) {
    uint64_t u = *lo;
    uint64_t v = barrett_mul(*hi, omega, Q, mu);
    *lo = mod_add(u, v, Q);
    *hi = mod_sub(u, v, Q);
}

// Gentleman-Sande butterfly for inverse NTT
inline void gs_butterfly(
    threadgroup uint64_t* lo,
    threadgroup uint64_t* hi,
    uint64_t omega,
    uint64_t Q,
    uint64_t mu
) {
    uint64_t u = *lo;
    uint64_t v = *hi;
    *lo = mod_add(u, v, Q);
    *hi = barrett_mul(mod_sub(u, v, Q), omega, Q, mu);
}

// =============================================================================
// Batched Add Kernel
// =============================================================================

kernel void dag_batch_add(
    device uint64_t* ciphertexts         [[buffer(0)]],
    device const BatchOp* ops            [[buffer(1)]],
    constant DAGBatchParams& params      [[buffer(2)]],

    uint3 tid                            [[thread_position_in_threadgroup]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint3 tg_size                        [[threads_per_threadgroup]]
) {
    uint32_t N = params.N;
    uint32_t poly_degree = params.poly_degree;
    uint64_t Q = params.Q;

    uint32_t op_idx = tgid.x;
    if (op_idx >= params.batch_size) return;

    uint32_t poly_idx = tgid.y;
    if (poly_idx > poly_degree) return;

    BatchOp op = ops[op_idx];

    uint32_t ct_stride = (poly_degree + 1) * N;
    device const uint64_t* in1 = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
    device const uint64_t* in2 = ciphertexts + op.input2_idx * ct_stride + poly_idx * N;
    device uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;

    for (uint32_t i = tid.x; i < N; i += tg_size.x) {
        out[i] = mod_add(in1[i], in2[i], Q);
    }
}

// =============================================================================
// Batched Sub Kernel
// =============================================================================

kernel void dag_batch_sub(
    device uint64_t* ciphertexts         [[buffer(0)]],
    device const BatchOp* ops            [[buffer(1)]],
    constant DAGBatchParams& params      [[buffer(2)]],

    uint3 tid                            [[thread_position_in_threadgroup]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint3 tg_size                        [[threads_per_threadgroup]]
) {
    uint32_t N = params.N;
    uint32_t poly_degree = params.poly_degree;
    uint64_t Q = params.Q;

    uint32_t op_idx = tgid.x;
    if (op_idx >= params.batch_size) return;

    uint32_t poly_idx = tgid.y;
    if (poly_idx > poly_degree) return;

    BatchOp op = ops[op_idx];

    uint32_t ct_stride = (poly_degree + 1) * N;
    device const uint64_t* in1 = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
    device const uint64_t* in2 = ciphertexts + op.input2_idx * ct_stride + poly_idx * N;
    device uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;

    for (uint32_t i = tid.x; i < N; i += tg_size.x) {
        out[i] = mod_sub(in1[i], in2[i], Q);
    }
}

// =============================================================================
// Batched Negate Kernel
// =============================================================================

kernel void dag_batch_negate(
    device uint64_t* ciphertexts         [[buffer(0)]],
    device const BatchOp* ops            [[buffer(1)]],
    constant DAGBatchParams& params      [[buffer(2)]],

    uint3 tid                            [[thread_position_in_threadgroup]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint3 tg_size                        [[threads_per_threadgroup]]
) {
    uint32_t N = params.N;
    uint32_t poly_degree = params.poly_degree;
    uint64_t Q = params.Q;

    uint32_t op_idx = tgid.x;
    if (op_idx >= params.batch_size) return;

    uint32_t poly_idx = tgid.y;
    if (poly_idx > poly_degree) return;

    BatchOp op = ops[op_idx];

    uint32_t ct_stride = (poly_degree + 1) * N;
    device const uint64_t* in = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
    device uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;

    for (uint32_t i = tid.x; i < N; i += tg_size.x) {
        out[i] = mod_neg(in[i], Q);
    }
}

// =============================================================================
// Batched NTT Kernel (Forward)
// =============================================================================

kernel void dag_batch_ntt(
    device uint64_t* ciphertexts         [[buffer(0)]],
    device const BatchOp* ops            [[buffer(1)]],
    device const uint64_t* twiddles      [[buffer(2)]],
    device const uint64_t* precons       [[buffer(3)]],
    constant DAGBatchParams& params      [[buffer(4)]],

    uint3 tid                            [[thread_position_in_threadgroup]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint3 tg_size                        [[threads_per_threadgroup]],

    threadgroup uint64_t* shared         [[threadgroup(0)]]
) {
    uint32_t N = params.N;
    uint32_t log_N = params.log_N;
    uint32_t poly_degree = params.poly_degree;
    uint64_t Q = params.Q;
    uint64_t mu = params.barrett_mu;

    uint32_t op_idx = tgid.x;
    if (op_idx >= params.batch_size) return;

    uint32_t poly_idx = tgid.y;
    if (poly_idx > poly_degree) return;

    BatchOp op = ops[op_idx];
    uint32_t local_id = tid.x;
    uint32_t threads = tg_size.x;

    uint32_t ct_stride = (poly_degree + 1) * N;
    device const uint64_t* in = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
    device uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;

    // Load to shared memory
    for (uint32_t i = local_id; i < N; i += threads) {
        shared[i] = in[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back
    for (uint32_t i = local_id; i < N; i += threads) {
        out[i] = shared[i];
    }
}

// =============================================================================
// Batched INTT Kernel (Inverse)
// =============================================================================

kernel void dag_batch_intt(
    device uint64_t* ciphertexts         [[buffer(0)]],
    device const BatchOp* ops            [[buffer(1)]],
    device const uint64_t* inv_twiddles  [[buffer(2)]],
    device const uint64_t* inv_precons   [[buffer(3)]],
    constant DAGBatchParams& params      [[buffer(4)]],

    uint3 tid                            [[thread_position_in_threadgroup]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint3 tg_size                        [[threads_per_threadgroup]],

    threadgroup uint64_t* shared         [[threadgroup(0)]]
) {
    uint32_t N = params.N;
    uint32_t log_N = params.log_N;
    uint32_t poly_degree = params.poly_degree;
    uint64_t Q = params.Q;
    uint64_t mu = params.barrett_mu;
    uint64_t N_inv = params.N_inv;
    uint64_t N_inv_precon = params.N_inv_precon;

    uint32_t op_idx = tgid.x;
    if (op_idx >= params.batch_size) return;

    uint32_t poly_idx = tgid.y;
    if (poly_idx > poly_degree) return;

    BatchOp op = ops[op_idx];
    uint32_t local_id = tid.x;
    uint32_t threads = tg_size.x;

    uint32_t ct_stride = (poly_degree + 1) * N;
    device const uint64_t* in = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
    device uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;

    // Load to shared memory
    for (uint32_t i = local_id; i < N; i += threads) {
        shared[i] = in[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
        threadgroup_barrier(mem_flags::mem_threadgroup);
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

kernel void dag_batch_copy(
    device uint64_t* ciphertexts         [[buffer(0)]],
    device const BatchOp* ops            [[buffer(1)]],
    constant DAGBatchParams& params      [[buffer(2)]],

    uint3 tid                            [[thread_position_in_threadgroup]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint3 tg_size                        [[threads_per_threadgroup]]
) {
    uint32_t N = params.N;
    uint32_t poly_degree = params.poly_degree;

    uint32_t op_idx = tgid.x;
    if (op_idx >= params.batch_size) return;

    uint32_t poly_idx = tgid.y;
    if (poly_idx > poly_degree) return;

    BatchOp op = ops[op_idx];

    uint32_t ct_stride = (poly_degree + 1) * N;
    device const uint64_t* in = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
    device uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;

    for (uint32_t i = tid.x; i < N; i += tg_size.x) {
        out[i] = in[i];
    }
}

// =============================================================================
// Fused Add-Add Kernel
// =============================================================================
// Computes: out = (in1 + in2) + in3 in a single pass

struct FusedAddOp {
    uint32_t input1_idx;
    uint32_t input2_idx;
    uint32_t input3_idx;
    uint32_t output_idx;
};

kernel void dag_fused_add_add(
    device uint64_t* ciphertexts         [[buffer(0)]],
    device const FusedAddOp* ops         [[buffer(1)]],
    constant DAGBatchParams& params      [[buffer(2)]],

    uint3 tid                            [[thread_position_in_threadgroup]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint3 tg_size                        [[threads_per_threadgroup]]
) {
    uint32_t N = params.N;
    uint32_t poly_degree = params.poly_degree;
    uint64_t Q = params.Q;

    uint32_t op_idx = tgid.x;
    if (op_idx >= params.batch_size) return;

    uint32_t poly_idx = tgid.y;
    if (poly_idx > poly_degree) return;

    FusedAddOp op = ops[op_idx];

    uint32_t ct_stride = (poly_degree + 1) * N;
    device const uint64_t* in1 = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
    device const uint64_t* in2 = ciphertexts + op.input2_idx * ct_stride + poly_idx * N;
    device const uint64_t* in3 = ciphertexts + op.input3_idx * ct_stride + poly_idx * N;
    device uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;

    for (uint32_t i = tid.x; i < N; i += tg_size.x) {
        uint64_t tmp = mod_add(in1[i], in2[i], Q);
        out[i] = mod_add(tmp, in3[i], Q);
    }
}

// =============================================================================
// Unified Dispatch Kernel
// =============================================================================
// For small batches or heterogeneous operations

kernel void dag_dispatch(
    device uint64_t* ciphertexts         [[buffer(0)]],
    device const BatchOp* ops            [[buffer(1)]],
    device const uint64_t* twiddles      [[buffer(2)]],
    device const uint64_t* precons       [[buffer(3)]],
    device const uint64_t* inv_twiddles  [[buffer(4)]],
    device const uint64_t* inv_precons   [[buffer(5)]],
    constant DAGBatchParams& params      [[buffer(6)]],

    uint3 tid                            [[thread_position_in_threadgroup]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint3 tg_size                        [[threads_per_threadgroup]],

    threadgroup uint64_t* shared         [[threadgroup(0)]]
) {
    uint32_t N = params.N;
    uint32_t poly_degree = params.poly_degree;
    uint64_t Q = params.Q;

    uint32_t op_idx = tgid.x;
    if (op_idx >= params.batch_size) return;

    uint32_t poly_idx = tgid.y;
    if (poly_idx > poly_degree) return;

    BatchOp op = ops[op_idx];
    uint32_t ct_stride = (poly_degree + 1) * N;

    switch (params.op_type) {
        case DAG_OP_ADD: {
            device const uint64_t* in1 = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
            device const uint64_t* in2 = ciphertexts + op.input2_idx * ct_stride + poly_idx * N;
            device uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;
            for (uint32_t i = tid.x; i < N; i += tg_size.x) {
                out[i] = mod_add(in1[i], in2[i], Q);
            }
            break;
        }

        case DAG_OP_SUB: {
            device const uint64_t* in1 = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
            device const uint64_t* in2 = ciphertexts + op.input2_idx * ct_stride + poly_idx * N;
            device uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;
            for (uint32_t i = tid.x; i < N; i += tg_size.x) {
                out[i] = mod_sub(in1[i], in2[i], Q);
            }
            break;
        }

        case DAG_OP_NEGATE: {
            device const uint64_t* in = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
            device uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;
            for (uint32_t i = tid.x; i < N; i += tg_size.x) {
                out[i] = mod_neg(in[i], Q);
            }
            break;
        }

        case DAG_OP_COPY: {
            device const uint64_t* in = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
            device uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;
            for (uint32_t i = tid.x; i < N; i += tg_size.x) {
                out[i] = in[i];
            }
            break;
        }

        default:
            break;
    }
}

// =============================================================================
// Multi-Operation Batch Kernel
// =============================================================================
// Process multiple different operations in one kernel (for small mixed batches)

struct MultiOp {
    uint32_t input1_idx;
    uint32_t input2_idx;
    uint32_t output_idx;
    uint32_t op_type;
};

kernel void dag_multi_dispatch(
    device uint64_t* ciphertexts         [[buffer(0)]],
    device const MultiOp* ops            [[buffer(1)]],
    constant DAGBatchParams& params      [[buffer(2)]],

    uint3 tid                            [[thread_position_in_threadgroup]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint3 tg_size                        [[threads_per_threadgroup]]
) {
    uint32_t N = params.N;
    uint32_t poly_degree = params.poly_degree;
    uint64_t Q = params.Q;

    uint32_t op_idx = tgid.x;
    if (op_idx >= params.batch_size) return;

    uint32_t poly_idx = tgid.y;
    if (poly_idx > poly_degree) return;

    MultiOp op = ops[op_idx];
    uint32_t ct_stride = (poly_degree + 1) * N;

    device const uint64_t* in1 = ciphertexts + op.input1_idx * ct_stride + poly_idx * N;
    device const uint64_t* in2 = ciphertexts + op.input2_idx * ct_stride + poly_idx * N;
    device uint64_t* out = ciphertexts + op.output_idx * ct_stride + poly_idx * N;

    switch (op.op_type) {
        case DAG_OP_ADD:
            for (uint32_t i = tid.x; i < N; i += tg_size.x) {
                out[i] = mod_add(in1[i], in2[i], Q);
            }
            break;

        case DAG_OP_SUB:
            for (uint32_t i = tid.x; i < N; i += tg_size.x) {
                out[i] = mod_sub(in1[i], in2[i], Q);
            }
            break;

        case DAG_OP_NEGATE:
            for (uint32_t i = tid.x; i < N; i += tg_size.x) {
                out[i] = mod_neg(in1[i], Q);
            }
            break;

        case DAG_OP_COPY:
            for (uint32_t i = tid.x; i < N; i += tg_size.x) {
                out[i] = in1[i];
            }
            break;

        default:
            break;
    }
}
