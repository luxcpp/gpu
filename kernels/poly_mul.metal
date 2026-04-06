// Copyright (c) 2024-2026 Lux Industries Inc. All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Polynomial Multiplication - High-Performance Metal Implementation
// Supports schoolbook and NTT-based multiplication for lattice cryptography.

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// 64-bit Arithmetic
// ============================================================================

struct U64 {
    uint lo;
    uint hi;
};

inline U64 u64_from(ulong v) {
    return {uint(v & 0xFFFFFFFFu), uint(v >> 32)};
}

inline ulong u64_to(U64 v) {
    return ulong(v.lo) | (ulong(v.hi) << 32);
}

inline U64 u64_zero() { return {0u, 0u}; }
inline U64 u64_one() { return {1u, 0u}; }

inline bool u64_gte(U64 a, U64 b) {
    if (a.hi > b.hi) return true;
    if (a.hi < b.hi) return false;
    return a.lo >= b.lo;
}

inline U64 u64_add(U64 a, U64 b) {
    uint lo = a.lo + b.lo;
    uint carry = (lo < a.lo) ? 1u : 0u;
    return {lo, a.hi + b.hi + carry};
}

inline U64 u64_sub(U64 a, U64 b) {
    uint borrow = (a.lo < b.lo) ? 1u : 0u;
    return {a.lo - b.lo, a.hi - b.hi - borrow};
}

inline U64 mul32_to_64(uint a, uint b) {
    uint a_lo = a & 0xFFFFu;
    uint a_hi = a >> 16u;
    uint b_lo = b & 0xFFFFu;
    uint b_hi = b >> 16u;

    uint p0 = a_lo * b_lo;
    uint p1 = a_lo * b_hi;
    uint p2 = a_hi * b_lo;
    uint p3 = a_hi * b_hi;

    uint mid = p1 + p2;
    uint mid_carry = (mid < p1) ? 0x10000u : 0u;

    uint lo = p0 + (mid << 16u);
    uint carry = (lo < p0) ? 1u : 0u;

    return {lo, p3 + (mid >> 16u) + mid_carry + carry};
}

inline void mul64_to_128(U64 a, U64 b, thread U64& lo, thread U64& hi) {
    U64 p0 = mul32_to_64(a.lo, b.lo);
    U64 p1 = mul32_to_64(a.lo, b.hi);
    U64 p2 = mul32_to_64(a.hi, b.lo);
    U64 p3 = mul32_to_64(a.hi, b.hi);

    lo.lo = p0.lo;
    uint sum1 = p0.hi + p1.lo;
    uint c1 = (sum1 < p0.hi) ? 1u : 0u;
    uint sum2 = sum1 + p2.lo;
    uint c2 = (sum2 < sum1) ? 1u : 0u;
    lo.hi = sum2;

    hi = u64_add(p3, {c1 + c2 + p1.hi + p2.hi, 0u});
}

// ============================================================================
// Modular Arithmetic
// ============================================================================

inline U64 mod_add(U64 a, U64 b, U64 q) {
    U64 sum = u64_add(a, b);
    if (u64_gte(sum, q)) sum = u64_sub(sum, q);
    return sum;
}

inline U64 mod_sub(U64 a, U64 b, U64 q) {
    if (u64_gte(a, b)) return u64_sub(a, b);
    return u64_sub(u64_add(a, q), b);
}

inline U64 mont_reduce(U64 lo, U64 hi, U64 q, U64 q_inv) {
    U64 m_lo, m_hi;
    mul64_to_128(lo, q_inv, m_lo, m_hi);

    U64 prod_lo, prod_hi;
    mul64_to_128(m_lo, q, prod_lo, prod_hi);

    U64 sum = u64_add(lo, prod_lo);
    uint carry = (sum.lo < lo.lo || sum.hi < lo.hi) ? 1u : 0u;

    U64 result = u64_add(hi, prod_hi);
    result = u64_add(result, {carry, 0u});

    if (u64_gte(result, q)) result = u64_sub(result, q);
    return result;
}

inline U64 mont_mul(U64 a, U64 b, U64 q, U64 q_inv) {
    U64 lo, hi;
    mul64_to_128(a, b, lo, hi);
    return mont_reduce(lo, hi, q, q_inv);
}

// ============================================================================
// NTT Butterfly Operations
// ============================================================================

inline void ct_butterfly(thread U64& x0, thread U64& x1, U64 w, U64 q, U64 q_inv) {
    U64 t = mont_mul(x1, w, q, q_inv);
    x1 = mod_sub(x0, t, q);
    x0 = mod_add(x0, t, q);
}

inline void gs_butterfly(thread U64& x0, thread U64& x1, U64 w, U64 q, U64 q_inv) {
    U64 t = mod_sub(x0, x1, q);
    x0 = mod_add(x0, x1, q);
    x1 = mont_mul(t, w, q, q_inv);
}

// ============================================================================
// Schoolbook Polynomial Multiplication
// ============================================================================

kernel void poly_mul_schoolbook(
    device const U64* a [[buffer(0)]],
    device const U64* b [[buffer(1)]],
    device U64* c [[buffer(2)]],
    constant U64& q [[buffer(3)]],
    constant U64& q_inv [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& poly_idx [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup U64* s_a [[threadgroup(0)]],
    threadgroup U64* s_b [[threadgroup(1)]]
) {
    uint offset = poly_idx * n;

    // Load to threadgroup memory
    if (tid < n) {
        s_a[tid] = a[offset + tid];
        s_b[tid] = b[offset + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid >= n) return;

    // Negacyclic convolution: c[k] = sum(a[i]*b[j]) where i+j=k
    //                              - sum(a[i]*b[j]) where i+j=k+n
    U64 sum = u64_zero();

    // Positive terms
    for (uint i = 0; i <= tid; i++) {
        U64 prod = mont_mul(s_a[i], s_b[tid - i], q, q_inv);
        sum = mod_add(sum, prod, q);
    }

    // Negative terms (wraparound)
    for (uint i = tid + 1; i < n; i++) {
        U64 prod = mont_mul(s_a[i], s_b[n + tid - i], q, q_inv);
        sum = mod_sub(sum, prod, q);
    }

    c[offset + tid] = sum;
}

// ============================================================================
// NTT-Based Polynomial Multiplication
// ============================================================================

kernel void ntt_forward_stage(
    device U64* data [[buffer(0)]],
    device const U64* twiddles [[buffer(1)]],
    constant U64& q [[buffer(2)]],
    constant U64& q_inv [[buffer(3)]],
    constant uint& stage [[buffer(4)]],
    constant uint& log_n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = 1u << log_n;
    uint half_n = n >> 1;

    if (gid >= half_n) return;

    uint half_size = 1u << stage;
    uint group_id = gid / half_size;
    uint idx_in_group = gid % half_size;

    uint idx0 = group_id * (half_size << 1) + idx_in_group;
    uint idx1 = idx0 + half_size;

    U64 x0 = data[idx0];
    U64 x1 = data[idx1];
    U64 w = twiddles[group_id + half_size];

    ct_butterfly(x0, x1, w, q, q_inv);

    data[idx0] = x0;
    data[idx1] = x1;
}

kernel void ntt_inverse_stage(
    device U64* data [[buffer(0)]],
    device const U64* twiddles_inv [[buffer(1)]],
    constant U64& q [[buffer(2)]],
    constant U64& q_inv [[buffer(3)]],
    constant uint& stage [[buffer(4)]],
    constant uint& log_n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = 1u << log_n;
    uint half_n = n >> 1;

    if (gid >= half_n) return;

    uint half_size = 1u << stage;
    uint group_id = gid / half_size;
    uint idx_in_group = gid % half_size;

    uint idx0 = group_id * (half_size << 1) + idx_in_group;
    uint idx1 = idx0 + half_size;

    U64 x0 = data[idx0];
    U64 x1 = data[idx1];
    U64 w = twiddles_inv[group_id + half_size];

    gs_butterfly(x0, x1, w, q, q_inv);

    data[idx0] = x0;
    data[idx1] = x1;
}

kernel void ntt_scale(
    device U64* data [[buffer(0)]],
    constant U64& q [[buffer(1)]],
    constant U64& q_inv [[buffer(2)]],
    constant U64& n_inv [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = mont_mul(data[gid], n_inv, q, q_inv);
}

// ============================================================================
// Pointwise Operations
// ============================================================================

kernel void poly_mul_pointwise(
    device const U64* a_ntt [[buffer(0)]],
    device const U64* b_ntt [[buffer(1)]],
    device U64* c_ntt [[buffer(2)]],
    constant U64& q [[buffer(3)]],
    constant U64& q_inv [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    c_ntt[gid] = mont_mul(a_ntt[gid], b_ntt[gid], q, q_inv);
}

kernel void poly_mul_acc(
    device const U64* a_ntt [[buffer(0)]],
    device const U64* b_ntt [[buffer(1)]],
    device U64* c_ntt [[buffer(2)]],
    constant U64& q [[buffer(3)]],
    constant U64& q_inv [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    U64 prod = mont_mul(a_ntt[gid], b_ntt[gid], q, q_inv);
    c_ntt[gid] = mod_add(c_ntt[gid], prod, q);
}

kernel void poly_scalar_mul(
    device const U64* a [[buffer(0)]],
    device U64* c [[buffer(1)]],
    constant U64& scalar [[buffer(2)]],
    constant U64& q [[buffer(3)]],
    constant U64& q_inv [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    c[gid] = mont_mul(a[gid], scalar, q, q_inv);
}

// ============================================================================
// Batch Operations
// ============================================================================

kernel void poly_batch_mul_pointwise(
    device const U64* a [[buffer(0)]],
    device const U64* b [[buffer(1)]],
    device U64* c [[buffer(2)]],
    constant U64& q [[buffer(3)]],
    constant U64& q_inv [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = n * batch_size;
    if (gid >= total) return;
    c[gid] = mont_mul(a[gid], b[gid], q, q_inv);
}

kernel void poly_batch_add(
    device const U64* a [[buffer(0)]],
    device const U64* b [[buffer(1)]],
    device U64* c [[buffer(2)]],
    constant U64& q [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = n * batch_size;
    if (gid >= total) return;
    c[gid] = mod_add(a[gid], b[gid], q);
}

kernel void poly_batch_sub(
    device const U64* a [[buffer(0)]],
    device const U64* b [[buffer(1)]],
    device U64* c [[buffer(2)]],
    constant U64& q [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = n * batch_size;
    if (gid >= total) return;
    c[gid] = mod_sub(a[gid], b[gid], q);
}
