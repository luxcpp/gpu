// Copyright (c) 2024-2026 Lux Industries Inc. All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Polynomial Multiplication - CUDA Port of poly_mul.metal
// Byte-identical output. Supports schoolbook and NTT-based multiplication.

#include <cstdint>

#ifdef __CUDA_ARCH__
#define PM_DEVICE __device__ __forceinline__
#else
#define PM_DEVICE inline
#define __global__
#define __shared__
static inline void __syncthreads() {}
#endif

// ============================================================================
// 64-bit Arithmetic (matches Metal's emulated 128-bit via U64 struct)
// On CUDA we have native __int128, but we keep the same struct interface
// to guarantee byte-identical intermediate values.
// ============================================================================

struct U64 { uint32_t lo; uint32_t hi; };

PM_DEVICE U64 u64_from(uint64_t v) { return {(uint32_t)(v & 0xFFFFFFFFu), (uint32_t)(v >> 32)}; }
PM_DEVICE uint64_t u64_to(U64 v) { return (uint64_t)v.lo | ((uint64_t)v.hi << 32); }
PM_DEVICE U64 u64_zero() { return {0u, 0u}; }

PM_DEVICE bool u64_gte(U64 a, U64 b) {
    if (a.hi > b.hi) return true;
    if (a.hi < b.hi) return false;
    return a.lo >= b.lo;
}

PM_DEVICE U64 u64_add(U64 a, U64 b) {
    uint32_t lo = a.lo + b.lo;
    uint32_t carry = (lo < a.lo) ? 1u : 0u;
    return {lo, a.hi + b.hi + carry};
}

PM_DEVICE U64 u64_sub(U64 a, U64 b) {
    uint32_t borrow = (a.lo < b.lo) ? 1u : 0u;
    return {a.lo - b.lo, a.hi - b.hi - borrow};
}

PM_DEVICE U64 mul32_to_64(uint32_t a, uint32_t b) {
    uint32_t a_lo = a & 0xFFFFu, a_hi = a >> 16u;
    uint32_t b_lo = b & 0xFFFFu, b_hi = b >> 16u;
    uint32_t p0 = a_lo * b_lo, p1 = a_lo * b_hi;
    uint32_t p2 = a_hi * b_lo, p3 = a_hi * b_hi;
    uint32_t mid = p1 + p2;
    uint32_t mid_carry = (mid < p1) ? 0x10000u : 0u;
    uint32_t lo = p0 + (mid << 16u);
    uint32_t carry = (lo < p0) ? 1u : 0u;
    return {lo, p3 + (mid >> 16u) + mid_carry + carry};
}

PM_DEVICE void mul64_to_128(U64 a, U64 b, U64& lo, U64& hi) {
    U64 p0 = mul32_to_64(a.lo, b.lo);
    U64 p1 = mul32_to_64(a.lo, b.hi);
    U64 p2 = mul32_to_64(a.hi, b.lo);
    U64 p3 = mul32_to_64(a.hi, b.hi);
    lo.lo = p0.lo;
    uint32_t sum1 = p0.hi + p1.lo;
    uint32_t c1 = (sum1 < p0.hi) ? 1u : 0u;
    uint32_t sum2 = sum1 + p2.lo;
    uint32_t c2 = (sum2 < sum1) ? 1u : 0u;
    lo.hi = sum2;
    hi = u64_add(p3, {c1 + c2 + p1.hi + p2.hi, 0u});
}

// ============================================================================
// Modular Arithmetic
// ============================================================================

PM_DEVICE U64 mod_add(U64 a, U64 b, U64 q) {
    U64 sum = u64_add(a, b);
    if (u64_gte(sum, q)) sum = u64_sub(sum, q);
    return sum;
}

PM_DEVICE U64 mod_sub(U64 a, U64 b, U64 q) {
    if (u64_gte(a, b)) return u64_sub(a, b);
    return u64_sub(u64_add(a, q), b);
}

PM_DEVICE U64 mont_reduce(U64 lo, U64 hi, U64 q, U64 q_inv) {
    U64 m_lo, m_hi;
    mul64_to_128(lo, q_inv, m_lo, m_hi);
    U64 prod_lo, prod_hi;
    mul64_to_128(m_lo, q, prod_lo, prod_hi);
    U64 sum = u64_add(lo, prod_lo);
    uint32_t carry = (sum.lo < lo.lo || sum.hi < lo.hi) ? 1u : 0u;
    U64 result = u64_add(hi, prod_hi);
    result = u64_add(result, {carry, 0u});
    if (u64_gte(result, q)) result = u64_sub(result, q);
    return result;
}

PM_DEVICE U64 mont_mul(U64 a, U64 b, U64 q, U64 q_inv) {
    U64 lo, hi;
    mul64_to_128(a, b, lo, hi);
    return mont_reduce(lo, hi, q, q_inv);
}

// ============================================================================
// NTT Butterfly Operations
// ============================================================================

PM_DEVICE void ct_butterfly(U64& x0, U64& x1, U64 w, U64 q, U64 q_inv) {
    U64 t = mont_mul(x1, w, q, q_inv);
    x1 = mod_sub(x0, t, q);
    x0 = mod_add(x0, t, q);
}

PM_DEVICE void gs_butterfly(U64& x0, U64& x1, U64 w, U64 q, U64 q_inv) {
    U64 t = mod_sub(x0, x1, q);
    x0 = mod_add(x0, x1, q);
    x1 = mont_mul(t, w, q, q_inv);
}

// ============================================================================
// Schoolbook Polynomial Multiplication
// ============================================================================

extern "C" __global__ void poly_mul_schoolbook(
    const U64* a, const U64* b, U64* c,
    U64 q, U64 q_inv, uint32_t n, uint32_t poly_idx)
{
#ifdef __CUDA_ARCH__
    extern __shared__ U64 smem[];
    U64* s_a = smem;
    U64* s_b = smem + n;
    uint32_t tid = threadIdx.x;
    uint32_t offset = poly_idx * n;

    if (tid < n) {
        s_a[tid] = a[offset + tid];
        s_b[tid] = b[offset + tid];
    }
    __syncthreads();
    if (tid >= n) return;

    U64 sum = u64_zero();
    for (uint32_t i = 0; i <= tid; i++) {
        U64 prod = mont_mul(s_a[i], s_b[tid - i], q, q_inv);
        sum = mod_add(sum, prod, q);
    }
    for (uint32_t i = tid + 1; i < n; i++) {
        U64 prod = mont_mul(s_a[i], s_b[n + tid - i], q, q_inv);
        sum = mod_sub(sum, prod, q);
    }
    c[offset + tid] = sum;
#endif
}

// ============================================================================
// NTT-Based Polynomial Multiplication
// ============================================================================

extern "C" __global__ void ntt_forward_stage(
    U64* data, const U64* twiddles, U64 q, U64 q_inv, uint32_t stage, uint32_t log_n)
{
#ifdef __CUDA_ARCH__
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = 1u << log_n;
    uint32_t half_n = n >> 1;
    if (gid >= half_n) return;

    uint32_t half_size = 1u << stage;
    uint32_t group_id = gid / half_size;
    uint32_t idx_in_group = gid % half_size;
    uint32_t idx0 = group_id * (half_size << 1) + idx_in_group;
    uint32_t idx1 = idx0 + half_size;

    U64 x0 = data[idx0], x1 = data[idx1];
    U64 w = twiddles[group_id + half_size];
    ct_butterfly(x0, x1, w, q, q_inv);
    data[idx0] = x0;
    data[idx1] = x1;
#endif
}

extern "C" __global__ void ntt_inverse_stage(
    U64* data, const U64* twiddles_inv, U64 q, U64 q_inv, uint32_t stage, uint32_t log_n)
{
#ifdef __CUDA_ARCH__
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = 1u << log_n;
    uint32_t half_n = n >> 1;
    if (gid >= half_n) return;

    uint32_t half_size = 1u << stage;
    uint32_t group_id = gid / half_size;
    uint32_t idx_in_group = gid % half_size;
    uint32_t idx0 = group_id * (half_size << 1) + idx_in_group;
    uint32_t idx1 = idx0 + half_size;

    U64 x0 = data[idx0], x1 = data[idx1];
    U64 w = twiddles_inv[group_id + half_size];
    gs_butterfly(x0, x1, w, q, q_inv);
    data[idx0] = x0;
    data[idx1] = x1;
#endif
}

extern "C" __global__ void ntt_scale(U64* data, U64 q, U64 q_inv, U64 n_inv, uint32_t n)
{
#ifdef __CUDA_ARCH__
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    data[gid] = mont_mul(data[gid], n_inv, q, q_inv);
#endif
}

// ============================================================================
// Pointwise Operations
// ============================================================================

extern "C" __global__ void poly_mul_pointwise(
    const U64* a_ntt, const U64* b_ntt, U64* c_ntt, U64 q, U64 q_inv, uint32_t n)
{
#ifdef __CUDA_ARCH__
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    c_ntt[gid] = mont_mul(a_ntt[gid], b_ntt[gid], q, q_inv);
#endif
}

extern "C" __global__ void poly_mul_acc(
    const U64* a_ntt, const U64* b_ntt, U64* c_ntt, U64 q, U64 q_inv, uint32_t n)
{
#ifdef __CUDA_ARCH__
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    U64 prod = mont_mul(a_ntt[gid], b_ntt[gid], q, q_inv);
    c_ntt[gid] = mod_add(c_ntt[gid], prod, q);
#endif
}

extern "C" __global__ void poly_scalar_mul(
    const U64* a, U64* c, U64 scalar, U64 q, U64 q_inv, uint32_t n)
{
#ifdef __CUDA_ARCH__
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    c[gid] = mont_mul(a[gid], scalar, q, q_inv);
#endif
}

// ============================================================================
// Batch Operations
// ============================================================================

extern "C" __global__ void poly_batch_mul_pointwise(
    const U64* a, const U64* b, U64* c, U64 q, U64 q_inv, uint32_t n, uint32_t batch_size)
{
#ifdef __CUDA_ARCH__
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = n * batch_size;
    if (gid >= total) return;
    c[gid] = mont_mul(a[gid], b[gid], q, q_inv);
#endif
}

extern "C" __global__ void poly_batch_add(
    const U64* a, const U64* b, U64* c, U64 q, uint32_t n, uint32_t batch_size)
{
#ifdef __CUDA_ARCH__
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = n * batch_size;
    if (gid >= total) return;
    c[gid] = mod_add(a[gid], b[gid], q);
#endif
}

extern "C" __global__ void poly_batch_sub(
    const U64* a, const U64* b, U64* c, U64 q, uint32_t n, uint32_t batch_size)
{
#ifdef __CUDA_ARCH__
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = n * batch_size;
    if (gid >= total) return;
    c[gid] = mod_sub(a[gid], b[gid], q);
#endif
}
