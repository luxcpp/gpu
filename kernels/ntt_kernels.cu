// =============================================================================
// Optimal NTT Kernels for Lux FHE - CUDA Port of ntt_kernels.metal
// =============================================================================
//
// Design based on OpenFHE's NumberTheoreticTransformNat:
// - Forward: Cooley-Tukey (DIT) with bit-reversed output
// - Inverse: Gentleman-Sande (GS) with bit-reversed input
// - Barrett reduction with precomputed constants (ModMulFastConst)
// - Byte-identical output to Metal implementation
//
// CUDA advantages over Metal for this kernel:
// - Native __umul64hi for 64-bit mulhi
// - Larger shared memory (48KB vs 32KB)
// - Warp-synchronous execution eliminates some barriers

#include <cstdint>

#ifdef __CUDA_ARCH__
#define NTT_DEVICE __device__ __forceinline__
#else
#define NTT_DEVICE inline
#define __global__
#define __shared__
static inline uint64_t __umul64hi(uint64_t a, uint64_t b) {
    __uint128_t r = (__uint128_t)a * b;
    return (uint64_t)(r >> 64);
}
#endif

// =============================================================================
// NTT Parameters Structure
// =============================================================================

struct NTTParams {
    uint64_t Q;           // Prime modulus
    uint64_t mu;          // Barrett constant: floor(2^64 / Q)
    uint64_t N_inv;       // N^{-1} mod Q
    uint64_t N_inv_precon; // Barrett precomputation for N_inv
    uint32_t N;           // Ring dimension (power of 2)
    uint32_t log_N;       // log2(N)
};

// =============================================================================
// Barrett Modular Multiplication
// =============================================================================

NTT_DEVICE uint64_t mod_mul_barrett(uint64_t a, uint64_t omega, uint64_t Q, uint64_t precon_omega) {
    uint64_t q_approx = __umul64hi(a, precon_omega);
    uint64_t product = a * omega;
    uint64_t result = product - q_approx * Q;
    return result >= Q ? result - Q : result;
}

NTT_DEVICE uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t Q) {
#ifdef __CUDA_ARCH__
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);

    if (hi == 0) {
        return lo % Q;
    }

    uint64_t two64_mod_q = ((uint64_t(1) << 32) % Q);
    two64_mod_q = (two64_mod_q * two64_mod_q) % Q;

    return (lo % Q + (hi % Q) * two64_mod_q % Q) % Q;
#else
    __uint128_t prod = (__uint128_t)a * b;
    return (uint64_t)(prod % Q);
#endif
}

NTT_DEVICE uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    return sum - (sum >= Q ? Q : 0);
}

NTT_DEVICE uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    return a + (b > a ? Q : 0) - b;
}

// =============================================================================
// Butterfly Operations
// =============================================================================

NTT_DEVICE void ct_butterfly(uint64_t* data,
                              uint32_t idx_lo, uint32_t idx_hi,
                              uint64_t omega, uint64_t precon_omega,
                              uint64_t Q) {
    uint64_t lo_val = data[idx_lo];
    uint64_t hi_val = data[idx_hi];
    uint64_t omega_factor = mod_mul_barrett(hi_val, omega, Q, precon_omega);
    data[idx_lo] = mod_add(lo_val, omega_factor, Q);
    data[idx_hi] = mod_sub(lo_val, omega_factor, Q);
}

NTT_DEVICE void gs_butterfly(uint64_t* data,
                              uint32_t idx_lo, uint32_t idx_hi,
                              uint64_t omega, uint64_t precon_omega,
                              uint64_t Q) {
    uint64_t lo_val = data[idx_lo];
    uint64_t hi_val = data[idx_hi];
    uint64_t sum = mod_add(lo_val, hi_val, Q);
    uint64_t diff = mod_sub(lo_val, hi_val, Q);
    uint64_t diff_tw = mod_mul_barrett(diff, omega, Q, precon_omega);
    data[idx_lo] = sum;
    data[idx_hi] = diff_tw;
}

// =============================================================================
// Forward NTT Stage Kernel
// =============================================================================

extern "C" __global__ void ntt_forward_stage_optimal(
    uint64_t* data,
    const uint64_t* twiddles,
    const uint64_t* precon_twiddles,
    const NTTParams params,
    uint32_t stage,
    uint32_t batch_size)
{
#ifdef __CUDA_ARCH__
    uint32_t batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t butterfly_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size) return;

    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint32_t m = 1u << stage;
    uint32_t t = N >> (stage + 1);
    uint32_t num_butterflies = N >> 1;
    if (butterfly_idx >= num_butterflies) return;

    uint32_t i = butterfly_idx / t;
    uint32_t j = butterfly_idx % t;
    uint32_t idx_lo = (i << (params.log_N - stage)) + j;
    uint32_t idx_hi = idx_lo + t;
    uint32_t tw_idx = m + i;
    uint64_t omega = twiddles[tw_idx];
    uint64_t precon = precon_twiddles[tw_idx];

    uint64_t* poly = data + batch_idx * N;
    ct_butterfly(poly, idx_lo, idx_hi, omega, precon, Q);
#endif
}

// =============================================================================
// Inverse NTT Stage Kernel
// =============================================================================

extern "C" __global__ void ntt_inverse_stage_optimal(
    uint64_t* data,
    const uint64_t* inv_twiddles,
    const uint64_t* precon_inv_twiddles,
    const NTTParams params,
    uint32_t stage,
    uint32_t batch_size)
{
#ifdef __CUDA_ARCH__
    uint32_t batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t butterfly_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size) return;

    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint32_t m = N >> (stage + 1);
    uint32_t t = 1u << stage;
    uint32_t num_butterflies = N >> 1;
    if (butterfly_idx >= num_butterflies) return;

    uint32_t i = butterfly_idx / t;
    uint32_t j = butterfly_idx % t;
    uint32_t idx_lo = (i << (stage + 1)) + j;
    uint32_t idx_hi = idx_lo + t;
    uint32_t tw_idx = m + i;
    uint64_t omega = inv_twiddles[tw_idx];
    uint64_t precon = precon_inv_twiddles[tw_idx];

    uint64_t* poly = data + batch_idx * N;
    gs_butterfly(poly, idx_lo, idx_hi, omega, precon, Q);
#endif
}

// =============================================================================
// Scale by N^{-1} after inverse NTT
// =============================================================================

extern "C" __global__ void ntt_scale_optimal(
    uint64_t* data,
    const NTTParams params,
    uint32_t batch_size)
{
#ifdef __CUDA_ARCH__
    uint32_t batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || coeff_idx >= params.N) return;

    uint64_t* poly = data + batch_idx * params.N;
    poly[coeff_idx] = mod_mul_barrett(poly[coeff_idx], params.N_inv, params.Q, params.N_inv_precon);
#endif
}

// =============================================================================
// Complete Forward NTT (All Stages in Shared Memory)
// =============================================================================

extern "C" __global__ void ntt_forward_complete_optimal(
    uint64_t* data,
    const uint64_t* twiddles,
    const uint64_t* precon_twiddles,
    const NTTParams params,
    uint32_t batch_size)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared[];

    uint32_t batch_idx = blockIdx.y;
    uint32_t local_idx = threadIdx.x;
    uint32_t tg_size = blockDim.x;

    if (batch_idx >= batch_size) return;

    uint32_t N = params.N;
    uint32_t log_N = params.log_N;
    uint64_t Q = params.Q;
    uint64_t* poly = data + batch_idx * N;

    // Load into shared memory
    for (uint32_t i = local_idx; i < N; i += tg_size) {
        shared[i] = poly[i];
    }
    __syncthreads();

    // Cooley-Tukey stages
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = 1u << stage;
        uint32_t t = N >> (stage + 1);

        for (uint32_t butterfly_idx = local_idx; butterfly_idx < N / 2; butterfly_idx += tg_size) {
            uint32_t i = butterfly_idx / t;
            uint32_t j = butterfly_idx % t;
            uint32_t idx_lo = (i << (log_N - stage)) + j;
            uint32_t idx_hi = idx_lo + t;
            uint32_t tw_idx = m + i;

            uint64_t lo_val = shared[idx_lo];
            uint64_t hi_val = shared[idx_hi];
            uint64_t omega = twiddles[tw_idx];
            uint64_t precon = precon_twiddles[tw_idx];
            uint64_t omega_factor = mod_mul_barrett(hi_val, omega, Q, precon);

            shared[idx_lo] = mod_add(lo_val, omega_factor, Q);
            shared[idx_hi] = mod_sub(lo_val, omega_factor, Q);
        }
        __syncthreads();
    }

    // Write back
    for (uint32_t i = local_idx; i < N; i += tg_size) {
        poly[i] = shared[i];
    }
#endif
}

// =============================================================================
// Complete Inverse NTT (All Stages + Scaling)
// =============================================================================

extern "C" __global__ void ntt_inverse_complete_optimal(
    uint64_t* data,
    const uint64_t* inv_twiddles,
    const uint64_t* precon_inv_twiddles,
    const NTTParams params,
    uint32_t batch_size)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t shared[];

    uint32_t batch_idx = blockIdx.y;
    uint32_t local_idx = threadIdx.x;
    uint32_t tg_size = blockDim.x;

    if (batch_idx >= batch_size) return;

    uint32_t N = params.N;
    uint32_t log_N = params.log_N;
    uint64_t Q = params.Q;
    uint64_t N_inv = params.N_inv;
    uint64_t N_inv_precon = params.N_inv_precon;
    uint64_t* poly = data + batch_idx * N;

    for (uint32_t i = local_idx; i < N; i += tg_size) {
        shared[i] = poly[i];
    }
    __syncthreads();

    // Gentleman-Sande stages
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = N >> (stage + 1);
        uint32_t t = 1u << stage;

        for (uint32_t butterfly_idx = local_idx; butterfly_idx < N / 2; butterfly_idx += tg_size) {
            uint32_t i = butterfly_idx / t;
            uint32_t j = butterfly_idx % t;
            uint32_t idx_lo = (i << (stage + 1)) + j;
            uint32_t idx_hi = idx_lo + t;
            uint32_t tw_idx = m + i;

            uint64_t lo_val = shared[idx_lo];
            uint64_t hi_val = shared[idx_hi];
            uint64_t omega = inv_twiddles[tw_idx];
            uint64_t precon = precon_inv_twiddles[tw_idx];

            shared[idx_lo] = mod_add(lo_val, hi_val, Q);
            uint64_t diff = mod_sub(lo_val, hi_val, Q);
            shared[idx_hi] = mod_mul_barrett(diff, omega, Q, precon);
        }
        __syncthreads();
    }

    // Scale by N^{-1} and write back
    for (uint32_t i = local_idx; i < N; i += tg_size) {
        poly[i] = mod_mul_barrett(shared[i], N_inv, Q, N_inv_precon);
    }
#endif
}

// =============================================================================
// Negacyclic Rotation for Blind Rotation
// =============================================================================

extern "C" __global__ void negacyclic_rotate_optimal(
    uint64_t* output,
    const uint64_t* input,
    const NTTParams params,
    const int32_t* rotations,
    uint32_t batch_size)
{
#ifdef __CUDA_ARCH__
    uint32_t batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || coeff_idx >= params.N) return;

    uint32_t N = params.N;
    uint64_t Q = params.Q;

    int32_t k = rotations[batch_idx];
    int32_t two_N = 2 * (int32_t)N;
    k = ((k % two_N) + two_N) % two_N;

    int32_t src_idx = (int32_t)coeff_idx - k;
    bool negate = false;

    while (src_idx < 0) {
        src_idx += N;
        negate = !negate;
    }
    while (src_idx >= (int32_t)N) {
        src_idx -= N;
        negate = !negate;
    }

    uint32_t in_offset = batch_idx * N + (uint32_t)src_idx;
    uint32_t out_offset = batch_idx * N + coeff_idx;

    uint64_t val = input[in_offset];
    output[out_offset] = negate ? (Q - val) : val;
#endif
}

// =============================================================================
// Pointwise Multiply-Accumulate for External Product
// =============================================================================

extern "C" __global__ void ntt_pointwise_mac_optimal(
    uint64_t* acc,
    const uint64_t* a,
    const uint64_t* b,
    const NTTParams params,
    uint32_t batch_size)
{
#ifdef __CUDA_ARCH__
    uint32_t batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || coeff_idx >= params.N) return;

    uint32_t idx = batch_idx * params.N + coeff_idx;
    uint64_t Q = params.Q;
    uint64_t prod = mod_mul(a[idx], b[idx], Q);
    acc[idx] = mod_add(acc[idx], prod, Q);
#endif
}

// =============================================================================
// Digit Decomposition for External Product
// =============================================================================

extern "C" __global__ void decompose_digits(
    uint64_t* digits,
    const uint64_t* poly,
    const NTTParams params,
    uint64_t base,
    uint32_t num_levels,
    uint32_t batch_size)
{
#ifdef __CUDA_ARCH__
    uint32_t batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    uint32_t level = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || level >= num_levels || coeff_idx >= params.N) return;

    uint64_t val = poly[batch_idx * params.N + coeff_idx];

    for (uint32_t l = 0; l < level; ++l) {
        val /= base;
    }
    uint64_t digit = val % base;

    digits[batch_idx * num_levels * params.N + level * params.N + coeff_idx] = digit;
#endif
}

// =============================================================================
// CMux Difference
// =============================================================================

extern "C" __global__ void cmux_diff(
    uint64_t* diff,
    const uint64_t* d0,
    const uint64_t* d1,
    const NTTParams params,
    uint32_t batch_size)
{
#ifdef __CUDA_ARCH__
    uint32_t batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || coeff_idx >= params.N) return;

    uint32_t idx = batch_idx * params.N + coeff_idx;
    diff[idx] = mod_sub(d1[idx], d0[idx], params.Q);
#endif
}

// =============================================================================
// External Product Finalize
// =============================================================================

extern "C" __global__ void external_product_finalize(
    uint64_t* acc,
    const uint64_t* prod,
    const NTTParams params,
    uint32_t num_levels,
    uint32_t batch_size)
{
#ifdef __CUDA_ARCH__
    uint32_t batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || coeff_idx >= params.N) return;

    uint64_t Q = params.Q;
    uint64_t sum = 0;

    for (uint32_t l = 0; l < num_levels; ++l) {
        uint32_t idx = batch_idx * num_levels * params.N + l * params.N + coeff_idx;
        sum = mod_add(sum, prod[idx], Q);
    }

    uint32_t out_idx = batch_idx * params.N + coeff_idx;
    acc[out_idx] = mod_add(acc[out_idx], sum, Q);
#endif
}
