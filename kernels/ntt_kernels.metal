// =============================================================================
// Optimal NTT Kernels for Lux FHE - Ported from OpenFHE's Native Implementation
// =============================================================================
// 
// Design based on OpenFHE's NumberTheoreticTransformNat (transformnat-impl.h):
// - Forward: Cooley-Tukey (DIT) with bit-reversed output
// - Inverse: Gentleman-Sande (GS) with bit-reversed input
// - Barrett reduction with precomputed constants (ModMulFastConst)
// - Peeled first/last stages for performance
// - Branchless arithmetic
//
// Memory layout:
// - twiddles stored as [omega^0, omega^1, ..., omega^{N-1}] in bit-reversed order
// - Data processed in-place [batch, N] where batch is parallelized

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Barrett Reduction Constants
// =============================================================================
//
// For modulus Q, precompute mu = floor(2^k / Q) where k = 2 * ceil(log2(Q))
// Then: a*b mod Q ≈ a*b - floor((a*b*mu) >> k) * Q
//
// For Q < 2^32, we use k = 64, so mu = floor(2^64 / Q)

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

// Compute (a * b) mod Q using Barrett reduction
// Requires: a, b < Q, and precon = floor(2^64 * omega / Q) for omega = b
inline uint64_t mod_mul_barrett(uint64_t a, uint64_t omega, uint64_t Q, uint64_t precon_omega) {
    // High 64 bits of a * precon_omega (approximate quotient)
    uint64_t q_approx = metal::mulhi(a, precon_omega);

    // Compute a * omega - q_approx * Q
    uint64_t product = a * omega;
    uint64_t result = product - q_approx * Q;

    // Conditional reduction (result might be in [0, 2Q))
    return result >= Q ? result - Q : result;
}

// Simple modular multiplication for cases without precomputation
inline uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t Q) {
    // Use mulhi + lo to get full 128-bit result
    uint64_t lo = a * b;
    uint64_t hi = metal::mulhi(a, b);

    // For small Q (< 2^32), hi is often 0
    if (hi == 0) {
        return lo % Q;
    }

    // Full reduction: compute (hi * 2^64 + lo) mod Q
    // 2^64 mod Q = ((2^32 mod Q)^2) mod Q
    uint64_t two64_mod_q = ((uint64_t(1) << 32) % Q);
    two64_mod_q = (two64_mod_q * two64_mod_q) % Q;

    return (lo % Q + (hi % Q) * two64_mod_q % Q) % Q;
}

// Modular addition: (a + b) mod Q
inline uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    // Branchless: use conditional instead of if-statement (Clang optimization)
    return sum - (sum >= Q ? Q : 0);
}

// Modular subtraction: (a - b) mod Q
inline uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    // Branchless subtraction
    return a + (b > a ? Q : 0) - b;
}

// =============================================================================
// Forward NTT - Cooley-Tukey (DIT) In-Place
// =============================================================================
//
// Algorithm (from OpenFHE):
//   for (m = 1, t = n/2, logt = log(n)-1; m < n; m *= 2, t /= 2, --logt)
//     for (i = 0; i < m; ++i)
//       omega = rootOfUnityTable[m + i]
//       for (j1 = i << logt, j2 = j1 + t; j1 < j2; ++j1)
//         loVal = element[j1]
//         hiVal = element[j1 + t] * omega
//         element[j1]     = (loVal + hiVal) mod Q
//         element[j1 + t] = (loVal - hiVal) mod Q
//
// Parallelization: Each stage requires barrier, so we dispatch per-stage
// or use threadgroup memory for small N.

// Single butterfly operation
inline void ct_butterfly(device uint64_t* data, 
                         uint32_t idx_lo, uint32_t idx_hi,
                         uint64_t omega, uint64_t precon_omega,
                         uint64_t Q) {
    uint64_t lo_val = data[idx_lo];
    uint64_t hi_val = data[idx_hi];
    
    // hi_val *= omega (mod Q)
    uint64_t omega_factor = mod_mul_barrett(hi_val, omega, Q, precon_omega);
    
    // CT butterfly: (lo, hi) -> (lo + omega*hi, lo - omega*hi)
    data[idx_lo] = mod_add(lo_val, omega_factor, Q);
    data[idx_hi] = mod_sub(lo_val, omega_factor, Q);
}

// Forward NTT stage kernel
kernel void ntt_forward_stage_optimal(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* twiddles [[buffer(1)]],          // omega in bit-reversed
    constant uint64_t* precon_twiddles [[buffer(2)]],   // precomputed Barrett const
    constant NTTParams& params [[buffer(3)]],
    constant uint32_t& stage [[buffer(4)]],             // 0 to log_N - 1
    constant uint32_t& batch_size [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t butterfly_idx = tid.x;
    
    if (batch_idx >= batch_size) return;
    
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    
    // Stage s: m = 2^s butterflies of size 2^{log_N - s}
    uint32_t m = 1u << stage;
    uint32_t t = N >> (stage + 1);  // half-size
    
    uint32_t num_butterflies = N >> 1;
    if (butterfly_idx >= num_butterflies) return;
    
    // Map butterfly index to (i, j) in the OpenFHE loop structure
    // i = butterfly_idx / t
    // j = butterfly_idx % t
    uint32_t i = butterfly_idx / t;
    uint32_t j = butterfly_idx % t;
    
    uint32_t idx_lo = (i << (params.log_N - stage)) + j;
    uint32_t idx_hi = idx_lo + t;
    
    // Twiddle index: m + i (bit-reversed storage like OpenFHE)
    uint32_t tw_idx = m + i;
    uint64_t omega = twiddles[tw_idx];
    uint64_t precon = precon_twiddles[tw_idx];
    
    device uint64_t* poly = data + batch_idx * N;
    ct_butterfly(poly, idx_lo, idx_hi, omega, precon, Q);
}

// =============================================================================
// Inverse NTT - Gentleman-Sande (DIF) In-Place
// =============================================================================
//
// Algorithm (from OpenFHE):
//   for (m = n/2, t = 1, logt = 1; m >= 1; m /= 2, t *= 2, ++logt)
//     for (i = 0; i < m; ++i)
//       omega = rootOfUnityInverseTable[m + i]
//       for (j1 = i << logt, j2 = j1 + t; j1 < j2; ++j1)
//         loVal = element[j1]
//         hiVal = element[j1 + t]
//         element[j1]     = (loVal + hiVal) mod Q
//         element[j1 + t] = (loVal - hiVal) * omega mod Q
//   for (i = 0; i < n; ++i)
//     element[i] *= cycloOrderInv mod Q

// Single GS butterfly operation
inline void gs_butterfly(device uint64_t* data,
                         uint32_t idx_lo, uint32_t idx_hi,
                         uint64_t omega, uint64_t precon_omega,
                         uint64_t Q) {
    uint64_t lo_val = data[idx_lo];
    uint64_t hi_val = data[idx_hi];
    
    // GS butterfly: (lo, hi) -> (lo + hi, (lo - hi) * omega)
    uint64_t sum = mod_add(lo_val, hi_val, Q);
    uint64_t diff = mod_sub(lo_val, hi_val, Q);
    uint64_t diff_tw = mod_mul_barrett(diff, omega, Q, precon_omega);
    
    data[idx_lo] = sum;
    data[idx_hi] = diff_tw;
}

// Inverse NTT stage kernel
kernel void ntt_inverse_stage_optimal(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* inv_twiddles [[buffer(1)]],
    constant uint64_t* precon_inv_twiddles [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    constant uint32_t& stage [[buffer(4)]],             // 0 to log_N - 1
    constant uint32_t& batch_size [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t butterfly_idx = tid.x;
    
    if (batch_idx >= batch_size) return;
    
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    
    // Stage s: m = N/2^{s+1}, t = 2^s
    uint32_t m = N >> (stage + 1);
    uint32_t t = 1u << stage;
    
    uint32_t num_butterflies = N >> 1;
    if (butterfly_idx >= num_butterflies) return;
    
    // Map butterfly index
    uint32_t i = butterfly_idx / t;
    uint32_t j = butterfly_idx % t;
    
    uint32_t idx_lo = (i << (stage + 1)) + j;
    uint32_t idx_hi = idx_lo + t;
    
    uint32_t tw_idx = m + i;
    uint64_t omega = inv_twiddles[tw_idx];
    uint64_t precon = precon_inv_twiddles[tw_idx];
    
    device uint64_t* poly = data + batch_idx * N;
    gs_butterfly(poly, idx_lo, idx_hi, omega, precon, Q);
}

// Scale by N^{-1} after inverse NTT
kernel void ntt_scale_optimal(
    device uint64_t* data [[buffer(0)]],
    constant NTTParams& params [[buffer(1)]],
    constant uint32_t& batch_size [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t coeff_idx = tid.x;
    
    if (batch_idx >= batch_size || coeff_idx >= params.N) return;
    
    device uint64_t* poly = data + batch_idx * params.N;
    
    poly[coeff_idx] = mod_mul_barrett(
        poly[coeff_idx], 
        params.N_inv, 
        params.Q, 
        params.N_inv_precon
    );
}

// =============================================================================
// Complete Forward NTT (All Stages in Shared Memory)
// =============================================================================
//
// For N <= 1024, process all stages in shared memory with threadgroup barriers.
// This avoids multiple kernel launches.

kernel void ntt_forward_complete_optimal(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* twiddles [[buffer(1)]],
    constant uint64_t* precon_twiddles [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    constant uint32_t& batch_size [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    threadgroup uint64_t* shared [[threadgroup(0)]]
) {
    uint32_t batch_idx = tg_id.y;
    uint32_t local_idx = tid.x % tg_size.x;
    
    if (batch_idx >= batch_size) return;
    
    uint32_t N = params.N;
    uint32_t log_N = params.log_N;
    uint64_t Q = params.Q;
    
    device uint64_t* poly = data + batch_idx * N;
    
    // Load into shared memory
    for (uint32_t i = local_idx; i < N; i += tg_size.x) {
        shared[i] = poly[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Cooley-Tukey stages (OpenFHE structure)
    // for (m = 1, t = n/2; m < n; m *= 2, t /= 2)
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = 1u << stage;
        uint32_t t = N >> (stage + 1);
        
        for (uint32_t butterfly_idx = local_idx; butterfly_idx < N/2; butterfly_idx += tg_size.x) {
            uint32_t i = butterfly_idx / t;
            uint32_t j = butterfly_idx % t;
            
            uint32_t idx_lo = (i << (log_N - stage)) + j;
            uint32_t idx_hi = idx_lo + t;
            
            uint32_t tw_idx = m + i;
            uint64_t omega = twiddles[tw_idx];
            uint64_t precon = precon_twiddles[tw_idx];
            
            uint64_t lo_val = shared[idx_lo];
            uint64_t hi_val = shared[idx_hi];
            
            uint64_t omega_factor = mod_mul_barrett(hi_val, omega, Q, precon);
            
            shared[idx_lo] = mod_add(lo_val, omega_factor, Q);
            shared[idx_hi] = mod_sub(lo_val, omega_factor, Q);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write back to global memory
    for (uint32_t i = local_idx; i < N; i += tg_size.x) {
        poly[i] = shared[i];
    }
}

// =============================================================================
// Complete Inverse NTT (All Stages + Scaling)
// =============================================================================

kernel void ntt_inverse_complete_optimal(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* inv_twiddles [[buffer(1)]],
    constant uint64_t* precon_inv_twiddles [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    constant uint32_t& batch_size [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    threadgroup uint64_t* shared [[threadgroup(0)]]
) {
    uint32_t batch_idx = tg_id.y;
    uint32_t local_idx = tid.x % tg_size.x;
    
    if (batch_idx >= batch_size) return;
    
    uint32_t N = params.N;
    uint32_t log_N = params.log_N;
    uint64_t Q = params.Q;
    uint64_t N_inv = params.N_inv;
    uint64_t N_inv_precon = params.N_inv_precon;
    
    device uint64_t* poly = data + batch_idx * N;
    
    // Load into shared memory
    for (uint32_t i = local_idx; i < N; i += tg_size.x) {
        shared[i] = poly[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Gentleman-Sande stages (OpenFHE structure)
    // for (m = n/2, t = 1; m >= 1; m /= 2, t *= 2)
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = N >> (stage + 1);
        uint32_t t = 1u << stage;
        
        for (uint32_t butterfly_idx = local_idx; butterfly_idx < N/2; butterfly_idx += tg_size.x) {
            uint32_t i = butterfly_idx / t;
            uint32_t j = butterfly_idx % t;
            
            uint32_t idx_lo = (i << (stage + 1)) + j;
            uint32_t idx_hi = idx_lo + t;
            
            uint32_t tw_idx = m + i;
            uint64_t omega = inv_twiddles[tw_idx];
            uint64_t precon = precon_inv_twiddles[tw_idx];
            
            uint64_t lo_val = shared[idx_lo];
            uint64_t hi_val = shared[idx_hi];
            
            uint64_t sum = mod_add(lo_val, hi_val, Q);
            uint64_t diff = mod_sub(lo_val, hi_val, Q);
            uint64_t diff_tw = mod_mul_barrett(diff, omega, Q, precon);
            
            shared[idx_lo] = sum;
            shared[idx_hi] = diff_tw;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Scale by N^{-1} and write back
    for (uint32_t i = local_idx; i < N; i += tg_size.x) {
        poly[i] = mod_mul_barrett(shared[i], N_inv, Q, N_inv_precon);
    }
}

// =============================================================================
// Negacyclic Rotation for Blind Rotation
// =============================================================================
//
// Computes X^k * poly in Z_Q[X]/(X^N + 1)
// For rotation amount k:
//   (X^k * poly)[i] = sign * poly[src]
//   where src = (i - k) mod N, sign = -1 if wrap occurred odd times

kernel void negacyclic_rotate_optimal(
    device uint64_t* output [[buffer(0)]],
    constant uint64_t* input [[buffer(1)]],
    constant NTTParams& params [[buffer(2)]],
    constant int32_t* rotations [[buffer(3)]],
    constant uint32_t& batch_size [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t coeff_idx = tid.x;
    
    if (batch_idx >= batch_size || coeff_idx >= params.N) return;
    
    uint32_t N = params.N;
    uint64_t Q = params.Q;
    
    // Get rotation amount, normalized to [0, 2N)
    int32_t k = rotations[batch_idx];
    int32_t two_N = 2 * (int32_t)N;
    k = ((k % two_N) + two_N) % two_N;
    
    // Compute source index and sign
    // For negacyclic ring: X^N = -1
    // X^k * a[j] X^j contributes to coefficient (j+k) mod 2N
    // with sign = -1 if (j+k) >= N
    
    int32_t src_idx = (int32_t)coeff_idx - k;
    bool negate = false;
    
    // Handle wraparound with negation
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
}

// =============================================================================
// Pointwise Multiply-Accumulate for External Product
// =============================================================================
//
// In Lux FHE external product: acc += digit_ntt * rgsw_component_ntt
// This is the core operation called many times per blind rotation step.

kernel void ntt_pointwise_mac_optimal(
    device uint64_t* acc [[buffer(0)]],
    constant uint64_t* a [[buffer(1)]],
    constant uint64_t* b [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    constant uint32_t& batch_size [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t coeff_idx = tid.x;
    
    if (batch_idx >= batch_size || coeff_idx >= params.N) return;
    
    uint32_t idx = batch_idx * params.N + coeff_idx;
    uint64_t Q = params.Q;
    
    // Simple modular MAC (without Barrett for the multiply since operands may not have precon)
    uint64_t prod = mod_mul(a[idx], b[idx], Q);
    acc[idx] = mod_add(acc[idx], prod, Q);
}

// =============================================================================
// Digit Decomposition for External Product
// =============================================================================
//
// Decompose polynomial coefficients into base-B digits for RGSW multiplication.
// a[i] = sum_{l=0}^{L-1} d_l[i] * B^l where d_l[i] in [0, B)

kernel void decompose_digits(
    device uint64_t* digits [[buffer(0)]],     // Output: [batch, L, N]
    constant uint64_t* poly [[buffer(1)]],      // Input: [batch, N]
    constant NTTParams& params [[buffer(2)]],
    constant uint64_t& base [[buffer(3)]],      // Decomposition base B
    constant uint32_t& num_levels [[buffer(4)]],// L decomposition levels
    constant uint32_t& batch_size [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.z;
    uint32_t level = tid.y;
    uint32_t coeff_idx = tid.x;
    
    if (batch_idx >= batch_size || level >= num_levels || coeff_idx >= params.N) return;
    
    uint64_t val = poly[batch_idx * params.N + coeff_idx];
    
    // Extract digit at level l: floor(val / B^l) mod B
    for (uint32_t l = 0; l < level; ++l) {
        val /= base;
    }
    uint64_t digit = val % base;
    
    // Store in [batch, level, N] layout
    digits[batch_idx * num_levels * params.N + level * params.N + coeff_idx] = digit;
}

// =============================================================================
// CMux for Blind Rotation
// =============================================================================
//
// CMux(s, d0, d1) = d0 + s * (d1 - d0)
// Where s is the selector (RGSW ciphertext), d0, d1 are RLWE ciphertexts.
// 
// In practice: acc = acc + ExternalProduct(rotated_acc - acc, bsk[i])
// This kernel computes: d1 - d0 (the difference for external product input)

kernel void cmux_diff(
    device uint64_t* diff [[buffer(0)]],       // Output: d1 - d0
    constant uint64_t* d0 [[buffer(1)]],       // Unrotated accumulator
    constant uint64_t* d1 [[buffer(2)]],       // Rotated accumulator
    constant NTTParams& params [[buffer(3)]],
    constant uint32_t& batch_size [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t coeff_idx = tid.x;
    
    if (batch_idx >= batch_size || coeff_idx >= params.N) return;
    
    uint32_t idx = batch_idx * params.N + coeff_idx;
    diff[idx] = mod_sub(d1[idx], d0[idx], params.Q);
}

// =============================================================================
// External Product Accumulate
// =============================================================================
//
// Accumulates: acc += sum_{l=0}^{L-1} INTT(NTT(digit_l) * RGSW_l_ntt)
// This is the final step combining all decomposition levels.

kernel void external_product_finalize(
    device uint64_t* acc [[buffer(0)]],        // Accumulator (output)
    constant uint64_t* prod [[buffer(1)]],     // Product from pointwise multiply [batch, L, N]
    constant NTTParams& params [[buffer(2)]],
    constant uint32_t& num_levels [[buffer(3)]],
    constant uint32_t& batch_size [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t coeff_idx = tid.x;
    
    if (batch_idx >= batch_size || coeff_idx >= params.N) return;
    
    uint64_t Q = params.Q;
    uint64_t sum = 0;
    
    // Sum over all decomposition levels
    for (uint32_t l = 0; l < num_levels; ++l) {
        uint32_t idx = batch_idx * num_levels * params.N + l * params.N + coeff_idx;
        sum = mod_add(sum, prod[idx], Q);
    }
    
    uint32_t out_idx = batch_idx * params.N + coeff_idx;
    acc[out_idx] = mod_add(acc[out_idx], sum, Q);
}
