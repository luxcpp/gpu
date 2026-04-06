// =============================================================================
// Lux FHE GPU Kernels - Optimized for Apple Metal (M1/M2/M3)
// =============================================================================
//
// Kernel Architecture:
//   1. Fused NTT with shared memory optimization
//   2. Fused CMux (rotate + decompose + external product)  
//   3. Pipelined Key Switching
//   4. Barrett/Montgomery modular arithmetic
//
// Memory Layout: Structure of Arrays (SoA) for coalesced access
// Thread Organization: [coefficients, components, batch]
//
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// COMPILE-TIME CONFIGURATION (via function constants)
// =============================================================================

// Ring parameters
constant uint N [[function_constant(0)]];           // Ring dimension (1024)
constant uint LOG_N [[function_constant(1)]];       // log2(N) = 10
constant uint L [[function_constant(2)]];           // Decomposition levels (4)
constant uint BASE_LOG [[function_constant(3)]];    // log2(base) (7)

// LWE parameters  
constant uint n_lwe [[function_constant(4)]];       // LWE dimension (512)

// Modular arithmetic constants
constant uint64_t Q [[function_constant(5)]];       // Ring modulus (~2^28)
constant uint64_t Q_INV [[function_constant(6)]];   // -Q^-1 mod 2^32 (Montgomery)
constant uint64_t BARRETT_MU [[function_constant(7)]]; // floor(2^56 / Q)

// Derived constants
constant uint64_t BASE_MASK = (1ULL << BASE_LOG) - 1;
constant uint TWO_N = 2 * N;

// =============================================================================
// MODULAR ARITHMETIC
// =============================================================================

// Barrett reduction: x mod Q for x < Q^2
// Precompute: mu = floor(2^56 / Q)
inline uint64_t barrett_reduce(uint64_t x) {
    // q_hat = floor(x * mu / 2^56)
    uint64_t q_hat = metal::mulhi(x, BARRETT_MU);  // upper 64 bits of x * mu
    
    // r = x - q_hat * Q
    uint64_t r = x - q_hat * Q;
    
    // Conditional subtraction
    return r >= Q ? r - Q : r;
}

// Modular addition: (a + b) mod Q
inline uint64_t mod_add(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    return sum >= Q ? sum - Q : sum;
}

// Modular subtraction: (a - b) mod Q
inline uint64_t mod_sub(uint64_t a, uint64_t b) {
    return a >= b ? a - b : Q - b + a;
}

// Modular multiplication: (a * b) mod Q using Barrett reduction
inline uint64_t mod_mul(uint64_t a, uint64_t b) {
    // For Q < 2^28 and a, b < Q: product < 2^56
    uint64_t product = a * b;
    return barrett_reduce(product);
}

// Modular negation: -a mod Q
inline uint64_t mod_neg(uint64_t a) {
    return a == 0 ? 0 : Q - a;
}

// =============================================================================
// SHARED MEMORY STRUCTURES
// =============================================================================

// NTT shared memory (for N=1024, using 2x512 ping-pong buffers)
struct NTTSharedMem {
    uint64_t data[1024];
    uint64_t twiddles[512];
};

// CMux shared memory (for digit decomposition)
struct CMuxSharedMem {
    uint64_t diff[2][1024];      // diff[component][coefficient]
    uint64_t digits[2][4][256];  // digits[comp][level][chunk] - processed in chunks
};

// =============================================================================
// KERNEL 1: FORWARD NTT (Cooley-Tukey, Decimation in Time)
// =============================================================================
// 
// Input:  data[batch, N] - polynomials in coefficient domain
// Output: data[batch, N] - polynomials in NTT domain
//
// Grid:   [N/2, batch, 1]
// Threads: [256, 1, 1]
//
// Executes log2(N) stages with threadgroup synchronization

kernel void ntt_forward_fused(
    device uint64_t* data               [[buffer(0)]],
    constant uint64_t* twiddles         [[buffer(1)]],  // [N] precomputed omega^i
    constant uint& ring_dim             [[buffer(2)]],
    constant uint& log_ring_dim         [[buffer(3)]],
    constant uint& batch_size           [[buffer(4)]],
    
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tg_size3 [[threads_per_threadgroup]],
    
    threadgroup NTTSharedMem& shared [[threadgroup(0)]]
) {
    uint tg_size = tg_size3.x;
    uint batch_idx = tgid.y;
    uint local_idx = tid.x;

    if (batch_idx >= batch_size) return;

    device uint64_t* poly = data + batch_idx * ring_dim;

    // Load data to shared memory with bit-reversal permutation
    // Each thread loads 4 elements (for N=1024, 256 threads)
    for (uint i = local_idx; i < ring_dim; i += tg_size) {
        // Bit-reverse index
        uint j = 0;
        uint temp = i;
        for (uint k = 0; k < log_ring_dim; k++) {
            j = (j << 1) | (temp & 1);
            temp >>= 1;
        }
        shared.data[j] = poly[i];
    }

    // Load twiddle factors to shared memory
    for (uint i = local_idx; i < ring_dim / 2; i += tg_size) {
        shared.twiddles[i] = twiddles[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooley-Tukey butterfly stages
    for (uint stage = 0; stage < log_ring_dim; stage++) {
        uint len = 1u << (stage + 1);    // Butterfly span
        uint half_len = len >> 1;         // Half span
        uint step = ring_dim / len;       // Twiddle step

        // Each thread handles multiple butterflies
        for (uint idx = local_idx; idx < ring_dim / 2; idx += tg_size) {
            uint group = idx / half_len;
            uint j = idx % half_len;
            uint i = group * len + j;

            uint64_t w = shared.twiddles[j * step];
            uint64_t a = shared.data[i];
            uint64_t b = shared.data[i + half_len];

            // Butterfly: (a, b) -> (a + w*b, a - w*b)
            uint64_t wb = mod_mul(w, b);
            shared.data[i] = mod_add(a, wb);
            shared.data[i + half_len] = mod_sub(a, wb);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back to global memory
    for (uint i = local_idx; i < ring_dim; i += tg_size) {
        poly[i] = shared.data[i];
    }
}

// =============================================================================
// KERNEL 2: INVERSE NTT (Gentleman-Sande, Decimation in Frequency)
// =============================================================================

kernel void ntt_inverse_fused(
    device uint64_t* data               [[buffer(0)]],
    constant uint64_t* inv_twiddles     [[buffer(1)]],  // [N] precomputed omega^{-i}
    constant uint64_t& n_inv            [[buffer(2)]],  // N^{-1} mod Q
    constant uint& ring_dim             [[buffer(3)]],
    constant uint& log_ring_dim         [[buffer(4)]],
    constant uint& batch_size           [[buffer(5)]],
    
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tg_size3 [[threads_per_threadgroup]],
    
    threadgroup NTTSharedMem& shared [[threadgroup(0)]]
) {
    uint tg_size = tg_size3.x;
    uint batch_idx = tgid.y;
    uint local_idx = tid.x;

    if (batch_idx >= batch_size) return;

    device uint64_t* poly = data + batch_idx * ring_dim;

    // Load data and twiddles to shared memory
    for (uint i = local_idx; i < ring_dim; i += tg_size) {
        shared.data[i] = poly[i];
    }
    for (uint i = local_idx; i < ring_dim / 2; i += tg_size) {
        shared.twiddles[i] = inv_twiddles[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Gentleman-Sande butterfly stages (reverse order from forward)
    for (int stage = log_ring_dim - 1; stage >= 0; stage--) {
        uint len = 1u << (stage + 1);
        uint half_len = len >> 1;
        uint step = ring_dim / len;

        for (uint idx = local_idx; idx < ring_dim / 2; idx += tg_size) {
            uint group = idx / half_len;
            uint j = idx % half_len;
            uint i = group * len + j;

            uint64_t w = shared.twiddles[j * step];
            uint64_t a = shared.data[i];
            uint64_t b = shared.data[i + half_len];

            // Butterfly: (a, b) -> (a + b, (a - b) * w)
            shared.data[i] = mod_add(a, b);
            shared.data[i + half_len] = mod_mul(mod_sub(a, b), w);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Scale by N^{-1} and write back with bit-reversal
    for (uint i = local_idx; i < ring_dim; i += tg_size) {
        // Bit-reverse index for output
        uint j = 0;
        uint temp = i;
        for (uint k = 0; k < log_ring_dim; k++) {
            j = (j << 1) | (temp & 1);
            temp >>= 1;
        }
        poly[j] = mod_mul(shared.data[i], n_inv);
    }
}

// =============================================================================
// KERNEL 3: INITIALIZE ACCUMULATOR
// =============================================================================
//
// Initializes acc = (0, X^{-b} * TestPoly) for blind rotation
// TestPoly encodes the gate function
//
// Input:  test_poly[N] - test polynomial for the gate
//         rotations[batch] - rotation amounts (-b mod 2N)
// Output: acc[batch, 2, N] - initialized accumulators

kernel void init_accumulator(
    device uint64_t* acc                [[buffer(0)]],  // [B, 2, N] output
    constant uint64_t* test_poly        [[buffer(1)]],  // [N] test polynomial
    constant int* rotations             [[buffer(2)]],  // [B] rotation amounts
    constant uint& ring_dim             [[buffer(3)]],
    constant uint& batch_size           [[buffer(4)]],
    
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint coeff_idx = gid.x;
    uint batch_idx = gid.z;
    
    if (coeff_idx >= ring_dim || batch_idx >= batch_size) return;
    
    int rotation = rotations[batch_idx];
    uint two_n = 2 * ring_dim;
    
    // Normalize rotation to [0, 2N)
    rotation = ((rotation % (int)two_n) + (int)two_n) % (int)two_n;
    
    // Find source index for X^{-rotation} * TestPoly
    int src_idx = (int)coeff_idx + rotation;
    bool negate = false;
    
    if (src_idx >= (int)two_n) {
        src_idx -= two_n;
    }
    if (src_idx >= (int)ring_dim) {
        src_idx -= ring_dim;
        negate = true;  // X^N = -1 in negacyclic ring
    }
    
    uint64_t val = test_poly[src_idx];
    if (negate) val = mod_neg(val);
    
    // Write to accumulator: acc[batch, 0, :] = 0, acc[batch, 1, :] = rotated test poly
    uint out_idx_c0 = batch_idx * 2 * ring_dim + coeff_idx;
    uint out_idx_c1 = batch_idx * 2 * ring_dim + ring_dim + coeff_idx;
    
    acc[out_idx_c0] = 0;
    acc[out_idx_c1] = val;
}

// =============================================================================
// KERNEL 4: FUSED CMUX GATE
// =============================================================================
//
// CMux(BK[i], acc, a[i]) = acc + ExternalProduct(X^{a[i]} * acc - acc, BK[i])
//
// This kernel fuses:
//   1. Negacyclic rotation by a[i]
//   2. Difference computation (rotated - original)
//   3. Digit decomposition  
//   4. External product accumulation
//
// All operations are in NTT domain (pointwise)
//
// Input:  acc[batch, 2, N] - accumulators (NTT domain)
//         bsk[2, L, 2, N] - bootstrap key for current LWE index i
//         rotations[batch] - a[i] values
// Output: acc[batch, 2, N] - updated accumulators (in-place)

kernel void cmux_fused(
    device uint64_t* acc                [[buffer(0)]],  // [B, 2, N] in/out
    constant uint64_t* bsk              [[buffer(1)]],  // [2, L, 2, N] current BK[i]
    constant int* rotations             [[buffer(2)]],  // [B] rotation amounts a[i]
    constant uint& ring_dim             [[buffer(3)]],
    constant uint& num_levels           [[buffer(4)]],  // L
    constant uint& decomp_log           [[buffer(5)]],  // BASE_LOG
    constant uint& batch_size           [[buffer(6)]],
    
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tg_size3 [[threads_per_threadgroup]]
) {
    uint tg_size = tg_size3.x;
    uint coeff_idx = gid.x;
    uint out_comp = gid.y;      // Output component (0 or 1)
    uint batch_idx = gid.z;
    
    if (coeff_idx >= ring_dim || out_comp >= 2 || batch_idx >= batch_size) return;
    
    int rotation = rotations[batch_idx];
    uint64_t mask = (1ULL << decomp_log) - 1;
    uint two_n = 2 * ring_dim;
    
    // Skip if no rotation (optimization for sparse LWE)
    if (rotation == 0) return;
    
    // Normalize rotation
    rotation = ((rotation % (int)two_n) + (int)two_n) % (int)two_n;
    
    // Compute source index for rotation (in NTT domain, rotation is index shift)
    // Note: In NTT domain, X^k multiplication is a cyclic shift + sign changes
    int src_idx = (int)coeff_idx - rotation;
    bool negate = false;
    
    if (src_idx < 0) src_idx += two_n;
    if (src_idx >= (int)ring_dim) {
        src_idx -= ring_dim;
        negate = true;
    }
    
    // Load original and compute rotated values for both components
    uint64_t acc_orig[2];
    uint64_t acc_rot[2];
    uint64_t diff[2];
    
    for (int c = 0; c < 2; c++) {
        uint orig_idx = batch_idx * 2 * ring_dim + c * ring_dim + coeff_idx;
        uint rot_idx = batch_idx * 2 * ring_dim + c * ring_dim + src_idx;
        
        acc_orig[c] = acc[orig_idx];
        acc_rot[c] = acc[rot_idx];
        if (negate) acc_rot[c] = mod_neg(acc_rot[c]);
        
        // diff = rotated - original
        diff[c] = mod_sub(acc_rot[c], acc_orig[c]);
    }
    
    // Accumulate external product: sum over input components and decomposition levels
    uint64_t ext_prod_sum = 0;
    
    for (uint in_comp = 0; in_comp < 2; in_comp++) {
        uint64_t val = diff[in_comp];
        
        for (uint l = 0; l < num_levels; l++) {
            // Extract digit l
            uint64_t digit = (val >> (l * decomp_log)) & mask;
            
            // Look up BSK value: bsk[in_comp, l, out_comp, coeff]
            // Layout: [2, L, 2, N] -> index = in_comp * L*2*N + l * 2*N + out_comp * N + coeff
            uint bsk_idx = in_comp * num_levels * 2 * ring_dim +
                          l * 2 * ring_dim +
                          out_comp * ring_dim +
                          coeff_idx;
            uint64_t bsk_val = bsk[bsk_idx];
            
            // Accumulate: digit * bsk (pointwise in NTT domain)
            ext_prod_sum = mod_add(ext_prod_sum, mod_mul(digit, bsk_val));
        }
    }
    
    // Update accumulator: acc += external_product_result
    uint out_idx = batch_idx * 2 * ring_dim + out_comp * ring_dim + coeff_idx;
    acc[out_idx] = mod_add(acc_orig[out_comp], ext_prod_sum);
}

// =============================================================================
// KERNEL 5: KEY SWITCHING (RLWE to LWE)
// =============================================================================
//
// Converts RLWE(N) to LWE(n) using key switching key
// KSK[j, l] = LWE encryptions of s_RLWE[j] * 2^{l * base_log} under s_LWE
//
// Input:  rlwe[batch, 2, N] - RLWE ciphertexts (coefficient domain)
//         ksk[N, L_ks, n+1] - key switching key
// Output: lwe[batch, n+1] - LWE ciphertexts

kernel void key_switch_decompose(
    device uint64_t* digits             [[buffer(0)]],  // [B, N, L_ks] output digits
    constant uint64_t* rlwe             [[buffer(1)]],  // [B, 2, N] RLWE c0 component
    constant uint& ring_dim             [[buffer(2)]],
    constant uint& num_levels           [[buffer(3)]],  // L_ks
    constant uint& decomp_log           [[buffer(4)]],
    constant uint& batch_size           [[buffer(5)]],
    
    uint3 gid [[thread_position_in_grid]]
) {
    uint coeff_idx = gid.x;
    uint batch_idx = gid.y;
    
    if (coeff_idx >= ring_dim || batch_idx >= batch_size) return;
    
    uint64_t mask = (1ULL << decomp_log) - 1;
    
    // Load c0[coeff] from RLWE
    uint rlwe_idx = batch_idx * 2 * ring_dim + coeff_idx;  // c0 is first component
    uint64_t val = rlwe[rlwe_idx];
    
    // Decompose into L_ks digits
    for (uint l = 0; l < num_levels; l++) {
        uint digit_idx = batch_idx * ring_dim * num_levels + coeff_idx * num_levels + l;
        digits[digit_idx] = (val >> (l * decomp_log)) & mask;
    }
}

kernel void key_switch_accumulate(
    device uint64_t* lwe                [[buffer(0)]],  // [B, n+1] output
    constant uint64_t* digits           [[buffer(1)]],  // [B, N, L_ks] decomposed digits
    constant uint64_t* ksk              [[buffer(2)]],  // [N, L_ks, n+1] key switching key
    constant uint64_t* rlwe_c1          [[buffer(3)]],  // [B, N] RLWE c1 for body extraction
    constant uint& ring_dim             [[buffer(4)]],  // N
    constant uint& lwe_dim              [[buffer(5)]],  // n
    constant uint& num_levels           [[buffer(6)]],  // L_ks
    constant uint& batch_size           [[buffer(7)]],
    
    uint3 gid [[thread_position_in_grid]]
) {
    uint lwe_idx = gid.x;   // LWE coefficient index (0..n)
    uint batch_idx = gid.y;
    
    if (lwe_idx > lwe_dim || batch_idx >= batch_size) return;
    
    uint64_t sum = 0;
    
    // Accumulate: sum over RLWE coefficients and decomposition levels
    for (uint j = 0; j < ring_dim; j++) {
        for (uint l = 0; l < num_levels; l++) {
            // Get digit[j, l]
            uint digit_idx = batch_idx * ring_dim * num_levels + j * num_levels + l;
            uint64_t digit = digits[digit_idx];
            
            if (digit == 0) continue;  // Skip zero digits
            
            // Get KSK[j, l, lwe_idx]
            uint ksk_idx = j * num_levels * (lwe_dim + 1) + l * (lwe_dim + 1) + lwe_idx;
            uint64_t ksk_val = ksk[ksk_idx];
            
            sum = mod_add(sum, mod_mul(digit, ksk_val));
        }
    }
    
    // For body (lwe_idx == lwe_dim), add c1[0] (constant term)
    if (lwe_idx == lwe_dim) {
        uint c1_idx = batch_idx * ring_dim;  // c1[0]
        sum = mod_add(rlwe_c1[c1_idx], sum);
    }
    
    // Write output
    uint out_idx = batch_idx * (lwe_dim + 1) + lwe_idx;
    lwe[out_idx] = sum;
}

// =============================================================================
// KERNEL 6: POINTWISE POLYNOMIAL MULTIPLICATION (NTT Domain)
// =============================================================================

kernel void pointwise_mul(
    device uint64_t* result             [[buffer(0)]],  // [B, N] output
    constant uint64_t* poly_a           [[buffer(1)]],  // [B, N] input A (NTT domain)
    constant uint64_t* poly_b           [[buffer(2)]],  // [B, N] input B (NTT domain)
    constant uint& ring_dim             [[buffer(3)]],
    constant uint& batch_size           [[buffer(4)]],
    
    uint3 gid [[thread_position_in_grid]]
) {
    uint coeff_idx = gid.x;
    uint batch_idx = gid.y;
    
    if (coeff_idx >= ring_dim || batch_idx >= batch_size) return;
    
    uint idx = batch_idx * ring_dim + coeff_idx;
    result[idx] = mod_mul(poly_a[idx], poly_b[idx]);
}

// =============================================================================
// KERNEL 7: BATCH MODULAR OPERATIONS
// =============================================================================

kernel void batch_mod_add(
    device uint64_t* result             [[buffer(0)]],
    constant uint64_t* a                [[buffer(1)]],
    constant uint64_t* b                [[buffer(2)]],
    constant uint& size                 [[buffer(3)]],
    
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    result[gid] = mod_add(a[gid], b[gid]);
}

kernel void batch_mod_sub(
    device uint64_t* result             [[buffer(0)]],
    constant uint64_t* a                [[buffer(1)]],
    constant uint64_t* b                [[buffer(2)]],
    constant uint& size                 [[buffer(3)]],
    
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    result[gid] = mod_sub(a[gid], b[gid]);
}

// =============================================================================
// KERNEL 8: FULL BLIND ROTATION (Pipeline Orchestrator)
// =============================================================================
//
// This kernel orchestrates the full blind rotation pipeline:
// 1. Initialize accumulator
// 2. Forward NTT
// 3. Loop over n LWE coefficients, applying CMux
// 4. Inverse NTT (optional, for key switching)
//
// Note: In practice, this is typically orchestrated from host code
// with multiple kernel launches. This kernel is for single-dispatch mode.

kernel void blind_rotate_full(
    device uint64_t* acc                [[buffer(0)]],  // [B, 2, N] accumulator
    constant uint64_t* lwe              [[buffer(1)]],  // [B, n+1] LWE ciphertexts
    constant uint64_t* bsk              [[buffer(2)]],  // [n, 2, L, 2, N] bootstrap keys
    constant uint64_t* test_poly        [[buffer(3)]],  // [N] test polynomial
    constant uint64_t* twiddles         [[buffer(4)]],  // [N] NTT twiddles
    constant uint& ring_dim             [[buffer(5)]],  // N
    constant uint& lwe_dim              [[buffer(6)]],  // n
    constant uint& num_levels           [[buffer(7)]],  // L
    constant uint& decomp_log           [[buffer(8)]],
    constant uint& batch_size           [[buffer(9)]],
    
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    // This is a simplified placeholder - full implementation requires
    // multiple threadgroup synchronizations and careful memory management
    
    // In practice, blind rotation is launched as:
    // 1. init_accumulator kernel
    // 2. ntt_forward_fused kernel
    // 3. for i in 0..n: cmux_fused kernel with bsk[i]
    // 4. (optional) ntt_inverse_fused kernel
    
    // The host-side scheduler handles this pipeline
}
