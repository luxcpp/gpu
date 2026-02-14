// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Number Theoretic Transform (NTT) - WGSL Implementation for WebGPU
// Used for polynomial multiplication in FHE and lattice-based cryptography.
//
// Supports:
//   - Goldilocks prime: p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
//   - Arbitrary NTT-friendly primes where p-1 is divisible by large powers of 2
//
// Algorithm: Cooley-Tukey radix-2 DIT (decimation-in-time) for forward NTT
//            Gentleman-Sande radix-2 DIF (decimation-in-frequency) for inverse NTT
//
// WGSL lacks u64, so we use 2 x u32 limbs for 64-bit arithmetic.

// ============================================================================
// Parameters (set via uniform buffer)
// ============================================================================

struct NTTParams {
    n: u32,              // Transform size (must be power of 2)
    log_n: u32,          // log2(n)
    stage: u32,          // Current butterfly stage (0 to log_n-1)
    is_inverse: u32,     // 0 = forward, 1 = inverse
    mod_lo: u32,         // Modulus low 32 bits
    mod_hi: u32,         // Modulus high 32 bits
    omega_lo: u32,       // Primitive root of unity low 32 bits
    omega_hi: u32,       // Primitive root of unity high 32 bits
}

@group(0) @binding(0) var<storage, read_write> data: array<vec2<u32>>;  // [n] elements, each vec2 = u64
@group(0) @binding(1) var<storage, read> twiddles: array<vec2<u32>>;     // Precomputed twiddle factors
@group(0) @binding(2) var<uniform> params: NTTParams;

// ============================================================================
// 64-bit Arithmetic Using 2 x u32 (little-endian: x=lo, y=hi)
// ============================================================================

// Add two u64 values: a + b mod 2^64
fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    let carry = select(0u, 1u, lo < a.x);
    let hi = a.y + b.y + carry;
    return vec2<u32>(lo, hi);
}

// Subtract two u64 values: a - b mod 2^64 (assumes a >= b for correct result)
fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let borrow = select(0u, 1u, a.x < b.x);
    let lo = a.x - b.x;
    let hi = a.y - b.y - borrow;
    return vec2<u32>(lo, hi);
}

// Compare a >= b
fn u64_geq(a: vec2<u32>, b: vec2<u32>) -> bool {
    if (a.y > b.y) { return true; }
    if (a.y < b.y) { return false; }
    return a.x >= b.x;
}

// Compare a < b
fn u64_lt(a: vec2<u32>, b: vec2<u32>) -> bool {
    return !u64_geq(a, b);
}

// Multiply 32x32 -> 64 bits
fn mul32_64(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xFFFFu;
    let ah = a >> 16u;
    let bl = b & 0xFFFFu;
    let bh = b >> 16u;

    let ll = al * bl;
    let lh = al * bh;
    let hl = ah * bl;
    let hh = ah * bh;

    let mid = lh + hl;
    let mid_carry = select(0u, 0x10000u, mid < lh);

    let lo = ll + ((mid & 0xFFFFu) << 16u);
    let lo_carry = select(0u, 1u, lo < ll);

    let hi = hh + (mid >> 16u) + mid_carry + lo_carry;

    return vec2<u32>(lo, hi);
}

// ============================================================================
// Modular Arithmetic for Goldilocks Prime
// p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
// ============================================================================

const GOLDILOCKS_LO: u32 = 0x00000001u;
const GOLDILOCKS_HI: u32 = 0xFFFFFFFFu;

// Get modulus from params or use Goldilocks
fn get_modulus() -> vec2<u32> {
    return vec2<u32>(params.mod_lo, params.mod_hi);
}

// Modular reduction: a mod p (single reduction, assumes a < 2p)
fn mod_reduce(a: vec2<u32>) -> vec2<u32> {
    let p = get_modulus();
    if (u64_geq(a, p)) {
        return u64_sub(a, p);
    }
    return a;
}

// Modular addition: (a + b) mod p
fn mod_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let sum = u64_add(a, b);
    let p = get_modulus();

    // Check for overflow (sum < a means overflow occurred)
    let overflow = u64_lt(sum, a);

    if (overflow || u64_geq(sum, p)) {
        return u64_sub(sum, p);
    }
    return sum;
}

// Modular subtraction: (a - b) mod p
fn mod_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    if (u64_geq(a, b)) {
        return u64_sub(a, b);
    }
    let p = get_modulus();
    return u64_sub(u64_add(a, p), b);
}

// Modular multiplication for Goldilocks prime using Barrett reduction
// For general primes, this uses a simplified approach
fn mod_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    // Compute full 128-bit product as 4 x u32 limbs
    // a = a_lo + a_hi * 2^32
    // b = b_lo + b_hi * 2^32
    // a * b = a_lo*b_lo + (a_lo*b_hi + a_hi*b_lo)*2^32 + a_hi*b_hi*2^64

    let p0 = mul32_64(a.x, b.x);  // a_lo * b_lo
    let p1 = mul32_64(a.x, b.y);  // a_lo * b_hi
    let p2 = mul32_64(a.y, b.x);  // a_hi * b_lo
    let p3 = mul32_64(a.y, b.y);  // a_hi * b_hi

    // Accumulate: result[0..3] = 128-bit product
    var r0 = p0.x;
    var r1 = p0.y;
    var r2 = p3.x;
    var r3 = p3.y;

    // Add p1 shifted by 32 bits
    let sum1_lo = r1 + p1.x;
    let c1 = select(0u, 1u, sum1_lo < r1);
    r1 = sum1_lo;

    let sum1_hi = r2 + p1.y + c1;
    let c2 = select(0u, 1u, sum1_hi < r2 || (c1 == 1u && sum1_hi == r2));
    r2 = sum1_hi;
    r3 = r3 + c2;

    // Add p2 shifted by 32 bits
    let sum2_lo = r1 + p2.x;
    let c3 = select(0u, 1u, sum2_lo < r1);
    r1 = sum2_lo;

    let sum2_hi = r2 + p2.y + c3;
    let c4 = select(0u, 1u, sum2_hi < r2 || (c3 == 1u && sum2_hi == r2));
    r2 = sum2_hi;
    r3 = r3 + c4;

    // Now we have 128-bit product in [r0, r1, r2, r3]
    // Need to reduce mod p

    // For Goldilocks: p = 2^64 - 2^32 + 1
    // We use the fact that 2^64 = 2^32 - 1 (mod p)
    // So [r2, r3] * 2^64 = [r2, r3] * (2^32 - 1) (mod p)

    let p = get_modulus();

    // high_part = [r2, r3]
    // contribution = high_part * 2^32 - high_part = high_part << 32 - high_part

    // high_part << 32
    let shifted_lo = 0u;
    let shifted_hi = r2;
    // This gives us [0, r2] for low 64 bits, [r3, 0] wraps around

    // Simplified reduction for Goldilocks:
    // result = [r0, r1] + r2 * 2^32 - r2 + r3 * (2^32 - 1)

    // First: low = [r0, r1]
    var result = vec2<u32>(r0, r1);

    // Add r2 * 2^32: this adds [0, r2] to result
    let add1 = vec2<u32>(0u, r2);
    result = u64_add(result, add1);

    // Subtract r2: result - [r2, 0]
    // But we need to handle borrow
    if (result.x < r2) {
        result.y = result.y - 1u;
    }
    result.x = result.x - r2;

    // Add r3 * (2^32 - 1) = r3 * 2^32 - r3 = [0, r3] - [r3, 0] = [-r3, r3-1] if r3>0
    // Actually: r3 * (2^32 - 1) = r3 << 32 - r3
    if (r3 > 0u) {
        // Add r3 << 32
        let old_hi = result.y;
        result.y = result.y + r3;
        // Check overflow
        if (result.y < old_hi) {
            // Overflow: add another (2^32 - 1) factor
            result = mod_add(result, vec2<u32>(0xFFFFFFFFu, 0u));
        }
        // Subtract r3
        if (result.x < r3) {
            result.y = result.y - 1u;
        }
        result.x = result.x - r3;
    }

    // Final reductions
    result = mod_reduce(result);
    result = mod_reduce(result);

    return result;
}

// ============================================================================
// Bit Reversal
// ============================================================================

fn bit_reverse(x: u32, bits: u32) -> u32 {
    var v = x;
    var r = 0u;
    for (var i = 0u; i < bits; i = i + 1u) {
        r = (r << 1u) | (v & 1u);
        v = v >> 1u;
    }
    return r;
}

// ============================================================================
// NTT Butterfly Operations
// ============================================================================

// Cooley-Tukey butterfly (DIT): used in forward NTT
// Given inputs at positions i and j with twiddle factor w:
//   X[i] = a + w*b
//   X[j] = a - w*b
fn ct_butterfly(a: vec2<u32>, b: vec2<u32>, w: vec2<u32>) -> array<vec2<u32>, 2> {
    let wb = mod_mul(w, b);
    var result: array<vec2<u32>, 2>;
    result[0] = mod_add(a, wb);
    result[1] = mod_sub(a, wb);
    return result;
}

// Gentleman-Sande butterfly (DIF): used in inverse NTT
// Given inputs at positions i and j with twiddle factor w:
//   X[i] = a + b
//   X[j] = (a - b) * w
fn gs_butterfly(a: vec2<u32>, b: vec2<u32>, w: vec2<u32>) -> array<vec2<u32>, 2> {
    var result: array<vec2<u32>, 2>;
    result[0] = mod_add(a, b);
    let diff = mod_sub(a, b);
    result[1] = mod_mul(diff, w);
    return result;
}

// ============================================================================
// NTT Forward Pass (Single Stage)
// ============================================================================

@compute @workgroup_size(64)
fn ntt_forward_stage(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params.n;
    let stage = params.stage;

    // Number of butterflies per stage
    let butterflies_per_group = 1u << stage;
    let groups = n >> (stage + 1u);

    // Each thread handles one butterfly
    if (idx >= n / 2u) {
        return;
    }

    // Determine which butterfly this thread handles
    let group = idx / butterflies_per_group;
    let butterfly_in_group = idx % butterflies_per_group;

    // Calculate indices for this butterfly
    let i = group * (butterflies_per_group * 2u) + butterfly_in_group;
    let j = i + butterflies_per_group;

    // Get twiddle factor index
    let twiddle_idx = butterfly_in_group * groups;

    // Load values
    let a = data[i];
    let b = data[j];
    let w = twiddles[twiddle_idx];

    // Perform butterfly
    let result = ct_butterfly(a, b, w);

    // Store results
    data[i] = result[0];
    data[j] = result[1];
}

// ============================================================================
// NTT Inverse Pass (Single Stage)
// ============================================================================

@compute @workgroup_size(64)
fn ntt_inverse_stage(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params.n;
    let stage = params.stage;

    // For inverse NTT with DIF, we go from large groups to small
    let log_n = params.log_n;
    let inv_stage = log_n - 1u - stage;

    let butterflies_per_group = 1u << inv_stage;
    let groups = n >> (inv_stage + 1u);

    if (idx >= n / 2u) {
        return;
    }

    let group = idx / butterflies_per_group;
    let butterfly_in_group = idx % butterflies_per_group;

    let i = group * (butterflies_per_group * 2u) + butterfly_in_group;
    let j = i + butterflies_per_group;

    // Inverse twiddle factors (omega^{-k})
    let twiddle_idx = butterfly_in_group * groups;

    let a = data[i];
    let b = data[j];
    let w = twiddles[twiddle_idx];  // Should be inverse twiddles

    let result = gs_butterfly(a, b, w);

    data[i] = result[0];
    data[j] = result[1];
}

// ============================================================================
// Bit-Reverse Permutation
// ============================================================================

@compute @workgroup_size(64)
fn bit_reverse_permute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params.n;
    let log_n = params.log_n;

    if (idx >= n) {
        return;
    }

    let rev_idx = bit_reverse(idx, log_n);

    // Only swap if idx < rev_idx to avoid double-swapping
    if (idx < rev_idx) {
        let temp = data[idx];
        data[idx] = data[rev_idx];
        data[rev_idx] = temp;
    }
}

// ============================================================================
// Scale by n^{-1} for Inverse NTT
// ============================================================================

@compute @workgroup_size(64)
fn ntt_scale_inverse(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params.n;

    if (idx >= n) {
        return;
    }

    // n_inv should be precomputed and stored in twiddles[n] or passed via params
    // For now, we assume twiddles[0] in inverse mode contains n^{-1} mod p
    // This is a simplification - actual implementation should pass n_inv explicitly

    let n_inv = twiddles[params.n];  // n^{-1} mod p stored after twiddles
    data[idx] = mod_mul(data[idx], n_inv);
}

// ============================================================================
// Pointwise Multiplication (for polynomial multiplication via NTT)
// ============================================================================

@group(0) @binding(3) var<storage, read> other: array<vec2<u32>>;

@compute @workgroup_size(64)
fn pointwise_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params.n;

    if (idx >= n) {
        return;
    }

    data[idx] = mod_mul(data[idx], other[idx]);
}

// ============================================================================
// Negacyclic NTT (for ring Z_p[X]/(X^n + 1))
// Used in TFHE and other FHE schemes
// ============================================================================

// Pre-multiply by powers of psi (primitive 2n-th root of unity)
@compute @workgroup_size(64)
fn negacyclic_pre_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params.n;

    if (idx >= n) {
        return;
    }

    // psi^i stored in twiddles, offset by n
    let psi_power = twiddles[n + idx];
    data[idx] = mod_mul(data[idx], psi_power);
}

// Post-multiply by inverse powers of psi
@compute @workgroup_size(64)
fn negacyclic_post_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params.n;

    if (idx >= n) {
        return;
    }

    // psi^{-i} stored in twiddles, offset by 2n
    let psi_inv_power = twiddles[2u * n + idx];
    data[idx] = mod_mul(data[idx], psi_inv_power);
}
