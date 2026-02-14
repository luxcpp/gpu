// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Fully Homomorphic Encryption (FHE) Primitives - WGSL Implementation
// Supports TFHE-style operations for lattice-based cryptography.
//
// Operations:
//   - Blind rotation (core of TFHE bootstrap)
//   - Sample extraction
//   - Key switching
//   - External product (GGSW x GLWE)
//
// WGSL lacks u64, so we use 2 x u32 limbs for 64-bit arithmetic.

// ============================================================================
// Types and Constants
// ============================================================================

struct FHEParams {
    n_lwe: u32,          // LWE dimension
    N: u32,              // GLWE polynomial degree (power of 2)
    k: u32,              // GLWE dimension
    l: u32,              // Decomposition levels
    base_log: u32,       // Base log for decomposition
    mod_lo: u32,         // Modulus low 32 bits
    mod_hi: u32,         // Modulus high 32 bits
    stage: u32,          // Current computation stage
}

// 64-bit value as 2 x u32 (little-endian)
alias U64 = vec2<u32>;

// ============================================================================
// Buffer Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read_write> accumulator: array<U64>;  // GLWE accumulator [(k+1)*N]
@group(0) @binding(1) var<storage, read> bsk: array<U64>;                 // Bootstrapping key
@group(0) @binding(2) var<storage, read> lwe_input: array<U64>;           // LWE ciphertext [n_lwe+1]
@group(0) @binding(3) var<uniform> params: FHEParams;

// ============================================================================
// 64-bit Arithmetic
// ============================================================================

fn u64_add(a: U64, b: U64) -> U64 {
    let lo = a.x + b.x;
    let carry = select(0u, 1u, lo < a.x);
    let hi = a.y + b.y + carry;
    return U64(lo, hi);
}

fn u64_sub(a: U64, b: U64) -> U64 {
    let borrow = select(0u, 1u, a.x < b.x);
    let lo = a.x - b.x;
    let hi = a.y - b.y - borrow;
    return U64(lo, hi);
}

fn u64_geq(a: U64, b: U64) -> bool {
    if (a.y > b.y) { return true; }
    if (a.y < b.y) { return false; }
    return a.x >= b.x;
}

fn mul32_64(a: u32, b: u32) -> U64 {
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

    return U64(lo, hi);
}

// ============================================================================
// Modular Arithmetic
// ============================================================================

fn get_mod() -> U64 {
    return U64(params.mod_lo, params.mod_hi);
}

fn mod_reduce(a: U64) -> U64 {
    let p = get_mod();
    if (u64_geq(a, p)) {
        return u64_sub(a, p);
    }
    return a;
}

fn mod_add(a: U64, b: U64) -> U64 {
    let sum = u64_add(a, b);
    let p = get_mod();
    let overflow = sum.y < a.y || (sum.y == a.y && sum.x < a.x);
    if (overflow || u64_geq(sum, p)) {
        return u64_sub(sum, p);
    }
    return sum;
}

fn mod_sub(a: U64, b: U64) -> U64 {
    if (u64_geq(a, b)) {
        return u64_sub(a, b);
    }
    let p = get_mod();
    return u64_sub(u64_add(a, p), b);
}

// Simplified modular multiplication (for demonstration)
// Full implementation requires Montgomery or Barrett reduction
fn mod_mul(a: U64, b: U64) -> U64 {
    // For now, use schoolbook with reduction
    // This is a placeholder - real impl needs proper 128-bit reduction

    let p0 = mul32_64(a.x, b.x);
    let p1 = mul32_64(a.x, b.y);
    let p2 = mul32_64(a.y, b.x);
    let p3 = mul32_64(a.y, b.y);

    var r0 = p0.x;
    var r1 = p0.y;
    var r2 = p3.x;
    var r3 = p3.y;

    // Add p1 << 32
    let s1 = r1 + p1.x;
    let c1 = select(0u, 1u, s1 < r1);
    r1 = s1;
    let s2 = r2 + p1.y + c1;
    let c2 = select(0u, 1u, s2 < r2);
    r2 = s2;
    r3 = r3 + c2;

    // Add p2 << 32
    let s3 = r1 + p2.x;
    let c3 = select(0u, 1u, s3 < r1);
    r1 = s3;
    let s4 = r2 + p2.y + c3;
    let c4 = select(0u, 1u, s4 < r2);
    r2 = s4;
    r3 = r3 + c4;

    // Simplified reduction for Goldilocks
    var result = U64(r0, r1);
    let p = get_mod();

    // Reduce high part using p = 2^64 - 2^32 + 1
    // 2^64 = 2^32 - 1 mod p
    if (r2 > 0u || r3 > 0u) {
        let contrib = U64(0u, r2);
        result = u64_add(result, contrib);
        if (result.x >= r2) {
            result.x = result.x - r2;
        } else {
            result.y = result.y - 1u;
            result.x = result.x - r2;
        }
    }

    result = mod_reduce(result);
    return result;
}

// ============================================================================
// Negacyclic Polynomial Rotation
// Rotates polynomial by 'rot' positions in Z_q[X]/(X^N + 1)
// ============================================================================

fn negacyclic_rotate_coeff(poly_base: u32, coeff_idx: u32, rot: u32, N: u32) -> U64 {
    let src_idx = (coeff_idx + rot) % (2u * N);

    if (src_idx < N) {
        return accumulator[poly_base + src_idx];
    } else {
        // Wrap around with negation (X^N = -1)
        let p = get_mod();
        let val = accumulator[poly_base + src_idx - N];
        return mod_sub(p, val);
    }
}

// ============================================================================
// Gadget Decomposition
// Decomposes element a into l digits base 2^base_log
// ============================================================================

fn gadget_decompose(a: U64, level: u32) -> u32 {
    let base_log = params.base_log;
    let shift = (params.l - 1u - level) * base_log;
    let mask = (1u << base_log) - 1u;

    // Extract digit from appropriate position
    if (shift < 32u) {
        return (a.x >> shift) & mask;
    } else {
        return (a.y >> (shift - 32u)) & mask;
    }
}

// ============================================================================
// Blind Rotation - Core of TFHE Bootstrap
// Rotates accumulator by encrypted LWE values using BSK
// ============================================================================

@compute @workgroup_size(64)
fn blind_rotate_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let N = params.N;
    let k = params.k;
    let glwe_size = (k + 1u) * N;

    if (idx >= glwe_size) {
        return;
    }

    let current_lwe_idx = params.stage;
    if (current_lwe_idx >= params.n_lwe) {
        return;
    }

    // Get LWE coefficient a_i
    let a_i = lwe_input[current_lwe_idx];

    // Compute rotation exponent: round(a_i * 2N / q)
    // Simplified: we assume q is power of 2 or use approximation
    let N2 = N * 2u;

    // For 64-bit modulus, extract rotation amount
    // rot = (a_i * 2N) / q approximately
    // This is simplified - full implementation needs careful rounding
    let rot = (a_i.y >> (32u - params.base_log)) % N2;

    if (rot == 0u) {
        return;  // No rotation needed for this coefficient
    }

    // Determine polynomial and coefficient within GLWE
    let poly_idx = idx / N;
    let coeff_idx = idx % N;

    // Compute rotated value
    let rotated = negacyclic_rotate_coeff(poly_idx * N, coeff_idx, rot, N);

    // External product with BSK[current_lwe_idx] would go here
    // Simplified: just apply rotation difference
    let current = accumulator[idx];
    let diff = mod_sub(current, rotated);

    // Update accumulator (simplified CMux operation)
    // Full implementation requires external product with GGSW
    accumulator[idx] = mod_add(current, diff);
}

// ============================================================================
// Sample Extraction
// Extracts LWE ciphertext from GLWE accumulator
// ============================================================================

@group(0) @binding(4) var<storage, read_write> lwe_output: array<U64>;

@compute @workgroup_size(64)
fn sample_extract(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let N = params.N;
    let k = params.k;
    let lwe_dim = k * N;

    // Output LWE has dimension k*N
    if (idx > lwe_dim) {
        return;
    }

    let p = get_mod();

    if (idx == lwe_dim) {
        // Body: constant term of last GLWE polynomial
        lwe_output[idx] = accumulator[k * N];
    } else {
        // Mask coefficients with negacyclic unrolling
        let block = idx / N;
        let pos = idx % N;

        if (pos == 0u) {
            lwe_output[idx] = accumulator[block * N];
        } else {
            // Negate due to negacyclic ring
            let val = accumulator[block * N + (N - pos)];
            lwe_output[idx] = mod_sub(p, val);
        }
    }
}

// ============================================================================
// Key Switching
// Changes LWE secret key using key switching key
// ============================================================================

@group(0) @binding(5) var<storage, read> ksk: array<U64>;

struct KSParams {
    n_in: u32,
    n_out: u32,
    l: u32,
    base_log: u32,
    mod_lo: u32,
    mod_hi: u32,
    pad0: u32,
    pad1: u32,
}

@group(1) @binding(0) var<storage, read> lwe_ks_input: array<U64>;
@group(1) @binding(1) var<storage, read_write> lwe_ks_output: array<U64>;
@group(1) @binding(2) var<uniform> ks_params: KSParams;

@compute @workgroup_size(64)
fn keyswitch(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    let n_in = ks_params.n_in;
    let n_out = ks_params.n_out;
    let l = ks_params.l;
    let base_log = ks_params.base_log;
    let base = 1u << base_log;
    let mask = base - 1u;

    if (out_idx > n_out) {
        return;
    }

    var result = U64(0u, 0u);

    // Copy body if this is the last element
    if (out_idx == n_out) {
        result = lwe_ks_input[n_in];
    }

    // Apply key switching
    for (var i = 0u; i < n_in; i = i + 1u) {
        let a_i = lwe_ks_input[i];

        for (var j = 0u; j < l; j = j + 1u) {
            // Decompose a_i
            let shift = (l - 1u - j) * base_log;
            var digit: u32;
            if (shift < 32u) {
                digit = (a_i.x >> shift) & mask;
            } else {
                digit = (a_i.y >> (shift - 32u)) & mask;
            }

            if (digit != 0u) {
                // Subtract KSK entry
                let ksk_offset = (i * l * base + j * base + digit) * (n_out + 1u);
                let ksk_val = ksk[ksk_offset + out_idx];
                result = mod_sub(result, ksk_val);
            }
        }
    }

    lwe_ks_output[out_idx] = result;
}

// ============================================================================
// LWE Addition/Subtraction (for bootstrapping composition)
// ============================================================================

@compute @workgroup_size(64)
fn lwe_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params.n_lwe + 1u;

    if (idx >= n) {
        return;
    }

    let a = lwe_input[idx];
    let b = lwe_output[idx];
    lwe_output[idx] = mod_add(a, b);
}

@compute @workgroup_size(64)
fn lwe_sub(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params.n_lwe + 1u;

    if (idx >= n) {
        return;
    }

    let a = lwe_input[idx];
    let b = lwe_output[idx];
    lwe_output[idx] = mod_sub(a, b);
}

// ============================================================================
// Initialize Accumulator with Test Polynomial
// ============================================================================

@group(0) @binding(6) var<storage, read> test_poly: array<U64>;

@compute @workgroup_size(64)
fn init_accumulator(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let N = params.N;
    let k = params.k;
    let glwe_size = (k + 1u) * N;

    if (idx >= glwe_size) {
        return;
    }

    // Initialize: set mask polynomials to 0, body to rotated test polynomial
    let poly_idx = idx / N;
    let coeff_idx = idx % N;

    if (poly_idx < k) {
        // Mask polynomials: zero
        accumulator[idx] = U64(0u, 0u);
    } else {
        // Body polynomial: test polynomial rotated by -b (handled separately)
        accumulator[idx] = test_poly[coeff_idx];
    }
}

// ============================================================================
// External Product (GGSW x GLWE -> GLWE)
// Core operation for blind rotation
// ============================================================================

@group(0) @binding(7) var<storage, read> ggsw: array<U64>;  // GGSW ciphertext

@compute @workgroup_size(64)
fn external_product(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let N = params.N;
    let k = params.k;
    let l = params.l;
    let glwe_size = (k + 1u) * N;

    if (idx >= glwe_size) {
        return;
    }

    let poly_idx = idx / N;
    let coeff_idx = idx % N;

    var result = U64(0u, 0u);

    // For each GLWE polynomial in accumulator
    for (var i = 0u; i <= k; i = i + 1u) {
        // Decompose polynomial coefficient
        let acc_val = accumulator[i * N + coeff_idx];

        // For each decomposition level
        for (var j = 0u; j < l; j = j + 1u) {
            let digit = gadget_decompose(acc_val, j);

            if (digit != 0u) {
                // Get corresponding GGSW row
                // GGSW layout: [l rows per polynomial][k+1 GLWE polys per row]
                let ggsw_row = i * l + j;
                let ggsw_offset = ggsw_row * glwe_size + poly_idx * N + coeff_idx;

                // Multiply digit by GGSW element (simplified - should be polynomial mul)
                let ggsw_val = ggsw[ggsw_offset];
                let prod = mod_mul(U64(digit, 0u), ggsw_val);
                result = mod_add(result, prod);
            }
        }
    }

    accumulator[idx] = result;
}
