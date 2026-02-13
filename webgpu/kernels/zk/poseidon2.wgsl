// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Poseidon2 Hash - WGSL Implementation for WebGPU
// BN254 scalar field arithmetic using 8x u32 limbs (WGSL lacks u64)
//
// Field: Fr of BN254 (alt_bn128)
// Modulus p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// Montgomery form: R = 2^256 mod p
// R^2 mod p precomputed for toMontgomery

// BN254 scalar field modulus (little-endian limbs)
// p = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
const BN254_MOD_0: u32 = 0xf0000001u;
const BN254_MOD_1: u32 = 0x43e1f593u;
const BN254_MOD_2: u32 = 0x79b97091u;
const BN254_MOD_3: u32 = 0x2833e848u;
const BN254_MOD_4: u32 = 0x8181585du;
const BN254_MOD_5: u32 = 0xb85045b6u;
const BN254_MOD_6: u32 = 0xe131a029u;
const BN254_MOD_7: u32 = 0x30644e72u;

// Montgomery constant: -p^(-1) mod 2^32
// Used for Montgomery reduction
const MONT_INV: u32 = 0xefffffff;

// R^2 mod p (for converting to Montgomery form)
// R = 2^256, R^2 mod p precomputed
// Must match C++ BN254_R2: {0xf32cfc5b538afa89, 0xb5e71911d44501fb, 0x47ab1eff0a417ff6, 0x06d89f71cab8351f}
// Split into 32-bit limbs (little-endian): each 64-bit value splits as [lo32, hi32]
const R2_MOD_0: u32 = 0x538afa89u;  // low32 of 0xf32cfc5b538afa89
const R2_MOD_1: u32 = 0xf32cfc5bu;  // high32 of 0xf32cfc5b538afa89
const R2_MOD_2: u32 = 0xd44501fbu;  // low32 of 0xb5e71911d44501fb
const R2_MOD_3: u32 = 0xb5e71911u;  // high32 of 0xb5e71911d44501fb
const R2_MOD_4: u32 = 0x0a417ff6u;  // low32 of 0x47ab1eff0a417ff6
const R2_MOD_5: u32 = 0x47ab1effu;  // high32 of 0x47ab1eff0a417ff6
const R2_MOD_6: u32 = 0xcab8351fu;  // low32 of 0x06d89f71cab8351f
const R2_MOD_7: u32 = 0x06d89f71u;  // high32 of 0x06d89f71cab8351f

// Poseidon2 parameters for t=3 on BN254
// Must match C++ zk_ops.cpp: RF=8, RP=56, t=3
const POSEIDON2_RF: u32 = 8u;   // Full rounds (4 before partial, 4 after)
const POSEIDON2_RP: u32 = 56u;  // Partial rounds
const POSEIDON2_T: u32 = 3u;    // State width

// Round constant count: (RF/2)*t + RP + (RF/2)*t = 4*3 + 56 + 4*3 = 80
// Internal diagonal D = [1, 1, 2] (matches C++ INTERNAL_DIAG)

// Fr256 as 8 x u32 limbs (little-endian)
struct Fr256 {
    limbs: array<u32, 8>,
}

// Poseidon2 state
struct Poseidon2State {
    s0: Fr256,
    s1: Fr256,
    s2: Fr256,
}

// Input/output buffers
@group(0) @binding(0) var<storage, read> left_inputs: array<Fr256>;
@group(0) @binding(1) var<storage, read> right_inputs: array<Fr256>;
@group(0) @binding(2) var<storage, read_write> outputs: array<Fr256>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // [num_hashes, 0, 0, 0]

// Round constants stored in storage buffer
@group(0) @binding(4) var<storage, read> round_constants: array<Fr256>;

// ============================================================================
// 256-bit Arithmetic Without u64
// ============================================================================

// Add with carry: returns (sum, carry) packed as sum in result, carry in return
// For a + b + cin, detects overflow without u64
fn adc(a: u32, b: u32, cin: u32) -> vec2<u32> {
    // First add a + b
    let sum1 = a + b;
    let carry1 = select(0u, 1u, sum1 < a);

    // Then add carry in
    let sum2 = sum1 + cin;
    let carry2 = select(0u, 1u, sum2 < sum1);

    return vec2<u32>(sum2, carry1 + carry2);
}

// Subtract with borrow: returns (diff, borrow)
fn sbb(a: u32, b: u32, bin: u32) -> vec2<u32> {
    // First subtract b from a
    let diff1 = a - b;
    let borrow1 = select(0u, 1u, a < b);

    // Then subtract borrow in
    let diff2 = diff1 - bin;
    let borrow2 = select(0u, 1u, diff1 < bin);

    return vec2<u32>(diff2, borrow1 + borrow2);
}

// Multiply 32x32 -> 64 bits, return (lo, hi)
fn mul32(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xFFFFu;
    let ah = a >> 16u;
    let bl = b & 0xFFFFu;
    let bh = b >> 16u;

    let ll = al * bl;
    let lh = al * bh;
    let hl = ah * bl;
    let hh = ah * bh;

    // Combine: ll + (lh + hl) << 16 + hh << 32
    let mid = lh + hl;
    let mid_carry = select(0u, 0x10000u, mid < lh); // Carry from mid overflow

    let lo1 = ll + ((mid & 0xFFFFu) << 16u);
    let lo_carry = select(0u, 1u, lo1 < ll);

    let hi = hh + (mid >> 16u) + mid_carry + lo_carry;

    return vec2<u32>(lo1, hi);
}

// Multiply-add: a*b + c + d -> (lo, hi)
fn mac(a: u32, b: u32, c: u32, d: u32) -> vec2<u32> {
    let prod = mul32(a, b);

    let sum1 = prod.x + c;
    let carry1 = select(0u, 1u, sum1 < prod.x);

    let sum2 = sum1 + d;
    let carry2 = select(0u, 1u, sum2 < sum1);

    return vec2<u32>(sum2, prod.y + carry1 + carry2);
}

// ============================================================================
// Field Operations
// ============================================================================

// Compare a >= b (returns true if a >= b)
fn fr_geq(a: Fr256, b_arr: array<u32, 8>) -> bool {
    for (var i = 7i; i >= 0i; i = i - 1i) {
        let idx = u32(i);
        if (a.limbs[idx] > b_arr[idx]) {
            return true;
        } else if (a.limbs[idx] < b_arr[idx]) {
            return false;
        }
    }
    return true; // Equal
}

// Subtract modulus: a - p
fn fr_sub_mod(a: Fr256) -> Fr256 {
    var mod_arr: array<u32, 8>;
    mod_arr[0] = BN254_MOD_0;
    mod_arr[1] = BN254_MOD_1;
    mod_arr[2] = BN254_MOD_2;
    mod_arr[3] = BN254_MOD_3;
    mod_arr[4] = BN254_MOD_4;
    mod_arr[5] = BN254_MOD_5;
    mod_arr[6] = BN254_MOD_6;
    mod_arr[7] = BN254_MOD_7;

    var result: Fr256;
    var borrow: u32 = 0u;

    for (var i = 0u; i < 8u; i = i + 1u) {
        let sb = sbb(a.limbs[i], mod_arr[i], borrow);
        result.limbs[i] = sb.x;
        borrow = sb.y;
    }

    return result;
}

// Reduce if >= modulus (single reduction)
fn fr_reduce(a: Fr256) -> Fr256 {
    var mod_arr: array<u32, 8>;
    mod_arr[0] = BN254_MOD_0;
    mod_arr[1] = BN254_MOD_1;
    mod_arr[2] = BN254_MOD_2;
    mod_arr[3] = BN254_MOD_3;
    mod_arr[4] = BN254_MOD_4;
    mod_arr[5] = BN254_MOD_5;
    mod_arr[6] = BN254_MOD_6;
    mod_arr[7] = BN254_MOD_7;

    if (fr_geq(a, mod_arr)) {
        return fr_sub_mod(a);
    }
    return a;
}

// Add two Fr256 with modular reduction
fn fr_add(a: Fr256, b: Fr256) -> Fr256 {
    var result: Fr256;
    var carry: u32 = 0u;

    for (var i = 0u; i < 8u; i = i + 1u) {
        let ac = adc(a.limbs[i], b.limbs[i], carry);
        result.limbs[i] = ac.x;
        carry = ac.y;
    }

    // If carry out or result >= mod, subtract modulus
    var mod_arr: array<u32, 8>;
    mod_arr[0] = BN254_MOD_0;
    mod_arr[1] = BN254_MOD_1;
    mod_arr[2] = BN254_MOD_2;
    mod_arr[3] = BN254_MOD_3;
    mod_arr[4] = BN254_MOD_4;
    mod_arr[5] = BN254_MOD_5;
    mod_arr[6] = BN254_MOD_6;
    mod_arr[7] = BN254_MOD_7;

    if (carry > 0u || fr_geq(result, mod_arr)) {
        return fr_sub_mod(result);
    }

    return result;
}

// Subtract two Fr256: a - b mod p
fn fr_sub(a: Fr256, b: Fr256) -> Fr256 {
    var result: Fr256;
    var borrow: u32 = 0u;

    for (var i = 0u; i < 8u; i = i + 1u) {
        let sb = sbb(a.limbs[i], b.limbs[i], borrow);
        result.limbs[i] = sb.x;
        borrow = sb.y;
    }

    // If underflow, add modulus
    if (borrow > 0u) {
        var carry: u32 = 0u;
        var mod_arr: array<u32, 8>;
        mod_arr[0] = BN254_MOD_0;
        mod_arr[1] = BN254_MOD_1;
        mod_arr[2] = BN254_MOD_2;
        mod_arr[3] = BN254_MOD_3;
        mod_arr[4] = BN254_MOD_4;
        mod_arr[5] = BN254_MOD_5;
        mod_arr[6] = BN254_MOD_6;
        mod_arr[7] = BN254_MOD_7;

        for (var i = 0u; i < 8u; i = i + 1u) {
            let ac = adc(result.limbs[i], mod_arr[i], carry);
            result.limbs[i] = ac.x;
            carry = ac.y;
        }
    }

    return result;
}

// ============================================================================
// Montgomery Multiplication
// ============================================================================

// Montgomery reduction: given T (16 limbs), compute T * R^(-1) mod p
fn mont_reduce(t: array<u32, 16>) -> Fr256 {
    var tmp: array<u32, 17>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        tmp[i] = t[i];
    }
    tmp[16] = 0u;

    var mod_arr: array<u32, 8>;
    mod_arr[0] = BN254_MOD_0;
    mod_arr[1] = BN254_MOD_1;
    mod_arr[2] = BN254_MOD_2;
    mod_arr[3] = BN254_MOD_3;
    mod_arr[4] = BN254_MOD_4;
    mod_arr[5] = BN254_MOD_5;
    mod_arr[6] = BN254_MOD_6;
    mod_arr[7] = BN254_MOD_7;

    // Montgomery reduction loop
    for (var i = 0u; i < 8u; i = i + 1u) {
        // m = tmp[i] * MONT_INV mod 2^32
        let m = tmp[i] * MONT_INV;

        // tmp += m * modulus << (i * 32)
        var carry: u32 = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let mc = mac(m, mod_arr[j], tmp[i + j], carry);
            tmp[i + j] = mc.x;
            carry = mc.y;
        }

        // Propagate carry
        for (var j = i + 8u; j < 17u; j = j + 1u) {
            let ac = adc(tmp[j], carry, 0u);
            tmp[j] = ac.x;
            carry = ac.y;
            if (carry == 0u) {
                break;
            }
        }
    }

    // Result is in tmp[8..16]
    var result: Fr256;
    for (var i = 0u; i < 8u; i = i + 1u) {
        result.limbs[i] = tmp[i + 8u];
    }

    // Final reduction if needed
    return fr_reduce(result);
}

// Montgomery multiplication: a * b * R^(-1) mod p
// Assumes a, b are in Montgomery form
fn fr_mul_mont(a: Fr256, b: Fr256) -> Fr256 {
    var product: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        product[i] = 0u;
    }

    // Schoolbook multiplication
    for (var i = 0u; i < 8u; i = i + 1u) {
        var carry: u32 = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let mc = mac(a.limbs[i], b.limbs[j], product[i + j], carry);
            product[i + j] = mc.x;
            carry = mc.y;
        }
        product[i + 8u] = carry;
    }

    return mont_reduce(product);
}

// Field multiplication using Montgomery form
// For Poseidon2, all field elements are kept in Montgomery form.
// a_mont * b_mont via Montgomery = (a*R) * (b*R) * R^-1 mod p = a*b*R mod p
// This is the Montgomery product, result stays in Montgomery form.
fn fr_mul(a: Fr256, b: Fr256) -> Fr256 {
    var product: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        product[i] = 0u;
    }

    // Schoolbook multiplication: a * b -> 512-bit product
    for (var i = 0u; i < 8u; i = i + 1u) {
        var carry: u32 = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let mc = mac(a.limbs[i], b.limbs[j], product[i + j], carry);
            product[i + j] = mc.x;
            carry = mc.y;
        }
        product[i + 8u] = carry;
    }

    // Montgomery reduction: product * R^-1 mod p
    return mont_reduce(product);
}

// Square Fr256
fn fr_square(a: Fr256) -> Fr256 {
    return fr_mul(a, a);
}

// Convert to Montgomery form: a -> a*R mod p
// Computed as a * R^2 * R^-1 = a * R mod p
fn to_montgomery(a: Fr256) -> Fr256 {
    var r2: Fr256;
    r2.limbs[0] = R2_MOD_0;
    r2.limbs[1] = R2_MOD_1;
    r2.limbs[2] = R2_MOD_2;
    r2.limbs[3] = R2_MOD_3;
    r2.limbs[4] = R2_MOD_4;
    r2.limbs[5] = R2_MOD_5;
    r2.limbs[6] = R2_MOD_6;
    r2.limbs[7] = R2_MOD_7;
    return fr_mul(a, r2);
}

// Convert from Montgomery form: a_mont -> a_mont * R^-1 mod p = a
fn from_montgomery(a: Fr256) -> Fr256 {
    var one: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        one[i] = 0u;
    }
    // Copy a into lower 8 limbs, upper 8 are zero
    for (var i = 0u; i < 8u; i = i + 1u) {
        one[i] = a.limbs[i];
    }
    // mont_reduce computes T * R^-1 mod p
    return mont_reduce(one);
}

// ============================================================================
// Poseidon2 Hash
// ============================================================================

// Compute x^5 (S-box for Poseidon2)
fn sbox(x: Fr256) -> Fr256 {
    let x2 = fr_square(x);
    let x4 = fr_square(x2);
    return fr_mul(x4, x);
}

// Apply Poseidon2 external matrix M_E to state
// M_E for t=3: [[2,1,1],[1,2,1],[1,1,2]]
// Optimized form: s_i' = s_i + sum(all s_j)
fn apply_external_matrix(state: ptr<function, Poseidon2State>) {
    let sum = fr_add(fr_add((*state).s0, (*state).s1), (*state).s2);
    (*state).s0 = fr_add((*state).s0, sum);
    (*state).s1 = fr_add((*state).s1, sum);
    (*state).s2 = fr_add((*state).s2, sum);
}

// Apply Poseidon2 internal matrix M_I for t=3
// M_I = diag(d_0, d_1, d_2) + J where J is all-ones matrix
// For BN254 t=3: D = [1, 1, 2] (matches C++ INTERNAL_DIAG)
// s[i] = d[i] * s[i] + sum
// Result: s0 = 1*s0 + sum, s1 = 1*s1 + sum, s2 = 2*s2 + sum
fn apply_internal_matrix(state: ptr<function, Poseidon2State>) {
    let sum = fr_add(fr_add((*state).s0, (*state).s1), (*state).s2);

    // d[0] = 1: s0 = s0 + sum
    (*state).s0 = fr_add((*state).s0, sum);
    // d[1] = 1: s1 = s1 + sum
    (*state).s1 = fr_add((*state).s1, sum);
    // d[2] = 2: s2 = 2*s2 + sum
    let s2_doubled = fr_add((*state).s2, (*state).s2);
    (*state).s2 = fr_add(s2_doubled, sum);
}

// Full round: S-box on all state elements, then external matrix
fn full_round(state: ptr<function, Poseidon2State>, rc_offset: u32) {
    // Add round constants
    (*state).s0 = fr_add((*state).s0, round_constants[rc_offset]);
    (*state).s1 = fr_add((*state).s1, round_constants[rc_offset + 1u]);
    (*state).s2 = fr_add((*state).s2, round_constants[rc_offset + 2u]);

    // S-box on all elements
    (*state).s0 = sbox((*state).s0);
    (*state).s1 = sbox((*state).s1);
    (*state).s2 = sbox((*state).s2);

    // External matrix
    apply_external_matrix(state);
}

// Partial round: S-box only on s0, then internal matrix
fn partial_round(state: ptr<function, Poseidon2State>, rc_offset: u32) {
    // Add round constant to s0 only
    (*state).s0 = fr_add((*state).s0, round_constants[rc_offset]);

    // S-box only on s0
    (*state).s0 = sbox((*state).s0);

    // Internal matrix
    apply_internal_matrix(state);
}

// Poseidon2 hash: H(left, right) -> output
// CONTRACT:
// - Inputs (left, right) must be in Montgomery form
// - Round constants must be in Montgomery form
// - Output is in Montgomery form
// - Caller must convert to/from Montgomery form as needed
fn poseidon2_hash(left: Fr256, right: Fr256) -> Fr256 {
    var state: Poseidon2State;

    // Initialize state: [0, left, right] (domain separation)
    // Note: 0 in standard form = 0 in Montgomery form
    for (var i = 0u; i < 8u; i = i + 1u) {
        state.s0.limbs[i] = 0u;
    }
    state.s1 = left;
    state.s2 = right;

    var rc_idx = 0u;

    // First half of full rounds (RF/2 = 4)
    for (var r = 0u; r < POSEIDON2_RF / 2u; r = r + 1u) {
        full_round(&state, rc_idx);
        rc_idx = rc_idx + 3u;
    }

    // Partial rounds (RP = 56)
    for (var r = 0u; r < POSEIDON2_RP; r = r + 1u) {
        partial_round(&state, rc_idx);
        rc_idx = rc_idx + 1u;
    }

    // Second half of full rounds (RF/2 = 4)
    for (var r = 0u; r < POSEIDON2_RF / 2u; r = r + 1u) {
        full_round(&state, rc_idx);
        rc_idx = rc_idx + 3u;
    }

    // Output is s1 (standard Poseidon2 output selection)
    return state.s1;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_hashes = params.x;

    if (idx >= num_hashes) {
        return;
    }

    let left = left_inputs[idx];
    let right = right_inputs[idx];

    outputs[idx] = poseidon2_hash(left, right);
}
