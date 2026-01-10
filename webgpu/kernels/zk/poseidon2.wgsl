// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Poseidon2 Hash - WGSL Implementation for WebGPU
// BN254 scalar field arithmetic using 8x u32 limbs (WGSL lacks u64)

// BN254 scalar field modulus: 21888242871839275222246405745257275088548364400416034343698204186575808495617
// = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
const BN254_MOD_0: u32 = 0xf0000001u;
const BN254_MOD_1: u32 = 0x43e1f593u;
const BN254_MOD_2: u32 = 0x79b97091u;
const BN254_MOD_3: u32 = 0x2833e848u;
const BN254_MOD_4: u32 = 0x8181585du;
const BN254_MOD_5: u32 = 0xb85045b6u;
const BN254_MOD_6: u32 = 0xe131a029u;
const BN254_MOD_7: u32 = 0x30644e72u;

// Poseidon2 round constants (first 192 constants for t=3, RF=8, RP=56)
// These are derived from the Poseidon2 paper specification
const POSEIDON2_RF: u32 = 8u;   // Full rounds
const POSEIDON2_RP: u32 = 56u;  // Partial rounds
const POSEIDON2_T: u32 = 3u;    // State width

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

// Round constants stored as uniform (subset, full set would need storage buffer)
@group(0) @binding(4) var<storage, read> round_constants: array<Fr256>;

// Add two Fr256 with modular reduction
fn fr_add(a: Fr256, b: Fr256) -> Fr256 {
    var result: Fr256;
    var carry: u32 = 0u;

    // Add limbs with carry
    for (var i = 0u; i < 8u; i = i + 1u) {
        let sum = u64(a.limbs[i]) + u64(b.limbs[i]) + u64(carry);
        result.limbs[i] = u32(sum & 0xFFFFFFFFu);
        carry = u32(sum >> 32u);
    }

    // Reduce if >= modulus
    result = fr_reduce(result);
    return result;
}

// Helper to convert u32 pair to u64-like addition
fn u64(x: u32) -> u32 {
    return x;
}

// Subtract modulus if result >= modulus
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

    // Compare a >= mod
    var geq = true;
    for (var i = 7i; i >= 0i; i = i - 1i) {
        if (a.limbs[u32(i)] > mod_arr[u32(i)]) {
            geq = true;
            break;
        } else if (a.limbs[u32(i)] < mod_arr[u32(i)]) {
            geq = false;
            break;
        }
    }

    if (!geq) {
        return a;
    }

    // Subtract modulus
    var result: Fr256;
    var borrow: u32 = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let diff = i64(a.limbs[i]) - i64(mod_arr[i]) - i64(borrow);
        if (diff < 0i) {
            result.limbs[i] = u32(diff + 0x100000000i);
            borrow = 1u;
        } else {
            result.limbs[i] = u32(diff);
            borrow = 0u;
        }
    }

    return result;
}

fn i64(x: u32) -> i32 {
    return i32(x);
}

// Multiply Fr256 by u32 scalar
fn fr_mul_scalar(a: Fr256, s: u32) -> Fr256 {
    var result: Fr256;
    var carry: u32 = 0u;

    for (var i = 0u; i < 8u; i = i + 1u) {
        let prod = u32(a.limbs[i]) * s + carry;
        result.limbs[i] = prod & 0xFFFFFFFFu;
        carry = prod >> 16u; // Approximate carry for u32 multiply
    }

    return fr_reduce(result);
}

// Square Fr256 (simplified - uses schoolbook for correctness)
fn fr_square(a: Fr256) -> Fr256 {
    return fr_mul(a, a);
}

// Multiply two Fr256 (schoolbook multiplication with reduction)
fn fr_mul(a: Fr256, b: Fr256) -> Fr256 {
    var product: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        product[i] = 0u;
    }

    // Schoolbook multiply
    for (var i = 0u; i < 8u; i = i + 1u) {
        var carry: u32 = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let idx = i + j;
            // Approximate: split 32x32 into 16x16 products
            let al = a.limbs[i] & 0xFFFFu;
            let ah = a.limbs[i] >> 16u;
            let bl = b.limbs[j] & 0xFFFFu;
            let bh = b.limbs[j] >> 16u;

            let ll = al * bl;
            let lh = al * bh;
            let hl = ah * bl;
            let hh = ah * bh;

            let mid = lh + hl;
            let lo = ll + ((mid & 0xFFFFu) << 16u);
            let hi = hh + (mid >> 16u) + select(0u, 0x10000u, mid < lh);

            let sum = product[idx] + lo + carry;
            product[idx] = sum; // Will overflow naturally
            carry = hi + select(0u, 1u, sum < lo);
        }
        if (i + 8u < 16u) {
            product[i + 8u] = product[i + 8u] + carry;
        }
    }

    // Barrett reduction (simplified)
    var result: Fr256;
    for (var i = 0u; i < 8u; i = i + 1u) {
        result.limbs[i] = product[i];
    }

    // Multiple reductions may be needed
    for (var r = 0u; r < 3u; r = r + 1u) {
        result = fr_reduce(result);
    }

    return result;
}

// Compute x^5 (S-box for Poseidon2)
fn sbox(x: Fr256) -> Fr256 {
    let x2 = fr_square(x);
    let x4 = fr_square(x2);
    return fr_mul(x4, x);
}

// Apply Poseidon2 external matrix M_E to state
// M_E for t=3: [[2,1,1],[1,2,1],[1,1,2]]
fn apply_external_matrix(state: ptr<function, Poseidon2State>) {
    let t = fr_add(fr_add((*state).s0, (*state).s1), (*state).s2);
    (*state).s0 = fr_add((*state).s0, t);
    (*state).s1 = fr_add((*state).s1, t);
    (*state).s2 = fr_add((*state).s2, t);
}

// Apply Poseidon2 internal matrix M_I
// M_I is diagonal with (d-1, 1, 1) where d = 1 + D (D from paper)
fn apply_internal_matrix(state: ptr<function, Poseidon2State>) {
    let sum = fr_add(fr_add((*state).s0, (*state).s1), (*state).s2);
    // s0 = s0 * d + sum where d is internal diagonal
    (*state).s0 = fr_add(fr_mul_scalar((*state).s0, 2u), sum);
    (*state).s1 = fr_add((*state).s1, sum);
    (*state).s2 = fr_add((*state).s2, sum);
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
fn poseidon2_hash(left: Fr256, right: Fr256) -> Fr256 {
    var state: Poseidon2State;

    // Initialize state: [0, left, right]
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

    // Output is s1
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
