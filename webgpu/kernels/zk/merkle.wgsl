// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Merkle Tree Operations - WGSL Implementation for WebGPU
// Uses Poseidon2 as compression function

// BN254 scalar field modulus
const BN254_MOD_0: u32 = 0xf0000001u;
const BN254_MOD_1: u32 = 0x43e1f593u;
const BN254_MOD_2: u32 = 0x79b97091u;
const BN254_MOD_3: u32 = 0x2833e848u;
const BN254_MOD_4: u32 = 0x8181585du;
const BN254_MOD_5: u32 = 0xb85045b6u;
const BN254_MOD_6: u32 = 0xe131a029u;
const BN254_MOD_7: u32 = 0x30644e72u;

// Poseidon2 parameters
const POSEIDON2_RF: u32 = 8u;
const POSEIDON2_RP: u32 = 56u;

struct Fr256 {
    limbs: array<u32, 8>,
}

struct Poseidon2State {
    s0: Fr256,
    s1: Fr256,
    s2: Fr256,
}

// Merkle tree layer computation
@group(0) @binding(0) var<storage, read> input_nodes: array<Fr256>;
@group(0) @binding(1) var<storage, read_write> output_nodes: array<Fr256>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // [num_pairs, layer_idx, 0, 0]
@group(0) @binding(3) var<storage, read> round_constants: array<Fr256>;

// Field arithmetic (same as poseidon2.wgsl)
fn fr_add(a: Fr256, b: Fr256) -> Fr256 {
    var result: Fr256;
    var carry: u32 = 0u;

    for (var i = 0u; i < 8u; i = i + 1u) {
        let ai = a.limbs[i];
        let bi = b.limbs[i];
        let sum_lo = ai + bi;
        let sum = sum_lo + carry;
        result.limbs[i] = sum;
        // Carry if overflow occurred
        carry = select(0u, 1u, sum < ai || (sum_lo < ai));
    }

    return fr_reduce(result);
}

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

    var geq = true;
    for (var i = 7i; i >= 0i; i = i - 1i) {
        let idx = u32(i);
        if (a.limbs[idx] > mod_arr[idx]) {
            geq = true;
            break;
        } else if (a.limbs[idx] < mod_arr[idx]) {
            geq = false;
            break;
        }
    }

    if (!geq) {
        return a;
    }

    var result: Fr256;
    var borrow: u32 = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let ai = a.limbs[i];
        let mi = mod_arr[i];
        if (ai >= mi + borrow) {
            result.limbs[i] = ai - mi - borrow;
            borrow = 0u;
        } else {
            result.limbs[i] = ai + (0xFFFFFFFFu - mi - borrow) + 1u;
            borrow = 1u;
        }
    }

    return result;
}

fn fr_mul_scalar(a: Fr256, s: u32) -> Fr256 {
    var result: Fr256;
    var carry: u32 = 0u;

    for (var i = 0u; i < 8u; i = i + 1u) {
        let al = a.limbs[i] & 0xFFFFu;
        let ah = a.limbs[i] >> 16u;
        let prod_lo = al * s;
        let prod_hi = ah * s;
        let combined = prod_lo + ((prod_hi & 0xFFFFu) << 16u) + carry;
        result.limbs[i] = combined;
        carry = (prod_hi >> 16u) + select(0u, 1u, combined < prod_lo);
    }

    return fr_reduce(result);
}

fn fr_square(a: Fr256) -> Fr256 {
    return fr_mul(a, a);
}

fn fr_mul(a: Fr256, b: Fr256) -> Fr256 {
    var product: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        product[i] = 0u;
    }

    for (var i = 0u; i < 8u; i = i + 1u) {
        var carry: u32 = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let idx = i + j;
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
            product[idx] = sum;
            carry = hi + select(0u, 1u, sum < lo || sum < product[idx] - lo - carry + lo);
        }
        if (i + 8u < 16u) {
            product[i + 8u] = product[i + 8u] + carry;
        }
    }

    var result: Fr256;
    for (var i = 0u; i < 8u; i = i + 1u) {
        result.limbs[i] = product[i];
    }

    for (var r = 0u; r < 4u; r = r + 1u) {
        result = fr_reduce(result);
    }

    return result;
}

fn sbox(x: Fr256) -> Fr256 {
    let x2 = fr_square(x);
    let x4 = fr_square(x2);
    return fr_mul(x4, x);
}

fn apply_external_matrix(state: ptr<function, Poseidon2State>) {
    let t = fr_add(fr_add((*state).s0, (*state).s1), (*state).s2);
    (*state).s0 = fr_add((*state).s0, t);
    (*state).s1 = fr_add((*state).s1, t);
    (*state).s2 = fr_add((*state).s2, t);
}

fn apply_internal_matrix(state: ptr<function, Poseidon2State>) {
    let sum = fr_add(fr_add((*state).s0, (*state).s1), (*state).s2);
    (*state).s0 = fr_add(fr_mul_scalar((*state).s0, 2u), sum);
    (*state).s1 = fr_add((*state).s1, sum);
    (*state).s2 = fr_add((*state).s2, sum);
}

fn full_round(state: ptr<function, Poseidon2State>, rc_offset: u32) {
    (*state).s0 = fr_add((*state).s0, round_constants[rc_offset]);
    (*state).s1 = fr_add((*state).s1, round_constants[rc_offset + 1u]);
    (*state).s2 = fr_add((*state).s2, round_constants[rc_offset + 2u]);

    (*state).s0 = sbox((*state).s0);
    (*state).s1 = sbox((*state).s1);
    (*state).s2 = sbox((*state).s2);

    apply_external_matrix(state);
}

fn partial_round(state: ptr<function, Poseidon2State>, rc_offset: u32) {
    (*state).s0 = fr_add((*state).s0, round_constants[rc_offset]);
    (*state).s0 = sbox((*state).s0);
    apply_internal_matrix(state);
}

fn poseidon2_compress(left: Fr256, right: Fr256) -> Fr256 {
    var state: Poseidon2State;

    for (var i = 0u; i < 8u; i = i + 1u) {
        state.s0.limbs[i] = 0u;
    }
    state.s1 = left;
    state.s2 = right;

    var rc_idx = 0u;

    for (var r = 0u; r < POSEIDON2_RF / 2u; r = r + 1u) {
        full_round(&state, rc_idx);
        rc_idx = rc_idx + 3u;
    }

    for (var r = 0u; r < POSEIDON2_RP; r = r + 1u) {
        partial_round(&state, rc_idx);
        rc_idx = rc_idx + 1u;
    }

    for (var r = 0u; r < POSEIDON2_RF / 2u; r = r + 1u) {
        full_round(&state, rc_idx);
        rc_idx = rc_idx + 3u;
    }

    return state.s1;
}

// Compute one layer of Merkle tree
// Each thread computes: output[i] = Poseidon2(input[2*i], input[2*i+1])
@compute @workgroup_size(64)
fn merkle_layer(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_pairs = params.x;

    if (idx >= num_pairs) {
        return;
    }

    let left = input_nodes[idx * 2u];
    let right = input_nodes[idx * 2u + 1u];

    output_nodes[idx] = poseidon2_compress(left, right);
}

// Commitment: Poseidon2(Poseidon2(value, blinding), salt)
@compute @workgroup_size(64)
fn commitment(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let count = params.x;

    if (idx >= count) {
        return;
    }

    // Input layout: [values..., blindings..., salts...]
    let value = input_nodes[idx];
    let blinding = input_nodes[count + idx];
    let salt = input_nodes[count * 2u + idx];

    let inner = poseidon2_compress(value, blinding);
    output_nodes[idx] = poseidon2_compress(inner, salt);
}

// Nullifier: Poseidon2(Poseidon2(key, commitment), index)
@compute @workgroup_size(64)
fn nullifier(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let count = params.x;

    if (idx >= count) {
        return;
    }

    // Input layout: [keys..., commitments..., indices...]
    let key = input_nodes[idx];
    let commitment_val = input_nodes[count + idx];
    let index = input_nodes[count * 2u + idx];

    let inner = poseidon2_compress(key, commitment_val);
    output_nodes[idx] = poseidon2_compress(inner, index);
}
