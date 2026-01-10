// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Merkle Tree Operations - WGSL Implementation for WebGPU
// Uses Poseidon2 as compression function
//
// Field: Fr of BN254 (alt_bn128)
// Modulus p = 21888242871839275222246405745257275088548364400416034343698204186575808495617

// BN254 scalar field modulus (little-endian limbs)
const BN254_MOD_0: u32 = 0xf0000001u;
const BN254_MOD_1: u32 = 0x43e1f593u;
const BN254_MOD_2: u32 = 0x79b97091u;
const BN254_MOD_3: u32 = 0x2833e848u;
const BN254_MOD_4: u32 = 0x8181585du;
const BN254_MOD_5: u32 = 0xb85045b6u;
const BN254_MOD_6: u32 = 0xe131a029u;
const BN254_MOD_7: u32 = 0x30644e72u;

// Montgomery constant: -p^(-1) mod 2^32
const MONT_INV: u32 = 0xefffffff;

// Poseidon2 parameters for t=3 on BN254
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

// ============================================================================
// 256-bit Arithmetic Without u64
// ============================================================================

// Add with carry: returns (sum, carry)
fn adc(a: u32, b: u32, cin: u32) -> vec2<u32> {
    let sum1 = a + b;
    let carry1 = select(0u, 1u, sum1 < a);
    let sum2 = sum1 + cin;
    let carry2 = select(0u, 1u, sum2 < sum1);
    return vec2<u32>(sum2, carry1 + carry2);
}

// Subtract with borrow: returns (diff, borrow)
fn sbb(a: u32, b: u32, bin: u32) -> vec2<u32> {
    let diff1 = a - b;
    let borrow1 = select(0u, 1u, a < b);
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

    let mid = lh + hl;
    let mid_carry = select(0u, 0x10000u, mid < lh);

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

fn fr_geq(a: Fr256, b_arr: array<u32, 8>) -> bool {
    for (var i = 7i; i >= 0i; i = i - 1i) {
        let idx = u32(i);
        if (a.limbs[idx] > b_arr[idx]) {
            return true;
        } else if (a.limbs[idx] < b_arr[idx]) {
            return false;
        }
    }
    return true;
}

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

fn fr_add(a: Fr256, b: Fr256) -> Fr256 {
    var result: Fr256;
    var carry: u32 = 0u;

    for (var i = 0u; i < 8u; i = i + 1u) {
        let ac = adc(a.limbs[i], b.limbs[i], carry);
        result.limbs[i] = ac.x;
        carry = ac.y;
    }

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

// ============================================================================
// Montgomery Reduction
// ============================================================================

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

    for (var i = 0u; i < 8u; i = i + 1u) {
        let m = tmp[i] * MONT_INV;

        var carry: u32 = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let mc = mac(m, mod_arr[j], tmp[i + j], carry);
            tmp[i + j] = mc.x;
            carry = mc.y;
        }

        for (var j = i + 8u; j < 17u; j = j + 1u) {
            let ac = adc(tmp[j], carry, 0u);
            tmp[j] = ac.x;
            carry = ac.y;
            if (carry == 0u) {
                break;
            }
        }
    }

    var result: Fr256;
    for (var i = 0u; i < 8u; i = i + 1u) {
        result.limbs[i] = tmp[i + 8u];
    }

    return fr_reduce(result);
}

fn fr_mul(a: Fr256, b: Fr256) -> Fr256 {
    var product: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        product[i] = 0u;
    }

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

fn fr_square(a: Fr256) -> Fr256 {
    return fr_mul(a, a);
}

// ============================================================================
// Poseidon2 Hash
// ============================================================================

fn sbox(x: Fr256) -> Fr256 {
    let x2 = fr_square(x);
    let x4 = fr_square(x2);
    return fr_mul(x4, x);
}

fn apply_external_matrix(state: ptr<function, Poseidon2State>) {
    let sum = fr_add(fr_add((*state).s0, (*state).s1), (*state).s2);
    (*state).s0 = fr_add((*state).s0, sum);
    (*state).s1 = fr_add((*state).s1, sum);
    (*state).s2 = fr_add((*state).s2, sum);
}

fn apply_internal_matrix(state: ptr<function, Poseidon2State>) {
    let sum = fr_add(fr_add((*state).s0, (*state).s1), (*state).s2);
    (*state).s0 = fr_add((*state).s0, sum);
    (*state).s1 = sum;
    (*state).s2 = sum;
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

// ============================================================================
// Merkle Tree Kernels
// ============================================================================

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

// Pedersen-style commitment: Poseidon2(Poseidon2(value, blinding), salt)
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

// Nullifier computation: Poseidon2(Poseidon2(key, commitment), index)
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
