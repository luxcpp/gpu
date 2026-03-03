// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// EVM uint256 parallel operations in WGSL.
// uint256 = 8 x u32 limbs (little-endian, since WGSL has no u64).
// Matches evm256.metal output byte-for-byte.

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> result: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;
// params.x = num_items, params.y = op (0=add 1=sub 2=mul 3=div 4=mod)

const LIMBS: u32 = 8u; // 256 bits / 32 bits

fn load256(buf: ptr<storage, array<u32>, read>, idx: u32) -> array<u32, 8> {
    var v: array<u32, 8>;
    let base = idx * LIMBS;
    for (var i = 0u; i < LIMBS; i = i + 1u) { v[i] = (*buf)[base + i]; }
    return v;
}

fn store256(idx: u32, v: ptr<function, array<u32, 8>>) {
    let base = idx * LIMBS;
    for (var i = 0u; i < LIMBS; i = i + 1u) { result[base + i] = (*v)[i]; }
}

fn add256(x: ptr<function, array<u32, 8>>, y: ptr<function, array<u32, 8>>,
          r: ptr<function, array<u32, 8>>) -> u32 {
    var c = 0u;
    for (var i = 0u; i < LIMBS; i = i + 1u) {
        let s1 = (*x)[i] + c;
        c = select(0u, 1u, s1 < (*x)[i]);
        let s2 = s1 + (*y)[i];
        c = c + select(0u, 1u, s2 < s1);
        (*r)[i] = s2;
    }
    return c;
}

fn sub256(x: ptr<function, array<u32, 8>>, y: ptr<function, array<u32, 8>>,
          r: ptr<function, array<u32, 8>>) -> u32 {
    var bw = 0u;
    for (var i = 0u; i < LIMBS; i = i + 1u) {
        let d1 = (*x)[i] - bw;
        bw = select(0u, 1u, d1 > (*x)[i]);
        let d2 = d1 - (*y)[i];
        bw = bw + select(0u, 1u, d2 > d1);
        (*r)[i] = d2;
    }
    return bw;
}

fn is_zero256(v: ptr<function, array<u32, 8>>) -> bool {
    var acc = 0u;
    for (var i = 0u; i < LIMBS; i = i + 1u) { acc = acc | (*v)[i]; }
    return acc == 0u;
}

fn cmp256(x: ptr<function, array<u32, 8>>, y: ptr<function, array<u32, 8>>) -> i32 {
    for (var i = 7i; i >= 0; i = i - 1) {
        let ui = u32(i);
        if ((*x)[ui] > (*y)[ui]) { return 1; }
        if ((*x)[ui] < (*y)[ui]) { return -1; }
    }
    return 0;
}

fn mul256(x: ptr<function, array<u32, 8>>, y: ptr<function, array<u32, 8>>,
          r: ptr<function, array<u32, 8>>) {
    var prod: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { prod[i] = 0u; }

    for (var i = 0u; i < LIMBS; i = i + 1u) {
        var carry = 0u;
        for (var j = 0u; j < LIMBS; j = j + 1u) {
            // 32x32 -> 64 multiply
            let a_lo = (*x)[i] & 0xFFFFu;
            let a_hi = (*x)[i] >> 16u;
            let b_lo = (*y)[j] & 0xFFFFu;
            let b_hi = (*y)[j] >> 16u;
            let ll = a_lo * b_lo;
            let lh = a_lo * b_hi;
            let hl = a_hi * b_lo;
            let hh = a_hi * b_hi;
            let mid = lh + hl;
            let lo = ll + (mid << 16u);
            var hi = hh + (mid >> 16u) + select(0u, 1u, lo < ll) + select(0u, 0x10000u, mid < lh);

            // Add carry
            let s1 = lo + carry;
            hi = hi + select(0u, 1u, s1 < lo);
            // Add to accumulator
            let s2 = prod[i + j] + s1;
            hi = hi + select(0u, 1u, s2 < prod[i + j]);
            prod[i + j] = s2;
            carry = hi;
        }
        prod[i + LIMBS] = prod[i + LIMBS] + carry;
    }

    // Take lower 256 bits
    for (var i = 0u; i < LIMBS; i = i + 1u) { (*r)[i] = prod[i]; }
}

fn div256(num: ptr<function, array<u32, 8>>, den: ptr<function, array<u32, 8>>,
          q: ptr<function, array<u32, 8>>, rem: ptr<function, array<u32, 8>>) {
    // Zero outputs
    for (var i = 0u; i < LIMBS; i = i + 1u) { (*q)[i] = 0u; (*rem)[i] = 0u; }

    if (is_zero256(den)) { return; }

    // Long division bit by bit
    for (var bit = 255i; bit >= 0; bit = bit - 1) {
        // rem <<= 1
        var c = 0u;
        for (var j = 0u; j < LIMBS; j = j + 1u) {
            let temp = ((*rem)[j] << 1u) | c;
            c = (*rem)[j] >> 31u;
            (*rem)[j] = temp;
        }

        // rem[0] |= bit from numerator
        let limb_idx = u32(bit) / 32u;
        let bit_idx = u32(bit) % 32u;
        (*rem)[0] = (*rem)[0] | (((*num)[limb_idx] >> bit_idx) & 1u);

        // if rem >= den: rem -= den, q[bit] = 1
        if (cmp256(rem, den) >= 0) {
            let _ = sub256(rem, den, rem);
            (*q)[limb_idx] = (*q)[limb_idx] | (1u << bit_idx);
        }
    }
}

@compute @workgroup_size(256)
fn evm256_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.x) { return; }

    var va = load256(&a, tid);
    var vb = load256(&b, tid);
    var vr: array<u32, 8>;
    for (var i = 0u; i < LIMBS; i = i + 1u) { vr[i] = 0u; }

    let op = params.y;
    if (op == 0u) {
        let _ = add256(&va, &vb, &vr);
    } else if (op == 1u) {
        let _ = sub256(&va, &vb, &vr);
    } else if (op == 2u) {
        mul256(&va, &vb, &vr);
    } else if (op == 3u) {
        var rem: array<u32, 8>;
        div256(&va, &vb, &vr, &rem);
    } else if (op == 4u) {
        var quot: array<u32, 8>;
        div256(&va, &vb, &quot, &vr);
    }

    store256(tid, &vr);
}
