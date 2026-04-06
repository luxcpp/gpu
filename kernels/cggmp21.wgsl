// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// CGGMP21 threshold ECDSA partial signing in WGSL.
// Computes sigma_i = k_i * m + r * chi_i (mod n) for each participant.
// Uses secp256k1 order n for scalar arithmetic.

@group(0) @binding(0) var<storage, read> inputs: array<u32>;       // CGGMP21Input packed
@group(0) @binding(1) var<storage, read_write> outputs: array<u32>; // sigma_i
@group(0) @binding(2) var<storage, read> r_x: array<u32>;          // R.x (32 bytes)
@group(0) @binding(3) var<uniform> params: vec4<u32>; // params.x = num_ops

// secp256k1 order n (little-endian u32 limbs)
const SECP_N = array<u32, 8>(
    0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
    0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
);

// 256-bit addition mod n
fn scalar_add_mod_n(a: ptr<function, array<u32, 8>>,
                    b: ptr<function, array<u32, 8>>,
                    r: ptr<function, array<u32, 8>>) {
    var c = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let sum = (*a)[i] + c;
        c = select(0u, 1u, sum < (*a)[i]);
        let sum2 = sum + (*b)[i];
        c = c + select(0u, 1u, sum2 < sum);
        (*r)[i] = sum2;
    }
    // Reduce mod n
    var n_val: array<u32, 8> = SECP_N;
    var bw = 0u;
    var diff: array<u32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let d = (*r)[i] - bw;
        bw = select(0u, 1u, d > (*r)[i]);
        let d2 = d - n_val[i];
        bw = bw + select(0u, 1u, d2 > d);
        diff[i] = d2;
    }
    if (bw == 0u || c != 0u) {
        for (var i = 0u; i < 8u; i = i + 1u) { (*r)[i] = diff[i]; }
    }
}

// 256-bit multiplication mod n (schoolbook + iterative reduction)
fn scalar_mul_mod_n(a: ptr<function, array<u32, 8>>,
                    b: ptr<function, array<u32, 8>>,
                    r: ptr<function, array<u32, 8>>) {
    // Schoolbook 256x256 -> 512 bit multiply using 16-bit pieces
    var t: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { t[i] = 0u; }

    for (var i = 0u; i < 8u; i = i + 1u) {
        var carry = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let a_lo = (*a)[i] & 0xFFFFu;
            let a_hi = (*a)[i] >> 16u;
            let b_lo = (*b)[j] & 0xFFFFu;
            let b_hi = (*b)[j] >> 16u;

            let ll = a_lo * b_lo;
            let lh = a_lo * b_hi;
            let hl = a_hi * b_lo;
            let hh = a_hi * b_hi;

            let mid = lh + hl;
            let lo = ll + (mid << 16u) + carry + t[i + j];
            let hi = hh + (mid >> 16u) + select(0u, 1u, lo < t[i + j]);

            t[i + j] = lo;
            carry = hi;
        }
        t[i + 8u] = carry;
    }

    // Take low 256 bits and reduce mod n iteratively
    for (var i = 0u; i < 8u; i = i + 1u) { (*r)[i] = t[i]; }

    var n_val: array<u32, 8> = SECP_N;
    for (var iter = 0u; iter < 4u; iter = iter + 1u) {
        var bw = 0u;
        var diff: array<u32, 8>;
        for (var i = 0u; i < 8u; i = i + 1u) {
            let d = (*r)[i] - bw;
            bw = select(0u, 1u, d > (*r)[i]);
            let d2 = d - n_val[i];
            bw = bw + select(0u, 1u, d2 > d);
            diff[i] = d2;
        }
        if (bw == 0u) {
            for (var i = 0u; i < 8u; i = i + 1u) { (*r)[i] = diff[i]; }
        }
    }
}

@compute @workgroup_size(64)
fn cggmp21_partial_sign_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.x) { return; }

    // Input layout per thread: k_share[8] || chi_share[8] || msg_hash[8] || gamma[8] = 32 u32
    let in_base = tid * 32u;

    var k: array<u32, 8>;
    var chi: array<u32, 8>;
    var msg: array<u32, 8>;

    for (var i = 0u; i < 8u; i = i + 1u) {
        k[i] = inputs[in_base + i];
        chi[i] = inputs[in_base + 8u + i];
        msg[i] = inputs[in_base + 16u + i];
    }

    var r: array<u32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) { r[i] = r_x[i]; }

    // sigma_i = k_i * m + r * chi_i  (mod n)
    var km: array<u32, 8>;
    var rchi: array<u32, 8>;
    var sigma: array<u32, 8>;

    scalar_mul_mod_n(&k, &msg, &km);
    scalar_mul_mod_n(&r, &chi, &rchi);
    scalar_add_mod_n(&km, &rchi, &sigma);

    let out_base = tid * 8u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        outputs[out_base + i] = sigma[i];
    }
}
