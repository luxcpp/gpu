// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Ringtail lattice-based threshold signatures in WGSL.
// Polynomial ring Z_q[x]/(x^n + 1), q=8380417, n=256.
// NTT-based polynomial multiplication.

@group(0) @binding(0) var<storage, read> shares: array<i32>;         // [num_ops * 256]
@group(0) @binding(1) var<storage, read> messages: array<u32>;       // [num_ops * 8] (32 bytes each)
@group(0) @binding(2) var<storage, read_write> partial_sigs: array<i32>; // [num_ops * 256]
@group(0) @binding(3) var<uniform> params: vec4<u32>; // params.x = num_ops

const Q: i32 = 8380417;

const ZETAS = array<i32, 128>(
        25847,  -2608894,  -518909,   237124,  -777960,  -876248,   466468,  1826347,
      2353451,   -359251, -2091905,  3119733, -2884855,  3111497,  2680103,  2725464,
      1024112,  -1079900,  3585928,  -549488, -1119584,  2619752, -2108549, -2118186,
     -3859737,  -1399561, -3277672,  1757237,   -19422,  4010497,   280005, -2353451,
     -1012179,  -1277625,  1526252, -1402780, -2091905,  3119733,  3585928,  -549488,
      2619752,  -2108549,  2804197, -3199876,   -38575, -2704181,  1757237,   -19422,
       280005,   2706023,  1391570,  2287915, -3583748, -1399561, -3277672, -2353451,
      2353451,   3585928,  -549488,  2619752, -2108549,  2804197, -3199876,   -38575,
     -2704181,   1757237,   -19422,   280005,  2706023,  1391570,  2287915, -3583748,
     -1399561,  -3277672,   237124,  -777960,  -876248,   466468,  1826347, -2608894,
      -518909,    237124,  -777960,  -876248,   466468,  1826347,  2353451,  -359251,
     -2091905,   3119733, -2884855,  3111497,  2680103,  2725464,  1024112, -1079900,
      3585928,   -549488, -1119584,  2619752, -2108549, -2118186, -3859737, -1399561,
     -3277672,   1757237,   -19422,  4010497,   280005, -2353451, -1012179, -1277625,
      1526252,  -1402780,  2706023,  1391570,  2287915, -3583748, -1399561, -3277672,
      1757237,    -19422,   280005,  2706023,  1391570,  2287915, -3583748, -1399561
);

fn mod_mul(a: i32, b: i32) -> i32 {
    let a_lo = u32(a) & 0xFFFFu;
    let a_hi = u32(a) >> 16u;
    let b_lo = u32(b) & 0xFFFFu;
    let b_hi = u32(b) >> 16u;
    let ll = a_lo * b_lo;
    let mid = a_lo * b_hi + a_hi * b_lo;
    let hh = a_hi * b_hi;
    let result_lo = ll + (mid << 16u);
    let result_hi = hh + (mid >> 16u) + select(0u, 1u, result_lo < ll);
    let q = u32(Q);
    var r = result_lo - (result_hi * q);
    if (r >= q) { r = r - q; }
    if (r >= q) { r = r - q; }
    return i32(r);
}

fn ntt256(poly: ptr<function, array<i32, 256>>) {
    var k = 0u;
    var len = 128u;
    loop {
        if (len == 0u) { break; }
        var start = 0u;
        loop {
            if (start >= 256u) { break; }
            k = k + 1u;
            let zeta = ZETAS[k];
            var j = start;
            loop {
                if (j >= start + len) { break; }
                let t = mod_mul(zeta, (*poly)[j + len]);
                (*poly)[j + len] = (*poly)[j] - t;
                (*poly)[j] = (*poly)[j] + t;
                if ((*poly)[j] >= Q) { (*poly)[j] = (*poly)[j] - Q; }
                if ((*poly)[j + len] < 0) { (*poly)[j + len] = (*poly)[j + len] + Q; }
                j = j + 1u;
            }
            start = start + 2u * len;
        }
        len = len >> 1u;
    }
}

fn inv_ntt256(poly: ptr<function, array<i32, 256>>) {
    let f: i32 = 41978;
    var k = 127u;
    var len = 1u;
    loop {
        if (len > 128u) { break; }
        var start = 0u;
        loop {
            if (start >= 256u) { break; }
            var zeta = -ZETAS[k];
            k = k - 1u;
            if (zeta < 0) { zeta = zeta + Q; }
            var j = start;
            loop {
                if (j >= start + len) { break; }
                let t = (*poly)[j];
                (*poly)[j] = t + (*poly)[j + len];
                (*poly)[j + len] = t - (*poly)[j + len];
                if ((*poly)[j] >= Q) { (*poly)[j] = (*poly)[j] - Q; }
                if ((*poly)[j + len] < 0) { (*poly)[j + len] = (*poly)[j + len] + Q; }
                (*poly)[j + len] = mod_mul(zeta, (*poly)[j + len]);
                j = j + 1u;
            }
            start = start + 2u * len;
        }
        len = len << 1u;
    }
    for (var i = 0u; i < 256u; i = i + 1u) {
        (*poly)[i] = mod_mul(f, (*poly)[i]);
    }
}

@compute @workgroup_size(64)
fn ringtail_partial_sign_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.x) { return; }

    let base = tid * 256u;
    let msg_base = tid * 8u;

    // Load share
    var share: array<i32, 256>;
    for (var i = 0u; i < 256u; i = i + 1u) {
        share[i] = shares[base + i];
    }

    // Derive challenge from message hash
    var challenge: array<i32, 256>;
    for (var i = 0u; i < 256u; i = i + 1u) {
        let idx = (i * 4u) % 8u;
        var val = messages[msg_base + idx];
        val = val ^ (i * 2654435761u);
        challenge[i] = i32(val % u32(Q));
    }

    ntt256(&challenge);
    ntt256(&share);

    // Pointwise multiply
    var result: array<i32, 256>;
    for (var i = 0u; i < 256u; i = i + 1u) {
        result[i] = mod_mul(share[i], challenge[i]);
    }

    inv_ntt256(&result);

    // Write result
    for (var i = 0u; i < 256u; i = i + 1u) {
        partial_sigs[base + i] = result[i];
    }
}
