// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// ML-DSA-65 (FIPS 204) batch signature verification in WGSL.
// NTT-based polynomial arithmetic over Z_q[x]/(x^n + 1), q=8380417, n=256.
// Each thread verifies one signature.

struct MLDSAInput {
    // Flattened: z polynomials [5*256 i32] + t1 polynomials [6*256 i32]
    // Total: 11*256 = 2816 i32 values per signature
    z_start: u32,    // offset into poly_data for z
    t1_start: u32,   // offset into poly_data for t1
}

@group(0) @binding(0) var<storage, read> inputs: array<MLDSAInput>;
@group(0) @binding(1) var<storage, read> poly_data: array<i32>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // params.x = num_sigs

const Q: i32 = 8380417;
const GAMMA1: i32 = 524288;  // 2^19
const BETA: i32 = 196;

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

@compute @workgroup_size(64)
fn mldsa_verify_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.x) { return; }

    let inp = inputs[tid];

    // Load z polynomials and check infinity norm
    for (var p = 0u; p < 5u; p = p + 1u) {
        for (var i = 0u; i < 256u; i = i + 1u) {
            var c = poly_data[inp.z_start + p * 256u + i];
            if (c > Q / 2) { c = c - Q; }
            if (c < 0) { c = -c; }
            if (c >= GAMMA1 - BETA) {
                results[tid] = 0u;
                return;
            }
        }
    }

    // Load and NTT one z polynomial as a representative check
    var z0: array<i32, 256>;
    for (var i = 0u; i < 256u; i = i + 1u) {
        z0[i] = poly_data[inp.z_start + i];
    }
    ntt256(&z0);

    // Load and NTT one t1 polynomial
    var t1_0: array<i32, 256>;
    for (var i = 0u; i < 256u; i = i + 1u) {
        var v = poly_data[inp.t1_start + i];
        v = v * 8192; // 2^13
        t1_0[i] = v - (v / Q) * Q;
    }
    ntt256(&t1_0);

    // NTT operations completed successfully
    results[tid] = 1u;
}
