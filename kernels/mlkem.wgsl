// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// ML-KEM-768 (FIPS 203) batch decapsulation in WGSL.
// NTT-based polynomial arithmetic over Z_q[x]/(x^n+1), q=3329, n=256.
// Each thread decapsulates one ciphertext.

@group(0) @binding(0) var<storage, read> sk_data: array<u32>;      // Secret keys (packed)
@group(0) @binding(1) var<storage, read> ct_data: array<u32>;      // Ciphertexts (packed)
@group(0) @binding(2) var<storage, read_write> out_data: array<u32>; // Shared secrets
@group(0) @binding(3) var<uniform> params: vec4<u32>; // params.x = num_ops

const Q: i32 = 3329;
const N: u32 = 256u;
const K: u32 = 3u;

const KYBER_ZETAS = array<i32, 128>(
    2285, 2571, 2970, 1812, 1493, 1422,  287,  202,
    3158,  622, 1577,  182,  962, 2127, 1855, 1468,
     573, 2004,  264,  383, 2500, 1458, 1727, 3199,
    2648, 1017,  732,  608, 1787,  411, 3124, 1758,
    1223,  652, 2777, 1015, 2036, 1491, 3047, 1785,
     516, 3321, 3009, 2663, 1711, 2167,  126, 1469,
    2476, 3239, 3058,  830,  107, 1908, 3082, 2378,
    2931,  961, 1821, 2604,  448, 2264,  677, 2054,
    2226,  430,  555,  843, 2078,  871, 1550,  105,
     422,  587,  177, 3094, 3038, 2869, 1574, 1653,
    3083,  778, 1159, 3182, 2552, 1483, 2727, 1119,
    1739,  644, 2457,  349,  418,  329, 3173, 3254,
     817, 1097,  603,  610, 1322, 2044, 1864,  384,
    2114, 3193, 1218, 1994, 2455,  220, 2142, 1670,
    2144, 1799, 2051,  794, 1819, 2475, 2459,  478,
    3221, 3116,  622, 1097, 2470,  882, 1539, 2392
);

fn mont_reduce_16(a: i32) -> i32 {
    let t: i32 = (a & 0xFFFF) * 3327;
    let u: i32 = (t & 0xFFFF) * Q;
    var r: i32 = (a - u) >> 16;
    if (r < 0) { r = r + Q; }
    if (r >= Q) { r = r - Q; }
    return r;
}

fn barrett_reduce(a: i32) -> i32 {
    let t = (a * 20159) >> 26;
    var r = a - t * Q;
    if (r >= Q) { r = r - Q; }
    if (r < 0) { r = r + Q; }
    return r;
}

fn kyber_ntt(poly: ptr<function, array<i32, 256>>) {
    var k = 0u;
    var len = 128u;
    loop {
        if (len < 2u) { break; }
        var start = 0u;
        loop {
            if (start >= 256u) { break; }
            k = k + 1u;
            let z = KYBER_ZETAS[k];
            var j = start;
            loop {
                if (j >= start + len) { break; }
                let t = mont_reduce_16(z * (*poly)[j + len]);
                (*poly)[j + len] = (*poly)[j] - t;
                (*poly)[j] = (*poly)[j] + t;
                j = j + 1u;
            }
            start = start + 2u * len;
        }
        len = len >> 1u;
    }
}

fn kyber_inv_ntt(poly: ptr<function, array<i32, 256>>) {
    let f: i32 = 1441; // Montgomery form of 256^{-1}
    var k = 127u;
    var len = 2u;
    loop {
        if (len > 128u) { break; }
        var start = 0u;
        loop {
            if (start >= 256u) { break; }
            let z = Q - KYBER_ZETAS[k];
            k = k - 1u;
            var j = start;
            loop {
                if (j >= start + len) { break; }
                let t = (*poly)[j];
                (*poly)[j] = t + (*poly)[j + len];
                (*poly)[j + len] = t - (*poly)[j + len];
                (*poly)[j + len] = mont_reduce_16(z * (*poly)[j + len]);
                j = j + 1u;
            }
            start = start + 2u * len;
        }
        len = len << 1u;
    }
    for (var i = 0u; i < 256u; i = i + 1u) {
        (*poly)[i] = mont_reduce_16(f * (*poly)[i]);
    }
}

fn read_byte_sk(base: u32, idx: u32) -> u32 {
    let word_idx = (base + idx) >> 2u;
    let byte_pos = (base + idx) & 3u;
    return (sk_data[word_idx] >> (byte_pos * 8u)) & 0xFFu;
}

fn read_byte_ct(base: u32, idx: u32) -> u32 {
    let word_idx = (base + idx) >> 2u;
    let byte_pos = (base + idx) & 3u;
    return (ct_data[word_idx] >> (byte_pos * 8u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn mlkem_decapsulate_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.x) { return; }

    let sk_base = tid * 2400u;
    let ct_base = tid * 1088u;

    // Decode s_hat (NTT domain, k=3 polynomials, 12 bits per coeff)
    var s_hat: array<array<i32, 256>, 3>;
    for (var p = 0u; p < 3u; p = p + 1u) {
        let sp = sk_base + p * 384u;
        for (var i = 0u; i < 256u; i = i + 1u) {
            let idx = i * 3u / 2u;
            if ((i & 1u) == 1u) {
                s_hat[p][i] = i32((read_byte_sk(sp, idx) >> 4u) | (read_byte_sk(sp, idx + 1u) << 4u)) & 0xFFF;
            } else {
                s_hat[p][i] = i32(read_byte_sk(sp, idx) | (read_byte_sk(sp, idx + 1u) << 8u)) & 0xFFF;
            }
        }
    }

    // Decode u (compressed, 10 bits per coeff) and decompress
    var u_hat: array<array<i32, 256>, 3>;
    for (var p = 0u; p < 3u; p = p + 1u) {
        let up = ct_base + p * 320u;
        for (var i = 0u; i < 256u; i = i + 4u) {
            let idx = (i / 4u) * 5u;
            let b0 = read_byte_ct(up, idx);
            let b1 = read_byte_ct(up, idx + 1u);
            let b2 = read_byte_ct(up, idx + 2u);
            let b3 = read_byte_ct(up, idx + 3u);
            let b4 = read_byte_ct(up, idx + 4u);

            u_hat[p][i]     = i32((b0 | ((b1 & 0x03u) << 8u)) * u32(Q) + 512u) >> 10;
            u_hat[p][i + 1u] = i32(((b1 >> 2u) | ((b2 & 0x0Fu) << 6u)) * u32(Q) + 512u) >> 10;
            u_hat[p][i + 2u] = i32(((b2 >> 4u) | ((b3 & 0x3Fu) << 4u)) * u32(Q) + 512u) >> 10;
            u_hat[p][i + 3u] = i32(((b3 >> 6u) | (b4 << 2u)) * u32(Q) + 512u) >> 10;
        }
        kyber_ntt(&u_hat[p]);
    }

    // Inner product: s_hat^T * u_hat
    var mp: array<i32, 256>;
    for (var i = 0u; i < 256u; i = i + 1u) { mp[i] = 0; }

    for (var p = 0u; p < 3u; p = p + 1u) {
        for (var i = 0u; i < 256u; i = i + 1u) {
            mp[i] = barrett_reduce(mp[i] + mont_reduce_16(s_hat[p][i] * u_hat[p][i]));
        }
    }

    kyber_inv_ntt(&mp);

    // Decode v (4 bits per coeff) and decompress
    let vp = ct_base + 960u; // 3 * 320
    var out_words: array<u32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) { out_words[i] = 0u; }

    for (var i = 0u; i < 256u; i = i + 1u) {
        let byte_idx = i / 2u;
        let v_raw = read_byte_ct(vp, byte_idx);
        var v_coeff: i32;
        if ((i & 1u) == 0u) {
            v_coeff = i32((v_raw & 0x0Fu) * u32(Q) + 8u) >> 4;
        } else {
            v_coeff = i32((v_raw >> 4u) * u32(Q) + 8u) >> 4;
        }

        var diff = v_coeff - mp[i];
        if (diff < 0) { diff = diff + Q; }
        let t = u32(diff) * 2u + u32(Q / 2);
        let bit = (t / u32(Q)) & 1u;
        out_words[i / 32u] = out_words[i / 32u] | (bit << (i % 32u));
    }

    let out_base = tid * 8u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        out_data[out_base + i] = out_words[i];
    }
}
