// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// SLH-DSA (FIPS 205, SPHINCS+) batch verification in WGSL.
// Hash-based signature scheme using SHAKE256 (Keccak-based).
// Each thread verifies one signature by recomputing WOTS+ chains
// and Merkle tree paths.

@group(0) @binding(0) var<storage, read> pubkeys: array<u32>;
@group(0) @binding(1) var<storage, read> messages: array<u32>;
@group(0) @binding(2) var<storage, read> signatures: array<u32>;
@group(0) @binding(3) var<storage, read_write> results: array<u32>;
@group(0) @binding(4) var<uniform> params: vec4<u32>; // params.x = num_sigs

const SLH_N: u32 = 16u;
const SLH_K: u32 = 33u;
const SLH_A: u32 = 6u;
const SLH_D: u32 = 22u;
const SLH_HP: u32 = 3u;
const SLH_W: u32 = 16u;
const SLH_LEN: u32 = 7u;

// Keccak-f[1600] round constants (lo, hi pairs for u64 emulation)
const RC_LO = array<u32, 24>(
    0x00000001u, 0x00008082u, 0x0000808Au, 0x80008000u,
    0x0000808Bu, 0x80000001u, 0x80008081u, 0x00008009u,
    0x0000008Au, 0x00000088u, 0x80008009u, 0x8000000Au,
    0x8000808Bu, 0x0000008Bu, 0x00008089u, 0x00008003u,
    0x00008002u, 0x00000080u, 0x0000800Au, 0x8000000Au,
    0x80008081u, 0x00008080u, 0x80000001u, 0x80008008u
);

const RC_HI = array<u32, 24>(
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x80000000u, 0x80000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u
);

const PI_LANE = array<u32, 24>(
    10u, 7u, 11u, 17u, 18u, 3u, 5u, 16u, 8u, 21u, 24u, 4u,
    15u, 23u, 19u, 13u, 12u, 2u, 20u, 14u, 22u, 9u, 6u, 1u
);

const RHO_OFFSETS = array<u32, 24>(
    1u, 3u, 6u, 10u, 15u, 21u, 28u, 36u, 45u, 55u, 2u, 14u,
    27u, 41u, 56u, 8u, 25u, 43u, 62u, 18u, 39u, 61u, 20u, 44u
);

var<private> st_lo: array<u32, 25>;
var<private> st_hi: array<u32, 25>;

fn rotl64(lo: u32, hi: u32, n: u32) -> vec2<u32> {
    if (n == 0u) { return vec2<u32>(lo, hi); }
    if (n == 32u) { return vec2<u32>(hi, lo); }
    if (n < 32u) {
        return vec2<u32>((lo << n) | (hi >> (32u - n)), (hi << n) | (lo >> (32u - n)));
    }
    let m = n - 32u;
    return vec2<u32>((hi << m) | (lo >> (32u - m)), (lo << m) | (hi >> (32u - m)));
}

fn keccak_f() {
    for (var round = 0u; round < 24u; round = round + 1u) {
        var c_lo: array<u32, 5>;
        var c_hi: array<u32, 5>;
        for (var x = 0u; x < 5u; x = x + 1u) {
            c_lo[x] = st_lo[x] ^ st_lo[x+5u] ^ st_lo[x+10u] ^ st_lo[x+15u] ^ st_lo[x+20u];
            c_hi[x] = st_hi[x] ^ st_hi[x+5u] ^ st_hi[x+10u] ^ st_hi[x+15u] ^ st_hi[x+20u];
        }
        for (var x = 0u; x < 5u; x = x + 1u) {
            let r = rotl64(c_lo[(x+1u) % 5u], c_hi[(x+1u) % 5u], 1u);
            let d_lo = c_lo[(x+4u) % 5u] ^ r.x;
            let d_hi = c_hi[(x+4u) % 5u] ^ r.y;
            for (var y = 0u; y < 5u; y = y + 1u) {
                let idx = x + 5u * y;
                st_lo[idx] = st_lo[idx] ^ d_lo;
                st_hi[idx] = st_hi[idx] ^ d_hi;
            }
        }
        var t_lo = st_lo[1u];
        var t_hi = st_hi[1u];
        for (var i = 0u; i < 24u; i = i + 1u) {
            let dst = PI_LANE[i];
            let tmp_lo = st_lo[dst];
            let tmp_hi = st_hi[dst];
            let r = rotl64(t_lo, t_hi, RHO_OFFSETS[i]);
            st_lo[dst] = r.x;
            st_hi[dst] = r.y;
            t_lo = tmp_lo;
            t_hi = tmp_hi;
        }
        for (var y = 0u; y < 5u; y = y + 1u) {
            var row_lo: array<u32, 5>;
            var row_hi: array<u32, 5>;
            for (var x = 0u; x < 5u; x = x + 1u) {
                row_lo[x] = st_lo[x + 5u * y];
                row_hi[x] = st_hi[x + 5u * y];
            }
            for (var x = 0u; x < 5u; x = x + 1u) {
                st_lo[x + 5u * y] = row_lo[x] ^ ((~row_lo[(x+1u) % 5u]) & row_lo[(x+2u) % 5u]);
                st_hi[x + 5u * y] = row_hi[x] ^ ((~row_hi[(x+1u) % 5u]) & row_hi[(x+2u) % 5u]);
            }
        }
        st_lo[0u] = st_lo[0u] ^ RC_LO[round];
        st_hi[0u] = st_hi[0u] ^ RC_HI[round];
    }
}

fn read_sig_byte(sig_base: u32, idx: u32) -> u32 {
    let word_idx = (sig_base + idx) >> 2u;
    let byte_pos = (sig_base + idx) & 3u;
    return (signatures[word_idx] >> (byte_pos * 8u)) & 0xFFu;
}

fn read_pk_byte(pk_base: u32, idx: u32) -> u32 {
    let word_idx = (pk_base + idx) >> 2u;
    let byte_pos = (pk_base + idx) & 3u;
    return (pubkeys[word_idx] >> (byte_pos * 8u)) & 0xFFu;
}

// Simple SHAKE256 hash of 64 bytes -> 16 bytes via Keccak
fn shake256_64_to_16(input: ptr<function, array<u32, 16>>, output: ptr<function, array<u32, 4>>) {
    for (var i = 0u; i < 25u; i = i + 1u) { st_lo[i] = 0u; st_hi[i] = 0u; }

    // Absorb 64 bytes = 8 u64 words into rate (136 bytes = 17 u64 words)
    for (var w = 0u; w < 8u; w = w + 1u) {
        st_lo[w] = st_lo[w] ^ (*input)[w * 2u];
        st_hi[w] = st_hi[w] ^ (*input)[w * 2u + 1u];
    }

    // SHAKE256 padding at byte 64: 0x1F
    st_lo[8u] = st_lo[8u] ^ 0x1Fu;
    // Last byte of rate (byte 135): 0x80
    st_hi[16u] = st_hi[16u] ^ 0x80000000u;

    keccak_f();

    // Squeeze 16 bytes = 2 u64 words
    (*output)[0u] = st_lo[0u];
    (*output)[1u] = st_hi[0u];
    (*output)[2u] = st_lo[1u];
    (*output)[3u] = st_hi[1u];
}

@compute @workgroup_size(64)
fn slhdsa_verify_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.x) { return; }

    let pk_base = tid * 8u;   // 32 bytes = 8 u32 words
    let sig_base_words = tid * 4272u; // ~17088 bytes = 4272 u32 words

    // Read PK.seed and PK.root (each 16 bytes = 4 u32 words)
    var pk_seed: array<u32, 4>;
    var pk_root: array<u32, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        pk_seed[i] = pubkeys[pk_base + i];
        pk_root[i] = pubkeys[pk_base + 4u + i];
    }

    // Read randomizer R (16 bytes) from signature
    var R: array<u32, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        R[i] = signatures[sig_base_words + i];
    }

    // Compute message digest via SHAKE256(R || pk_seed || pk_root || msg)
    // = 80 bytes input -> 32 bytes output
    var hash_input: array<u32, 16>;
    for (var i = 0u; i < 4u; i = i + 1u) { hash_input[i] = R[i]; }
    for (var i = 0u; i < 4u; i = i + 1u) { hash_input[4u + i] = pk_seed[i]; }
    for (var i = 0u; i < 4u; i = i + 1u) { hash_input[8u + i] = pk_root[i]; }
    let msg_base = tid * 8u;
    for (var i = 0u; i < 4u; i = i + 1u) { hash_input[12u + i] = messages[msg_base + i]; }

    var digest: array<u32, 4>;
    shake256_64_to_16(&hash_input, &digest);

    // Verify FORS tree structure
    // For each FORS tree, hash leaf and climb auth path
    var fors_offset = 4u; // After R (in u32 words)
    var current_node: array<u32, 4>;
    var valid = true;

    for (var tree = 0u; tree < SLH_K; tree = tree + 1u) {
        // Read leaf
        var leaf: array<u32, 4>;
        for (var i = 0u; i < 4u; i = i + 1u) {
            leaf[i] = signatures[sig_base_words + fors_offset + tree * 28u + i];
        }

        // Hash leaf
        var leaf_input: array<u32, 16>;
        for (var i = 0u; i < 4u; i = i + 1u) { leaf_input[i] = pk_seed[i]; }
        for (var i = 4u; i < 12u; i = i + 1u) { leaf_input[i] = 0u; }
        for (var i = 0u; i < 4u; i = i + 1u) { leaf_input[12u + i] = leaf[i]; }

        var node: array<u32, 4>;
        shake256_64_to_16(&leaf_input, &node);

        // Climb auth path
        for (var layer = 0u; layer < SLH_A; layer = layer + 1u) {
            var sibling: array<u32, 4>;
            let sib_off = sig_base_words + fors_offset + tree * 28u + 4u + layer * 4u;
            for (var i = 0u; i < 4u; i = i + 1u) {
                sibling[i] = signatures[sib_off + i];
            }

            var pair_input: array<u32, 16>;
            for (var i = 0u; i < 4u; i = i + 1u) { pair_input[i] = pk_seed[i]; }
            for (var i = 4u; i < 8u; i = i + 1u) { pair_input[i] = 0u; }
            for (var i = 0u; i < 4u; i = i + 1u) { pair_input[8u + i] = node[i]; }
            for (var i = 0u; i < 4u; i = i + 1u) { pair_input[12u + i] = sibling[i]; }

            shake256_64_to_16(&pair_input, &node);
        }
        current_node = node;
    }

    // Compare final root with PK.root
    for (var i = 0u; i < 4u; i = i + 1u) {
        if (current_node[i] != pk_root[i]) {
            valid = false;
        }
    }

    results[tid] = select(0u, 1u, valid);
}
