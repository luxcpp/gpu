// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// BLAKE3 hash compute shader in WGSL.
//
// One thread per hash. Each thread reads its input descriptor, processes
// chunks through the BLAKE3 compression function (7 rounds), and writes
// a 32-byte digest.
//
// BLAKE3 is tree-based and parallelizable by design.

struct HashInput {
    offset: u32,
    length: u32,
}

@group(0) @binding(0) var<storage, read> inputs: array<HashInput>;
@group(0) @binding(1) var<storage, read> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputs: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // params.x = num_inputs

const IV = array<u32, 8>(
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
);

const CHUNK_START: u32 = 1u;
const CHUNK_END: u32   = 2u;
const ROOT: u32        = 8u;

const MSG_PERM = array<u32, 16>(
    2u, 6u, 3u, 10u, 7u, 0u, 4u, 13u, 1u, 11u, 12u, 5u, 9u, 14u, 15u, 8u
);

fn rotr32(x: u32, n: u32) -> u32 {
    return (x >> n) | (x << (32u - n));
}

fn blake3_g(state: ptr<function, array<u32, 16>>, a: u32, b: u32, c: u32, d: u32, mx: u32, my: u32) {
    (*state)[a] = (*state)[a] + (*state)[b] + mx;
    (*state)[d] = rotr32((*state)[d] ^ (*state)[a], 16u);
    (*state)[c] = (*state)[c] + (*state)[d];
    (*state)[b] = rotr32((*state)[b] ^ (*state)[c], 12u);
    (*state)[a] = (*state)[a] + (*state)[b] + my;
    (*state)[d] = rotr32((*state)[d] ^ (*state)[a], 8u);
    (*state)[c] = (*state)[c] + (*state)[d];
    (*state)[b] = rotr32((*state)[b] ^ (*state)[c], 7u);
}

fn blake3_round(state: ptr<function, array<u32, 16>>, m: ptr<function, array<u32, 16>>) {
    blake3_g(state, 0u, 4u, 8u, 12u, (*m)[0u], (*m)[1u]);
    blake3_g(state, 1u, 5u, 9u, 13u, (*m)[2u], (*m)[3u]);
    blake3_g(state, 2u, 6u, 10u, 14u, (*m)[4u], (*m)[5u]);
    blake3_g(state, 3u, 7u, 11u, 15u, (*m)[6u], (*m)[7u]);
    blake3_g(state, 0u, 5u, 10u, 15u, (*m)[8u], (*m)[9u]);
    blake3_g(state, 1u, 6u, 11u, 12u, (*m)[10u], (*m)[11u]);
    blake3_g(state, 2u, 7u, 8u, 13u, (*m)[12u], (*m)[13u]);
    blake3_g(state, 3u, 4u, 9u, 14u, (*m)[14u], (*m)[15u]);
}

fn read_byte(byte_offset: u32) -> u32 {
    let word_idx = byte_offset >> 2u;
    let byte_pos = byte_offset & 3u;
    return (data[word_idx] >> (byte_pos * 8u)) & 0xFFu;
}

fn blake3_compress(cv: ptr<function, array<u32, 8>>,
                   block: ptr<function, array<u32, 16>>,
                   counter: u32,
                   block_len: u32,
                   flags: u32,
                   out: ptr<function, array<u32, 8>>) {
    var state: array<u32, 16>;
    state[0u] = (*cv)[0u]; state[1u] = (*cv)[1u]; state[2u] = (*cv)[2u]; state[3u] = (*cv)[3u];
    state[4u] = (*cv)[4u]; state[5u] = (*cv)[5u]; state[6u] = (*cv)[6u]; state[7u] = (*cv)[7u];
    state[8u] = IV[0]; state[9u] = IV[1]; state[10u] = IV[2]; state[11u] = IV[3];
    state[12u] = counter;
    state[13u] = 0u;
    state[14u] = block_len;
    state[15u] = flags;

    var m: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        m[i] = (*block)[i];
    }

    for (var round = 0u; round < 7u; round = round + 1u) {
        blake3_round(&state, &m);
        var tmp: array<u32, 16>;
        for (var i = 0u; i < 16u; i = i + 1u) {
            tmp[i] = m[MSG_PERM[i]];
        }
        m = tmp;
    }

    for (var i = 0u; i < 8u; i = i + 1u) {
        (*out)[i] = state[i] ^ state[i + 8u];
    }
}

@compute @workgroup_size(64)
fn blake3_hash_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.x) { return; }

    let inp = inputs[tid];
    let offset = inp.offset;
    let len = inp.length;

    var cv: array<u32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) { cv[i] = IV[i]; }

    var remaining = len;
    var pos = 0u;
    var block_idx = 0u;

    loop {
        if (remaining == 0u && block_idx > 0u) { break; }

        var block: array<u32, 16>;
        for (var i = 0u; i < 16u; i = i + 1u) { block[i] = 0u; }

        var to_copy = remaining;
        if (to_copy > 64u) { to_copy = 64u; }

        // Load block bytes into u32 words (little-endian)
        for (var i = 0u; i < to_copy; i = i + 1u) {
            let byte_val = read_byte(offset + pos + i);
            let word_idx = i >> 2u;
            let byte_pos = i & 3u;
            block[word_idx] = block[word_idx] | (byte_val << (byte_pos * 8u));
        }

        var flags = 0u;
        if (block_idx == 0u) { flags = flags | CHUNK_START; }
        let is_last = (remaining <= 64u);
        if (is_last) { flags = flags | CHUNK_END | ROOT; }

        var out: array<u32, 8>;
        blake3_compress(&cv, &block, 0u, to_copy, flags, &out);

        if (is_last) {
            let out_base = tid * 8u;
            for (var i = 0u; i < 8u; i = i + 1u) {
                outputs[out_base + i] = out[i];
            }
            return;
        }

        cv = out;
        pos = pos + to_copy;
        remaining = remaining - to_copy;
        block_idx = block_idx + 1u;
    }
}
