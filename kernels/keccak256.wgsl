// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Keccak-256 (Ethereum variant) compute shader in WGSL.
//
// One thread per hash. Each thread reads its input descriptor, absorbs the
// data through Keccak-f[1600], and writes a 32-byte digest.
//
// Padding: 0x01 || 0x00...0x00 || 0x80 (Keccak, NOT SHA-3's 0x06)

struct HashInput {
    offset: u32,
    length: u32,
}

@group(0) @binding(0) var<storage, read> inputs: array<HashInput>;
@group(0) @binding(1) var<storage, read> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputs: array<u32>;

// Round constants for Keccak-f[1600] split into lo/hi u32 pairs.
// WGSL has no native u64, so we emulate with vec2<u32> (lo, hi).

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
    10u,  7u, 11u, 17u, 18u,  3u,  5u, 16u,  8u, 21u, 24u,  4u,
    15u, 23u, 19u, 13u, 12u,  2u, 20u, 14u, 22u,  9u,  6u,  1u
);

const RHO_OFFSETS = array<u32, 24>(
     1u,  3u,  6u, 10u, 15u, 21u, 28u, 36u, 45u, 55u,  2u, 14u,
    27u, 41u, 56u,  8u, 25u, 43u, 62u, 18u, 39u, 61u, 20u, 44u
);

// u64 emulation: each lane is state_lo[i], state_hi[i]
var<private> st_lo: array<u32, 25>;
var<private> st_hi: array<u32, 25>;

fn xor64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    return vec2<u32>(a_lo ^ b_lo, a_hi ^ b_hi);
}

fn rotl64(lo: u32, hi: u32, n: u32) -> vec2<u32> {
    if (n == 0u) { return vec2<u32>(lo, hi); }
    if (n == 32u) { return vec2<u32>(hi, lo); }
    if (n < 32u) {
        let r_lo = (lo << n) | (hi >> (32u - n));
        let r_hi = (hi << n) | (lo >> (32u - n));
        return vec2<u32>(r_lo, r_hi);
    }
    let m = n - 32u;
    let r_lo = (hi << m) | (lo >> (32u - m));
    let r_hi = (lo << m) | (hi >> (32u - m));
    return vec2<u32>(r_lo, r_hi);
}

fn and_not64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    return vec2<u32>((~a_lo) & b_lo, (~a_hi) & b_hi);
}

fn keccak_f() {
    for (var round = 0u; round < 24u; round = round + 1u) {
        // Theta
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

        // Rho + Pi
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

        // Chi
        for (var y = 0u; y < 5u; y = y + 1u) {
            var row_lo: array<u32, 5>;
            var row_hi: array<u32, 5>;
            for (var x = 0u; x < 5u; x = x + 1u) {
                row_lo[x] = st_lo[x + 5u * y];
                row_hi[x] = st_hi[x + 5u * y];
            }
            for (var x = 0u; x < 5u; x = x + 1u) {
                let an = and_not64(row_lo[(x+1u) % 5u], row_hi[(x+1u) % 5u],
                                   row_lo[(x+2u) % 5u], row_hi[(x+2u) % 5u]);
                st_lo[x + 5u * y] = row_lo[x] ^ an.x;
                st_hi[x + 5u * y] = row_hi[x] ^ an.y;
            }
        }

        // Iota
        st_lo[0u] = st_lo[0u] ^ RC_LO[round];
        st_hi[0u] = st_hi[0u] ^ RC_HI[round];
    }
}

// Read a byte from the data buffer (packed as u32 array, little-endian)
fn read_byte(byte_offset: u32) -> u32 {
    let word_idx = byte_offset >> 2u;
    let byte_pos = byte_offset & 3u;
    return (data[word_idx] >> (byte_pos * 8u)) & 0xFFu;
}

// Write a byte to a u32 array position in outputs
fn write_output_byte(base_word: u32, byte_in_word: u32, val: u32) {
    // Atomic or on the output word would be ideal, but we build the full word
    // in private memory and write once per word instead.
}

@compute @workgroup_size(64)
fn keccak256_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let inp = inputs[tid];
    let offset = inp.offset;
    let len = inp.length;
    let rate = 136u;

    // Zero state
    for (var i = 0u; i < 25u; i = i + 1u) {
        st_lo[i] = 0u;
        st_hi[i] = 0u;
    }

    // Absorb full blocks
    var absorbed = 0u;
    for (; absorbed + rate <= len; absorbed = absorbed + rate) {
        for (var w = 0u; w < 17u; w = w + 1u) {  // rate/8 = 17 words (64-bit)
            var lane_lo = 0u;
            var lane_hi = 0u;
            for (var b = 0u; b < 4u; b = b + 1u) {
                lane_lo = lane_lo | (read_byte(offset + absorbed + w * 8u + b) << (b * 8u));
            }
            for (var b = 0u; b < 4u; b = b + 1u) {
                lane_hi = lane_hi | (read_byte(offset + absorbed + w * 8u + 4u + b) << (b * 8u));
            }
            st_lo[w] = st_lo[w] ^ lane_lo;
            st_hi[w] = st_hi[w] ^ lane_hi;
        }
        keccak_f();
    }

    // Final block with padding
    // Build padded block in private memory
    var padded_lo: array<u32, 17>;
    var padded_hi: array<u32, 17>;
    for (var w = 0u; w < 17u; w = w + 1u) {
        padded_lo[w] = 0u;
        padded_hi[w] = 0u;
    }

    let remaining = len - absorbed;
    // Copy remaining bytes into padded block
    for (var i = 0u; i < remaining; i = i + 1u) {
        let byte_val = read_byte(offset + absorbed + i);
        let word_in_block = i >> 3u;  // which 64-bit word
        let byte_in_word = i & 7u;
        if (byte_in_word < 4u) {
            padded_lo[word_in_block] = padded_lo[word_in_block] | (byte_val << (byte_in_word * 8u));
        } else {
            padded_hi[word_in_block] = padded_hi[word_in_block] | (byte_val << ((byte_in_word - 4u) * 8u));
        }
    }

    // Keccak padding: byte[remaining] |= 0x01, byte[rate-1] |= 0x80
    let pad_word = remaining >> 3u;
    let pad_byte = remaining & 7u;
    if (pad_byte < 4u) {
        padded_lo[pad_word] = padded_lo[pad_word] | (0x01u << (pad_byte * 8u));
    } else {
        padded_hi[pad_word] = padded_hi[pad_word] | (0x01u << ((pad_byte - 4u) * 8u));
    }

    // Last byte of rate: byte[135] |= 0x80
    // 135 / 8 = 16 (word index), 135 % 8 = 7 (byte 7 => hi word, byte 3)
    padded_hi[16u] = padded_hi[16u] | (0x80u << (3u * 8u));

    // XOR padded block into state
    for (var w = 0u; w < 17u; w = w + 1u) {
        st_lo[w] = st_lo[w] ^ padded_lo[w];
        st_hi[w] = st_hi[w] ^ padded_hi[w];
    }
    keccak_f();

    // Squeeze: first 32 bytes = first 4 lanes (each lane = 8 bytes)
    let out_base = tid * 8u;  // 8 u32 words = 32 bytes
    for (var w = 0u; w < 4u; w = w + 1u) {
        outputs[out_base + w * 2u] = st_lo[w];
        outputs[out_base + w * 2u + 1u] = st_hi[w];
    }
}
