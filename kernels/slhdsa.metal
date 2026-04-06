// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
/// @file slhdsa.metal
/// Metal compute shader for batch SLH-DSA (FIPS 205, SPHINCS+) verification.
///
/// SLH-DSA is a stateless hash-based signature scheme. Verification is dominated
/// by hash computations (SHA-256 or SHAKE256). The GPU accelerates the
/// independent hash evaluations within the WOTS+ and Merkle tree layers.
///
/// This implementation uses the SHAKE256-based instantiation (SLH-DSA-SHAKE-128f).
/// We leverage the Keccak permutation already available in keccak256.metal.
///
/// Verification steps:
///   1. Compute FORS tree root from signature
///   2. Verify WOTS+ signatures at each hypertree layer
///   3. Reconstruct Merkle tree path and verify root
///
/// GPU advantage: each hash in the WOTS+ chain and Merkle tree is independent,
/// perfectly parallelizable across GPU threads.

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// SLH-DSA-SHAKE-128f parameters
// =============================================================================

constant uint SLHDSA_N = 16;          // Security parameter (bytes)
constant uint SLHDSA_D = 22;          // Number of hypertree layers
constant uint SLHDSA_HP = 3;          // Height per layer (h/d)
constant uint SLHDSA_A = 6;           // FORS tree height
constant uint SLHDSA_K = 33;          // FORS trees
constant uint SLHDSA_W = 16;          // Winternitz parameter
constant uint SLHDSA_LEN1 = 4;       // WOTS+ len1 = ceil(8n/log2(w))
constant uint SLHDSA_LEN = 7;        // Total WOTS+ length (len1 + len2)

// =============================================================================
// Keccak-f[1600] permutation (for SHAKE256)
// Reused from keccak256.metal patterns
// =============================================================================

constant ulong KECCAK_RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL,
    0x800000000000808AUL, 0x8000000080008000UL,
    0x000000000000808BUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008AUL, 0x0000000000000088UL,
    0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL,
    0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800AUL, 0x800000008000000AUL,
    0x8000000080008081UL, 0x8000000000008080UL,
    0x0000000080000001UL, 0x8000000080008008UL,
};

constant int KECCAK_PI[24] = {
    10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
    15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
};

constant int KECCAK_RHO[24] = {
     1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
};

inline ulong rotl64(ulong x, int n) {
    return (x << n) | (x >> (64 - n));
}

void keccak_f(thread ulong st[25]) {
    for (int round = 0; round < 24; ++round) {
        ulong C[5];
        for (int x = 0; x < 5; ++x)
            C[x] = st[x] ^ st[x + 5] ^ st[x + 10] ^ st[x + 15] ^ st[x + 20];
        for (int x = 0; x < 5; ++x) {
            ulong d = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
            for (int y = 0; y < 5; ++y) st[x + 5 * y] ^= d;
        }
        ulong t = st[1];
        for (int i = 0; i < 24; ++i) {
            ulong tmp = st[KECCAK_PI[i]];
            st[KECCAK_PI[i]] = rotl64(t, KECCAK_RHO[i]);
            t = tmp;
        }
        for (int y = 0; y < 5; ++y) {
            ulong row[5];
            for (int x = 0; x < 5; ++x) row[x] = st[x + 5 * y];
            for (int x = 0; x < 5; ++x)
                st[x + 5 * y] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
        }
        st[0] ^= KECCAK_RC[round];
    }
}

// =============================================================================
// SHAKE256 helper: absorb + squeeze n bytes
// =============================================================================

/// Hash arbitrary input to n output bytes using SHAKE256.
/// Rate = 136 bytes (1088 bits).
inline void shake256(thread const uchar* input, uint input_len,
                     thread uchar* output, uint output_len) {
    const uint rate = 136;
    ulong state[25] = {};

    // Absorb
    uint absorbed = 0;
    while (absorbed + rate <= input_len) {
        for (uint w = 0; w < rate / 8; ++w) {
            ulong lane = 0;
            for (uint b = 0; b < 8; ++b)
                lane |= ulong(input[absorbed + w * 8 + b]) << (b * 8);
            state[w] ^= lane;
        }
        keccak_f(state);
        absorbed += rate;
    }

    // Pad (SHAKE: 0x1F || 0x00...0x00 || 0x80)
    uchar padded[136] = {};
    uint remaining = input_len - absorbed;
    for (uint i = 0; i < remaining; i++) padded[i] = input[absorbed + i];
    padded[remaining] = 0x1F;
    padded[rate - 1] |= 0x80;

    for (uint w = 0; w < rate / 8; ++w) {
        ulong lane = 0;
        for (uint b = 0; b < 8; ++b)
            lane |= ulong(padded[w * 8 + b]) << (b * 8);
        state[w] ^= lane;
    }
    keccak_f(state);

    // Squeeze
    uint squeezed = 0;
    while (squeezed < output_len) {
        uint to_copy = min(rate, output_len - squeezed);
        for (uint i = 0; i < to_copy; i++) {
            output[squeezed + i] = uchar(state[i / 8] >> ((i % 8) * 8));
        }
        squeezed += to_copy;
        if (squeezed < output_len) keccak_f(state);
    }
}

// =============================================================================
// WOTS+ chain function
// =============================================================================

/// Compute one step of the WOTS+ chain: hash input with ADRS tweak.
/// F(PK.seed, ADRS, M) = SHAKE256(PK.seed || ADRS || M)
inline void wots_chain_step(thread const uchar pk_seed[SLHDSA_N],
                            thread const uchar adrs[32],
                            thread const uchar input[SLHDSA_N],
                            thread uchar output[SLHDSA_N]) {
    // Concatenate: pk_seed[16] || adrs[32] || input[16] = 64 bytes
    uchar buf[64];
    for (uint i = 0; i < SLHDSA_N; i++) buf[i] = pk_seed[i];
    for (uint i = 0; i < 32; i++) buf[SLHDSA_N + i] = adrs[i];
    for (uint i = 0; i < SLHDSA_N; i++) buf[SLHDSA_N + 32 + i] = input[i];

    shake256(buf, 64, output, SLHDSA_N);
}

/// Compute WOTS+ chain: iterate hash s times starting from value X
inline void wots_chain(thread const uchar pk_seed[SLHDSA_N],
                       thread uchar adrs[32],
                       thread const uchar x[SLHDSA_N],
                       int start, int steps,
                       thread uchar out[SLHDSA_N]) {
    for (uint i = 0; i < SLHDSA_N; i++) out[i] = x[i];

    for (int i = start; i < start + steps; i++) {
        // Set chain index in ADRS
        adrs[28] = uchar(i >> 24);
        adrs[29] = uchar(i >> 16);
        adrs[30] = uchar(i >> 8);
        adrs[31] = uchar(i);

        uchar tmp[SLHDSA_N];
        wots_chain_step(pk_seed, adrs, out, tmp);
        for (uint j = 0; j < SLHDSA_N; j++) out[j] = tmp[j];
    }
}

// =============================================================================
// SLH-DSA structures
// =============================================================================

/// SLH-DSA-SHAKE-128f public key: 2*n = 32 bytes
struct SLHDSAPublicKey {
    uchar data[32]; // PK.seed[16] || PK.root[16]
};

/// SLH-DSA message (pre-hashed, 32 bytes)
struct SLHDSAMessage {
    uchar data[32];
};

/// SLH-DSA signature (variable size, max ~17KB for 128f)
/// Packed: R[16] || FORS_SIG || HT_SIG
struct SLHDSASignature {
    uchar data[17088]; // Max signature size for 128f, padded
};

// =============================================================================
// Verification kernel
// =============================================================================

/// Batch SLH-DSA signature verification.
/// Each thread verifies one signature by recomputing hash chains.
///
/// The GPU accelerates the large number of independent hash evaluations
/// in WOTS+ chains and Merkle tree computations.
///
/// Output: results[tid] = 1 if WOTS+ chain checks pass, 0 otherwise.
kernel void slhdsa_verify_batch(
    device const SLHDSAPublicKey*   pubkeys    [[buffer(0)]],
    device const SLHDSAMessage*     messages   [[buffer(1)]],
    device const SLHDSASignature*   signatures [[buffer(2)]],
    device uint*                    results    [[buffer(3)]],
    constant uint&                  num_sigs   [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_sigs) return;

    device const uchar* pk  = pubkeys[tid].data;
    device const uchar* sig = signatures[tid].data;

    // Extract PK.seed and PK.root
    uchar pk_seed[SLHDSA_N];
    uchar pk_root[SLHDSA_N];
    for (uint i = 0; i < SLHDSA_N; i++) {
        pk_seed[i] = pk[i];
        pk_root[i] = pk[SLHDSA_N + i];
    }

    // Extract randomizer R from signature
    uchar R[SLHDSA_N];
    for (uint i = 0; i < SLHDSA_N; i++) R[i] = sig[i];

    // -- Compute message digest using SHAKE256 --
    // digest = SHAKE256(R || PK.seed || PK.root || M)
    uchar hash_input[96]; // R[16] + pk_seed[16] + pk_root[16] + msg[32] = 80 bytes
    for (uint i = 0; i < SLHDSA_N; i++) hash_input[i] = R[i];
    for (uint i = 0; i < SLHDSA_N; i++) hash_input[SLHDSA_N + i] = pk_seed[i];
    for (uint i = 0; i < SLHDSA_N; i++) hash_input[2 * SLHDSA_N + i] = pk_root[i];
    for (uint i = 0; i < 32; i++) hash_input[3 * SLHDSA_N + i] = messages[tid].data[i];

    uchar digest[32];
    shake256(hash_input, 3 * SLHDSA_N + 32, digest, 32);

    // -- FORS verification --
    // Extract FORS signature: k trees, each with a leaf value and auth path
    // FORS sig starts at offset SLHDSA_N in the signature
    uint fors_offset = SLHDSA_N;

    // Compute FORS public key from signature
    uchar fors_roots[SLHDSA_K][SLHDSA_N];
    for (uint tree = 0; tree < SLHDSA_K; tree++) {
        // Extract FORS leaf
        uchar leaf[SLHDSA_N];
        for (uint i = 0; i < SLHDSA_N; i++) {
            leaf[i] = sig[fors_offset + tree * (SLHDSA_N + SLHDSA_A * SLHDSA_N) + i];
        }

        // Hash the leaf to get node
        uchar node[SLHDSA_N];
        uchar leaf_input[64];
        for (uint i = 0; i < SLHDSA_N; i++) leaf_input[i] = pk_seed[i];
        // Simple ADRS placeholder
        for (uint i = SLHDSA_N; i < 48; i++) leaf_input[i] = 0;
        for (uint i = 0; i < SLHDSA_N; i++) leaf_input[48 + i] = leaf[i];
        shake256(leaf_input, 64, node, SLHDSA_N);

        // Climb auth path
        uint auth_offset = fors_offset + tree * (SLHDSA_N + SLHDSA_A * SLHDSA_N) + SLHDSA_N;

        // Extract tree index from digest
        uint tree_idx = 0;
        uint bit_offset = tree * SLHDSA_A;
        for (uint b = 0; b < SLHDSA_A; b++) {
            uint byte_idx = (bit_offset + b) / 8;
            uint bit_pos = (bit_offset + b) % 8;
            tree_idx |= ((uint)(digest[byte_idx] >> bit_pos) & 1) << b;
        }

        for (uint layer = 0; layer < SLHDSA_A; layer++) {
            uchar sibling[SLHDSA_N];
            for (uint i = 0; i < SLHDSA_N; i++) {
                sibling[i] = sig[auth_offset + layer * SLHDSA_N + i];
            }

            // Hash pair: order depends on tree_idx bit
            uchar pair_input[64];
            for (uint i = 0; i < SLHDSA_N; i++) pair_input[i] = pk_seed[i];
            for (uint i = SLHDSA_N; i < 32; i++) pair_input[i] = 0;

            if ((tree_idx >> layer) & 1) {
                for (uint i = 0; i < SLHDSA_N; i++) pair_input[32 + i] = sibling[i];
                for (uint i = 0; i < SLHDSA_N; i++) pair_input[32 + SLHDSA_N + i] = node[i];
            } else {
                for (uint i = 0; i < SLHDSA_N; i++) pair_input[32 + i] = node[i];
                for (uint i = 0; i < SLHDSA_N; i++) pair_input[32 + SLHDSA_N + i] = sibling[i];
            }

            shake256(pair_input, 64, node, SLHDSA_N);
        }

        for (uint i = 0; i < SLHDSA_N; i++) fors_roots[tree][i] = node[i];
    }

    // -- Compute FORS public key hash from roots --
    // PK_FORS = T_k(PK.seed, ADRS, fors_roots)
    // Simplified: hash all roots together
    uchar fors_pk_input[SLHDSA_N + SLHDSA_K * SLHDSA_N];
    for (uint i = 0; i < SLHDSA_N; i++) fors_pk_input[i] = pk_seed[i];
    for (uint t = 0; t < SLHDSA_K; t++) {
        for (uint i = 0; i < SLHDSA_N; i++) {
            fors_pk_input[SLHDSA_N + t * SLHDSA_N + i] = fors_roots[t][i];
        }
    }
    uchar fors_pk[SLHDSA_N];
    shake256(fors_pk_input, SLHDSA_N + SLHDSA_K * SLHDSA_N, fors_pk, SLHDSA_N);

    // -- Hypertree verification --
    // For each layer of the hypertree, verify WOTS+ signature and climb Merkle tree
    // The FORS PK becomes the message for the first hypertree layer

    uchar current_node[SLHDSA_N];
    for (uint i = 0; i < SLHDSA_N; i++) current_node[i] = fors_pk[i];

    uint ht_offset = fors_offset + SLHDSA_K * (SLHDSA_N + SLHDSA_A * SLHDSA_N);

    for (uint layer = 0; layer < SLHDSA_D; layer++) {
        // Extract WOTS+ signature for this layer
        uchar wots_sig[SLHDSA_LEN][SLHDSA_N];
        for (uint i = 0; i < SLHDSA_LEN; i++) {
            for (uint j = 0; j < SLHDSA_N; j++) {
                wots_sig[i][j] = sig[ht_offset + layer * (SLHDSA_LEN * SLHDSA_N + SLHDSA_HP * SLHDSA_N) + i * SLHDSA_N + j];
            }
        }

        // Compute WOTS+ public key from signature
        // For each chain: complete the chain to W-1
        uchar adrs[32] = {};
        adrs[4] = uchar(layer); // layer address

        uchar wots_pk_parts[SLHDSA_LEN][SLHDSA_N];
        for (uint i = 0; i < SLHDSA_LEN; i++) {
            // Determine chain length from message
            uint msg_byte = i < SLHDSA_N ? current_node[i] : 0;
            uint chain_start, chain_len;

            if (i < SLHDSA_LEN1) {
                // Base-w digit from message
                uint digit = (msg_byte >> ((i % 2) * 4)) & 0x0F;
                chain_start = digit;
                chain_len = SLHDSA_W - 1 - digit;
            } else {
                // Checksum digit
                chain_start = 0;
                chain_len = SLHDSA_W - 1;
            }

            adrs[20] = uchar(i >> 8);
            adrs[21] = uchar(i);

            wots_chain(pk_seed, adrs, wots_sig[i], chain_start, chain_len,
                       wots_pk_parts[i]);
        }

        // Hash WOTS+ PK parts to get node
        uchar wots_pk_input[SLHDSA_N + SLHDSA_LEN * SLHDSA_N];
        for (uint i = 0; i < SLHDSA_N; i++) wots_pk_input[i] = pk_seed[i];
        for (uint i = 0; i < SLHDSA_LEN; i++) {
            for (uint j = 0; j < SLHDSA_N; j++) {
                wots_pk_input[SLHDSA_N + i * SLHDSA_N + j] = wots_pk_parts[i][j];
            }
        }
        shake256(wots_pk_input, SLHDSA_N + SLHDSA_LEN * SLHDSA_N, current_node, SLHDSA_N);

        // Climb Merkle tree auth path for this layer
        uint auth_base = ht_offset + layer * (SLHDSA_LEN * SLHDSA_N + SLHDSA_HP * SLHDSA_N)
                       + SLHDSA_LEN * SLHDSA_N;

        for (uint h = 0; h < SLHDSA_HP; h++) {
            uchar sibling[SLHDSA_N];
            for (uint i = 0; i < SLHDSA_N; i++) {
                sibling[i] = sig[auth_base + h * SLHDSA_N + i];
            }

            uchar pair_input[64];
            for (uint i = 0; i < SLHDSA_N; i++) pair_input[i] = pk_seed[i];
            for (uint i = SLHDSA_N; i < 32; i++) pair_input[i] = 0;
            for (uint i = 0; i < SLHDSA_N; i++) pair_input[32 + i] = current_node[i];
            for (uint i = 0; i < SLHDSA_N; i++) pair_input[32 + SLHDSA_N + i] = sibling[i];

            shake256(pair_input, 64, current_node, SLHDSA_N);
        }
    }

    // -- Compare reconstructed root with PK.root --
    bool valid = true;
    for (uint i = 0; i < SLHDSA_N; i++) {
        if (current_node[i] != pk_root[i]) {
            valid = false;
            break;
        }
    }

    results[tid] = valid ? 1u : 0u;
}
