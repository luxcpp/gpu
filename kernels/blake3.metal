// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
/// @file blake3.metal
/// Metal compute shader for parallel BLAKE3 hashing.
///
/// BLAKE3 is a tree-based hash with 7 rounds per compression. Each 1024-byte
/// chunk is independently hashable, making it ideal for GPU parallelism.
///
/// Kernel: blake3_hash_batch
///   - Each thread hashes one input independently
///   - Supports inputs up to ~64KB (single chunk for simplicity; tree mode
///     for longer inputs would use chunk-level parallelism)
///   - Output: 32 bytes per input
///
/// Reference: https://github.com/BLAKE3-team/BLAKE3-spec

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// BLAKE3 constants
// =============================================================================

constant uint BLAKE3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

// Domain separation flags
constant uint BLAKE3_CHUNK_START = 1u;
constant uint BLAKE3_CHUNK_END   = 2u;
constant uint BLAKE3_ROOT        = 8u;

// Message word permutation (applied after each round)
constant uchar BLAKE3_MSG_PERM[16] = {
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8
};

// =============================================================================
// BLAKE3 quarter-round G function
// =============================================================================

inline uint rotr32(uint x, uint n) {
    return (x >> n) | (x << (32u - n));
}

inline void blake3_g(thread uint state[16], int a, int b, int c, int d,
                     uint mx, uint my) {
    state[a] = state[a] + state[b] + mx;
    state[d] = rotr32(state[d] ^ state[a], 16u);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 12u);
    state[a] = state[a] + state[b] + my;
    state[d] = rotr32(state[d] ^ state[a], 8u);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 7u);
}

// =============================================================================
// BLAKE3 round (column + diagonal)
// =============================================================================

inline void blake3_round(thread uint state[16], thread const uint m[16]) {
    // Columns
    blake3_g(state, 0, 4,  8, 12, m[0],  m[1]);
    blake3_g(state, 1, 5,  9, 13, m[2],  m[3]);
    blake3_g(state, 2, 6, 10, 14, m[4],  m[5]);
    blake3_g(state, 3, 7, 11, 15, m[6],  m[7]);
    // Diagonals
    blake3_g(state, 0, 5, 10, 15, m[8],  m[9]);
    blake3_g(state, 1, 6, 11, 12, m[10], m[11]);
    blake3_g(state, 2, 7,  8, 13, m[12], m[13]);
    blake3_g(state, 3, 4,  9, 14, m[14], m[15]);
}

// =============================================================================
// BLAKE3 compression function
// =============================================================================

/// Compress one 64-byte block.
/// cv: 8-word chaining value
/// block: 64-byte message block
/// counter: block counter within chunk
/// block_len: actual bytes in this block (0-64)
/// flags: domain separation flags
/// out: 8-word output chaining value
inline void blake3_compress(thread const uint cv[8],
                            thread const uchar block[64],
                            ulong counter,
                            uint block_len,
                            uint flags,
                            thread uint out[8]) {
    // Load message words (little-endian)
    uint m[16];
    for (int i = 0; i < 16; i++) {
        m[i] = uint(block[i * 4])
             | (uint(block[i * 4 + 1]) << 8)
             | (uint(block[i * 4 + 2]) << 16)
             | (uint(block[i * 4 + 3]) << 24);
    }

    uint state[16] = {
        cv[0], cv[1], cv[2], cv[3],
        cv[4], cv[5], cv[6], cv[7],
        BLAKE3_IV[0], BLAKE3_IV[1], BLAKE3_IV[2], BLAKE3_IV[3],
        uint(counter & 0xFFFFFFFFu),
        uint(counter >> 32),
        block_len,
        flags
    };

    // 7 rounds with message permutation
    for (int round = 0; round < 7; round++) {
        blake3_round(state, m);
        // Permute message words
        uint tmp[16];
        for (int i = 0; i < 16; i++) {
            tmp[i] = m[BLAKE3_MSG_PERM[i]];
        }
        for (int i = 0; i < 16; i++) m[i] = tmp[i];
    }

    // Output: state[0..7] XOR state[8..15]
    for (int i = 0; i < 8; i++) {
        out[i] = state[i] ^ state[i + 8];
    }
}

// =============================================================================
// Input descriptor
// =============================================================================

struct HashInput {
    uint offset;
    uint length;
};

// =============================================================================
// Kernel: blake3_hash_batch
// =============================================================================

/// Each thread hashes one input to a 32-byte BLAKE3 digest.
/// For inputs <= 1024 bytes, this is a single-chunk hash.
/// For inputs > 1024 bytes, we process multiple chunks and merge via tree mode.
kernel void blake3_hash_batch(
    device const HashInput* inputs  [[buffer(0)]],
    device const uchar*     data    [[buffer(1)]],
    device uchar*           outputs [[buffer(2)]],
    constant uint&          num_inputs [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_inputs) return;

    const uint offset = inputs[tid].offset;
    const uint len    = inputs[tid].length;

    // Process chunks (each chunk = 1024 bytes = 16 blocks of 64 bytes)
    const uint chunk_size = 1024;
    uint num_chunks = (len + chunk_size - 1) / chunk_size;
    if (num_chunks == 0) num_chunks = 1;

    // For single chunk (most common case), compute directly
    if (num_chunks == 1) {
        uint cv[8];
        for (int i = 0; i < 8; i++) cv[i] = BLAKE3_IV[i];

        uint remaining = len;
        uint pos = 0;
        uint block_idx = 0;

        while (remaining > 0 || block_idx == 0) {
            uchar block[64] = {};
            uint to_copy = (remaining > 64) ? 64 : remaining;
            for (uint i = 0; i < to_copy; i++) {
                block[i] = data[offset + pos + i];
            }

            uint flags = 0;
            if (block_idx == 0) flags |= BLAKE3_CHUNK_START;
            bool is_last = (remaining <= 64);
            if (is_last) flags |= BLAKE3_CHUNK_END | BLAKE3_ROOT;

            uint out[8];
            blake3_compress(cv, block, 0, to_copy, flags, out);

            if (is_last) {
                // Write final hash
                device uchar* dst = outputs + tid * 32;
                for (int i = 0; i < 8; i++) {
                    dst[i * 4]     = uchar(out[i] & 0xFF);
                    dst[i * 4 + 1] = uchar((out[i] >> 8) & 0xFF);
                    dst[i * 4 + 2] = uchar((out[i] >> 16) & 0xFF);
                    dst[i * 4 + 3] = uchar((out[i] >> 24) & 0xFF);
                }
                return;
            }

            for (int i = 0; i < 8; i++) cv[i] = out[i];
            pos += to_copy;
            remaining -= to_copy;
            block_idx++;
        }
    }

    // Multi-chunk: process each chunk independently, then tree-hash the CVs
    // Stack for tree hashing (max depth ~20 for 2^20 chunks)
    uint cv_stack[20][8];
    int stack_depth = 0;

    for (uint chunk = 0; chunk < num_chunks; chunk++) {
        uint cv[8];
        for (int i = 0; i < 8; i++) cv[i] = BLAKE3_IV[i];

        uint chunk_start = offset + chunk * chunk_size;
        uint chunk_len = (chunk == num_chunks - 1) ? (len - chunk * chunk_size) : chunk_size;
        uint remaining = chunk_len;
        uint pos = 0;
        uint block_idx = 0;

        // Process blocks within this chunk
        while (remaining > 0 || block_idx == 0) {
            uchar block[64] = {};
            uint to_copy = (remaining > 64) ? 64 : remaining;
            for (uint i = 0; i < to_copy; i++) {
                block[i] = data[chunk_start + pos + i];
            }

            uint flags = 0;
            if (block_idx == 0) flags |= BLAKE3_CHUNK_START;
            if (remaining <= 64) flags |= BLAKE3_CHUNK_END;

            uint out[8];
            blake3_compress(cv, block, chunk, to_copy, flags, out);

            if (remaining <= 64) {
                // Push chunk CV onto stack and merge
                for (int i = 0; i < 8; i++) cv_stack[stack_depth][i] = out[i];
                stack_depth++;

                // Merge pairs while we have complete pairs at same level
                while (stack_depth >= 2) {
                    // Check if this is the very last merge (root)
                    bool is_root = (chunk == num_chunks - 1) && (stack_depth == 2);

                    // Build parent block: left_cv[32] || right_cv[32]
                    uchar parent_block[64];
                    for (int i = 0; i < 8; i++) {
                        uint w = cv_stack[stack_depth - 2][i];
                        parent_block[i * 4]     = uchar(w & 0xFF);
                        parent_block[i * 4 + 1] = uchar((w >> 8) & 0xFF);
                        parent_block[i * 4 + 2] = uchar((w >> 16) & 0xFF);
                        parent_block[i * 4 + 3] = uchar((w >> 24) & 0xFF);
                    }
                    for (int i = 0; i < 8; i++) {
                        uint w = cv_stack[stack_depth - 1][i];
                        parent_block[32 + i * 4]     = uchar(w & 0xFF);
                        parent_block[32 + i * 4 + 1] = uchar((w >> 8) & 0xFF);
                        parent_block[32 + i * 4 + 2] = uchar((w >> 16) & 0xFF);
                        parent_block[32 + i * 4 + 3] = uchar((w >> 24) & 0xFF);
                    }

                    uint parent_cv[8];
                    for (int i = 0; i < 8; i++) parent_cv[i] = BLAKE3_IV[i];

                    uint parent_flags = 4u; // PARENT flag
                    if (is_root) parent_flags |= BLAKE3_ROOT;

                    uint parent_out[8];
                    blake3_compress(parent_cv, parent_block, 0, 64, parent_flags, parent_out);

                    stack_depth -= 2;

                    if (is_root) {
                        device uchar* dst = outputs + tid * 32;
                        for (int i = 0; i < 8; i++) {
                            dst[i * 4]     = uchar(parent_out[i] & 0xFF);
                            dst[i * 4 + 1] = uchar((parent_out[i] >> 8) & 0xFF);
                            dst[i * 4 + 2] = uchar((parent_out[i] >> 16) & 0xFF);
                            dst[i * 4 + 3] = uchar((parent_out[i] >> 24) & 0xFF);
                        }
                        return;
                    }

                    for (int i = 0; i < 8; i++) cv_stack[stack_depth][i] = parent_out[i];
                    stack_depth++;
                    break; // only merge one pair per chunk
                }
                break;
            }

            for (int i = 0; i < 8; i++) cv[i] = out[i];
            pos += to_copy;
            remaining -= to_copy;
            block_idx++;
        }
    }

    // Merge remaining stack entries (right-to-left)
    while (stack_depth >= 2) {
        bool is_root = (stack_depth == 2);
        uchar parent_block[64];
        for (int i = 0; i < 8; i++) {
            uint w = cv_stack[stack_depth - 2][i];
            parent_block[i * 4]     = uchar(w & 0xFF);
            parent_block[i * 4 + 1] = uchar((w >> 8) & 0xFF);
            parent_block[i * 4 + 2] = uchar((w >> 16) & 0xFF);
            parent_block[i * 4 + 3] = uchar((w >> 24) & 0xFF);
        }
        for (int i = 0; i < 8; i++) {
            uint w = cv_stack[stack_depth - 1][i];
            parent_block[32 + i * 4]     = uchar(w & 0xFF);
            parent_block[32 + i * 4 + 1] = uchar((w >> 8) & 0xFF);
            parent_block[32 + i * 4 + 2] = uchar((w >> 16) & 0xFF);
            parent_block[32 + i * 4 + 3] = uchar((w >> 24) & 0xFF);
        }

        uint parent_cv[8];
        for (int i = 0; i < 8; i++) parent_cv[i] = BLAKE3_IV[i];
        uint parent_flags = 4u;
        if (is_root) parent_flags |= BLAKE3_ROOT;

        uint parent_out[8];
        blake3_compress(parent_cv, parent_block, 0, 64, parent_flags, parent_out);

        stack_depth -= 2;
        if (is_root) {
            device uchar* dst = outputs + tid * 32;
            for (int i = 0; i < 8; i++) {
                dst[i * 4]     = uchar(parent_out[i] & 0xFF);
                dst[i * 4 + 1] = uchar((parent_out[i] >> 8) & 0xFF);
                dst[i * 4 + 2] = uchar((parent_out[i] >> 16) & 0xFF);
                dst[i * 4 + 3] = uchar((parent_out[i] >> 24) & 0xFF);
            }
            return;
        }
        for (int i = 0; i < 8; i++) cv_stack[stack_depth][i] = parent_out[i];
        stack_depth++;
    }
}
