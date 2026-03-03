// BLAKE3 batch hash — CUDA implementation
// Matches blake3.metal output byte-for-byte
// One thread per hash

#include <cstdint>

#ifndef __CUDA_ARCH__
#define __device__
#define __global__
#define __shared__
struct dim3 { unsigned x, y, z; };
static dim3 blockIdx, blockDim, threadIdx;
#endif

// =============================================================================
// BLAKE3 constants
// =============================================================================

__device__ static const uint32_t BLAKE3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

__device__ static const uint32_t BLAKE3_CHUNK_START = 1u;
__device__ static const uint32_t BLAKE3_CHUNK_END   = 2u;
__device__ static const uint32_t BLAKE3_ROOT        = 8u;

__device__ static const uint8_t BLAKE3_MSG_PERM[16] = {
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8
};

// =============================================================================
// BLAKE3 quarter-round G function
// =============================================================================

__device__ static inline uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}

__device__ static void blake3_g(uint32_t state[16], int a, int b, int c, int d,
                                uint32_t mx, uint32_t my) {
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

__device__ static void blake3_round(uint32_t state[16], const uint32_t m[16]) {
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

__device__ static void blake3_compress(const uint32_t cv[8],
                                       const uint8_t block[64],
                                       uint64_t counter,
                                       uint32_t block_len,
                                       uint32_t flags,
                                       uint32_t out[8]) {
    // Load message words (little-endian)
    uint32_t m[16];
    for (int i = 0; i < 16; i++) {
        m[i] = (uint32_t)block[i * 4]
             | ((uint32_t)block[i * 4 + 1] << 8)
             | ((uint32_t)block[i * 4 + 2] << 16)
             | ((uint32_t)block[i * 4 + 3] << 24);
    }

    uint32_t state[16] = {
        cv[0], cv[1], cv[2], cv[3],
        cv[4], cv[5], cv[6], cv[7],
        BLAKE3_IV[0], BLAKE3_IV[1], BLAKE3_IV[2], BLAKE3_IV[3],
        (uint32_t)(counter & 0xFFFFFFFFu),
        (uint32_t)(counter >> 32),
        block_len,
        flags
    };

    // 7 rounds with message permutation
    for (int round = 0; round < 7; round++) {
        blake3_round(state, m);
        // Permute message words
        uint32_t tmp[16];
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
    uint32_t offset;
    uint32_t length;
};

// =============================================================================
// Kernel: blake3_hash_batch
// =============================================================================

extern "C" __global__ void blake3_hash_batch(
    const HashInput* __restrict__ inputs,
    const uint8_t* __restrict__   data,
    uint8_t* __restrict__         outputs,
    const uint32_t                num_inputs)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_inputs) return;

    const uint32_t offset = inputs[tid].offset;
    const uint32_t len    = inputs[tid].length;

    // Process chunks (each chunk = 1024 bytes = 16 blocks of 64 bytes)
    const uint32_t chunk_size = 1024;
    uint32_t num_chunks = (len + chunk_size - 1) / chunk_size;
    if (num_chunks == 0) num_chunks = 1;

    // Single chunk (most common case)
    if (num_chunks == 1) {
        uint32_t cv[8];
        for (int i = 0; i < 8; i++) cv[i] = BLAKE3_IV[i];

        uint32_t remaining = len;
        uint32_t pos = 0;
        uint32_t block_idx = 0;

        while (remaining > 0 || block_idx == 0) {
            uint8_t block[64] = {};
            uint32_t to_copy = (remaining > 64) ? 64 : remaining;
            for (uint32_t i = 0; i < to_copy; i++) {
                block[i] = data[offset + pos + i];
            }

            uint32_t flags = 0;
            if (block_idx == 0) flags |= BLAKE3_CHUNK_START;
            bool is_last = (remaining <= 64);
            if (is_last) flags |= BLAKE3_CHUNK_END | BLAKE3_ROOT;

            uint32_t out[8];
            blake3_compress(cv, block, 0, to_copy, flags, out);

            if (is_last) {
                uint8_t* dst = outputs + tid * 32;
                for (int i = 0; i < 8; i++) {
                    dst[i * 4]     = (uint8_t)(out[i] & 0xFF);
                    dst[i * 4 + 1] = (uint8_t)((out[i] >> 8) & 0xFF);
                    dst[i * 4 + 2] = (uint8_t)((out[i] >> 16) & 0xFF);
                    dst[i * 4 + 3] = (uint8_t)((out[i] >> 24) & 0xFF);
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
    uint32_t cv_stack[20][8];
    int stack_depth = 0;

    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        uint32_t cv[8];
        for (int i = 0; i < 8; i++) cv[i] = BLAKE3_IV[i];

        uint32_t chunk_start = offset + chunk * chunk_size;
        uint32_t chunk_len = (chunk == num_chunks - 1) ? (len - chunk * chunk_size) : chunk_size;
        uint32_t remaining = chunk_len;
        uint32_t pos = 0;
        uint32_t block_idx = 0;

        while (remaining > 0 || block_idx == 0) {
            uint8_t block[64] = {};
            uint32_t to_copy = (remaining > 64) ? 64 : remaining;
            for (uint32_t i = 0; i < to_copy; i++) {
                block[i] = data[chunk_start + pos + i];
            }

            uint32_t flags = 0;
            if (block_idx == 0) flags |= BLAKE3_CHUNK_START;
            if (remaining <= 64) flags |= BLAKE3_CHUNK_END;

            uint32_t out[8];
            blake3_compress(cv, block, chunk, to_copy, flags, out);

            if (remaining <= 64) {
                for (int i = 0; i < 8; i++) cv_stack[stack_depth][i] = out[i];
                stack_depth++;

                while (stack_depth >= 2) {
                    bool is_root = (chunk == num_chunks - 1) && (stack_depth == 2);

                    uint8_t parent_block[64];
                    for (int i = 0; i < 8; i++) {
                        uint32_t w = cv_stack[stack_depth - 2][i];
                        parent_block[i * 4]     = (uint8_t)(w & 0xFF);
                        parent_block[i * 4 + 1] = (uint8_t)((w >> 8) & 0xFF);
                        parent_block[i * 4 + 2] = (uint8_t)((w >> 16) & 0xFF);
                        parent_block[i * 4 + 3] = (uint8_t)((w >> 24) & 0xFF);
                    }
                    for (int i = 0; i < 8; i++) {
                        uint32_t w = cv_stack[stack_depth - 1][i];
                        parent_block[32 + i * 4]     = (uint8_t)(w & 0xFF);
                        parent_block[32 + i * 4 + 1] = (uint8_t)((w >> 8) & 0xFF);
                        parent_block[32 + i * 4 + 2] = (uint8_t)((w >> 16) & 0xFF);
                        parent_block[32 + i * 4 + 3] = (uint8_t)((w >> 24) & 0xFF);
                    }

                    uint32_t parent_cv[8];
                    for (int i = 0; i < 8; i++) parent_cv[i] = BLAKE3_IV[i];

                    uint32_t parent_flags = 4u; // PARENT flag
                    if (is_root) parent_flags |= BLAKE3_ROOT;

                    uint32_t parent_out[8];
                    blake3_compress(parent_cv, parent_block, 0, 64, parent_flags, parent_out);

                    stack_depth -= 2;

                    if (is_root) {
                        uint8_t* dst = outputs + tid * 32;
                        for (int i = 0; i < 8; i++) {
                            dst[i * 4]     = (uint8_t)(parent_out[i] & 0xFF);
                            dst[i * 4 + 1] = (uint8_t)((parent_out[i] >> 8) & 0xFF);
                            dst[i * 4 + 2] = (uint8_t)((parent_out[i] >> 16) & 0xFF);
                            dst[i * 4 + 3] = (uint8_t)((parent_out[i] >> 24) & 0xFF);
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

    // Merge remaining stack entries
    while (stack_depth >= 2) {
        bool is_root = (stack_depth == 2);
        uint8_t parent_block[64];
        for (int i = 0; i < 8; i++) {
            uint32_t w = cv_stack[stack_depth - 2][i];
            parent_block[i * 4]     = (uint8_t)(w & 0xFF);
            parent_block[i * 4 + 1] = (uint8_t)((w >> 8) & 0xFF);
            parent_block[i * 4 + 2] = (uint8_t)((w >> 16) & 0xFF);
            parent_block[i * 4 + 3] = (uint8_t)((w >> 24) & 0xFF);
        }
        for (int i = 0; i < 8; i++) {
            uint32_t w = cv_stack[stack_depth - 1][i];
            parent_block[32 + i * 4]     = (uint8_t)(w & 0xFF);
            parent_block[32 + i * 4 + 1] = (uint8_t)((w >> 8) & 0xFF);
            parent_block[32 + i * 4 + 2] = (uint8_t)((w >> 16) & 0xFF);
            parent_block[32 + i * 4 + 3] = (uint8_t)((w >> 24) & 0xFF);
        }

        uint32_t parent_cv[8];
        for (int i = 0; i < 8; i++) parent_cv[i] = BLAKE3_IV[i];
        uint32_t parent_flags = 4u;
        if (is_root) parent_flags |= BLAKE3_ROOT;

        uint32_t parent_out[8];
        blake3_compress(parent_cv, parent_block, 0, 64, parent_flags, parent_out);

        stack_depth -= 2;
        if (is_root) {
            uint8_t* dst = outputs + tid * 32;
            for (int i = 0; i < 8; i++) {
                dst[i * 4]     = (uint8_t)(parent_out[i] & 0xFF);
                dst[i * 4 + 1] = (uint8_t)((parent_out[i] >> 8) & 0xFF);
                dst[i * 4 + 2] = (uint8_t)((parent_out[i] >> 16) & 0xFF);
                dst[i * 4 + 3] = (uint8_t)((parent_out[i] >> 24) & 0xFF);
            }
            return;
        }
        for (int i = 0; i < 8; i++) cv_stack[stack_depth][i] = parent_out[i];
        stack_depth++;
    }
}
