// SLH-DSA (FIPS 205, SPHINCS+) batch verify -- CUDA implementation
// Matches slhdsa.metal output byte-for-byte
// One thread per signature verification

#include <cstdint>

#ifndef __CUDA_ARCH__
#define __device__
#define __global__
#define __shared__
struct dim3 { unsigned x, y, z; };
static dim3 blockIdx, blockDim, threadIdx;
#endif

// =============================================================================
// SLH-DSA-SHAKE-128f parameters
// =============================================================================

#define SLHDSA_N    16      // Security parameter (bytes)
#define SLHDSA_D    22      // Number of hypertree layers
#define SLHDSA_HP   3       // Height per layer (h/d)
#define SLHDSA_A    6       // FORS tree height
#define SLHDSA_K    33      // FORS trees
#define SLHDSA_W    16      // Winternitz parameter
#define SLHDSA_LEN1 4       // WOTS+ len1 = ceil(8n/log2(w))
#define SLHDSA_LEN  7       // Total WOTS+ length (len1 + len2)

// =============================================================================
// Keccak-f[1600] permutation (for SHAKE256)
// =============================================================================

__device__ static const uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

__device__ static const int KECCAK_PI[24] = {
    10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
    15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
};

__device__ static const int KECCAK_RHO[24] = {
     1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
};

__device__ static uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__ static void keccak_f(uint64_t st[25]) {
    for (int round = 0; round < 24; ++round) {
        uint64_t C[5];
        for (int x = 0; x < 5; ++x)
            C[x] = st[x] ^ st[x + 5] ^ st[x + 10] ^ st[x + 15] ^ st[x + 20];
        for (int x = 0; x < 5; ++x) {
            uint64_t d = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
            for (int y = 0; y < 5; ++y) st[x + 5 * y] ^= d;
        }
        uint64_t t = st[1];
        for (int i = 0; i < 24; ++i) {
            uint64_t tmp = st[KECCAK_PI[i]];
            st[KECCAK_PI[i]] = rotl64(t, KECCAK_RHO[i]);
            t = tmp;
        }
        for (int y = 0; y < 5; ++y) {
            uint64_t row[5];
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

__device__ static void shake256(const uint8_t* input, uint32_t input_len,
                                 uint8_t* output, uint32_t output_len) {
    const uint32_t rate = 136;
    uint64_t state[25];
    for (int i = 0; i < 25; i++) state[i] = 0;

    // Absorb
    uint32_t absorbed = 0;
    while (absorbed + rate <= input_len) {
        for (uint32_t w = 0; w < rate / 8; ++w) {
            uint64_t lane = 0;
            for (uint32_t b = 0; b < 8; ++b)
                lane |= (uint64_t)input[absorbed + w * 8 + b] << (b * 8);
            state[w] ^= lane;
        }
        keccak_f(state);
        absorbed += rate;
    }

    // Pad (SHAKE: 0x1F || 0x00...0x00 || 0x80)
    uint8_t padded[136];
    for (uint32_t i = 0; i < 136; i++) padded[i] = 0;
    uint32_t remaining = input_len - absorbed;
    for (uint32_t i = 0; i < remaining; i++) padded[i] = input[absorbed + i];
    padded[remaining] = 0x1F;
    padded[rate - 1] |= 0x80;

    for (uint32_t w = 0; w < rate / 8; ++w) {
        uint64_t lane = 0;
        for (uint32_t b = 0; b < 8; ++b)
            lane |= (uint64_t)padded[w * 8 + b] << (b * 8);
        state[w] ^= lane;
    }
    keccak_f(state);

    // Squeeze
    uint32_t squeezed = 0;
    while (squeezed < output_len) {
        uint32_t to_copy = output_len - squeezed;
        if (to_copy > rate) to_copy = rate;
        for (uint32_t i = 0; i < to_copy; i++) {
            output[squeezed + i] = (uint8_t)(state[i / 8] >> ((i % 8) * 8));
        }
        squeezed += to_copy;
        if (squeezed < output_len) keccak_f(state);
    }
}

// =============================================================================
// WOTS+ chain function
// =============================================================================

__device__ static void wots_chain_step(const uint8_t pk_seed[SLHDSA_N],
                                        const uint8_t adrs[32],
                                        const uint8_t input[SLHDSA_N],
                                        uint8_t output[SLHDSA_N]) {
    uint8_t buf[64];
    for (uint32_t i = 0; i < SLHDSA_N; i++) buf[i] = pk_seed[i];
    for (uint32_t i = 0; i < 32; i++) buf[SLHDSA_N + i] = adrs[i];
    for (uint32_t i = 0; i < SLHDSA_N; i++) buf[SLHDSA_N + 32 + i] = input[i];
    shake256(buf, 64, output, SLHDSA_N);
}

__device__ static void wots_chain(const uint8_t pk_seed[SLHDSA_N],
                                   uint8_t adrs[32],
                                   const uint8_t x[SLHDSA_N],
                                   int start, int steps,
                                   uint8_t out[SLHDSA_N]) {
    for (uint32_t i = 0; i < SLHDSA_N; i++) out[i] = x[i];

    for (int i = start; i < start + steps; i++) {
        adrs[28] = (uint8_t)(i >> 24);
        adrs[29] = (uint8_t)(i >> 16);
        adrs[30] = (uint8_t)(i >> 8);
        adrs[31] = (uint8_t)(i);

        uint8_t tmp[SLHDSA_N];
        wots_chain_step(pk_seed, adrs, out, tmp);
        for (uint32_t j = 0; j < SLHDSA_N; j++) out[j] = tmp[j];
    }
}

// =============================================================================
// SLH-DSA structures
// =============================================================================

struct SLHDSAPublicKey {
    uint8_t data[32]; // PK.seed[16] || PK.root[16]
};

struct SLHDSAMessage {
    uint8_t data[32];
};

struct SLHDSASignature {
    uint8_t data[17088]; // Max signature size for 128f, padded
};

// =============================================================================
// Verification kernel
// =============================================================================

extern "C" __global__ void slhdsa_verify_batch(
    const SLHDSAPublicKey*   __restrict__ pubkeys,
    const SLHDSAMessage*     __restrict__ messages,
    const SLHDSASignature*   __restrict__ signatures,
    uint32_t*                __restrict__ results,
    const uint32_t*          __restrict__ num_sigs_ptr)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t num_sigs = *num_sigs_ptr;
    if (tid >= num_sigs) return;

    const uint8_t* pk  = pubkeys[tid].data;
    const uint8_t* sig = signatures[tid].data;

    // Extract PK.seed and PK.root
    uint8_t pk_seed[SLHDSA_N];
    uint8_t pk_root[SLHDSA_N];
    for (uint32_t i = 0; i < SLHDSA_N; i++) {
        pk_seed[i] = pk[i];
        pk_root[i] = pk[SLHDSA_N + i];
    }

    // Extract randomizer R from signature
    uint8_t R[SLHDSA_N];
    for (uint32_t i = 0; i < SLHDSA_N; i++) R[i] = sig[i];

    // -- Compute message digest using SHAKE256 --
    // digest = SHAKE256(R || PK.seed || PK.root || M)
    uint8_t hash_input[96]; // R[16] + pk_seed[16] + pk_root[16] + msg[32] = 80 bytes
    for (uint32_t i = 0; i < SLHDSA_N; i++) hash_input[i] = R[i];
    for (uint32_t i = 0; i < SLHDSA_N; i++) hash_input[SLHDSA_N + i] = pk_seed[i];
    for (uint32_t i = 0; i < SLHDSA_N; i++) hash_input[2 * SLHDSA_N + i] = pk_root[i];
    for (uint32_t i = 0; i < 32; i++) hash_input[3 * SLHDSA_N + i] = messages[tid].data[i];

    uint8_t digest[32];
    shake256(hash_input, 3 * SLHDSA_N + 32, digest, 32);

    // -- FORS verification --
    uint32_t fors_offset = SLHDSA_N;

    uint8_t fors_roots[SLHDSA_K][SLHDSA_N];
    for (uint32_t tree = 0; tree < SLHDSA_K; tree++) {
        // Extract FORS leaf
        uint8_t leaf[SLHDSA_N];
        for (uint32_t i = 0; i < SLHDSA_N; i++) {
            leaf[i] = sig[fors_offset + tree * (SLHDSA_N + SLHDSA_A * SLHDSA_N) + i];
        }

        // Hash the leaf to get node
        uint8_t node[SLHDSA_N];
        uint8_t leaf_input[64];
        for (uint32_t i = 0; i < SLHDSA_N; i++) leaf_input[i] = pk_seed[i];
        for (uint32_t i = SLHDSA_N; i < 48; i++) leaf_input[i] = 0;
        for (uint32_t i = 0; i < SLHDSA_N; i++) leaf_input[48 + i] = leaf[i];
        shake256(leaf_input, 64, node, SLHDSA_N);

        // Climb auth path
        uint32_t auth_offset = fors_offset + tree * (SLHDSA_N + SLHDSA_A * SLHDSA_N) + SLHDSA_N;

        // Extract tree index from digest
        uint32_t tree_idx = 0;
        uint32_t bit_offset = tree * SLHDSA_A;
        for (uint32_t b = 0; b < SLHDSA_A; b++) {
            uint32_t byte_idx = (bit_offset + b) / 8;
            uint32_t bit_pos = (bit_offset + b) % 8;
            tree_idx |= ((uint32_t)(digest[byte_idx] >> bit_pos) & 1) << b;
        }

        for (uint32_t layer = 0; layer < SLHDSA_A; layer++) {
            uint8_t sibling[SLHDSA_N];
            for (uint32_t i = 0; i < SLHDSA_N; i++) {
                sibling[i] = sig[auth_offset + layer * SLHDSA_N + i];
            }

            uint8_t pair_input[64];
            for (uint32_t i = 0; i < SLHDSA_N; i++) pair_input[i] = pk_seed[i];
            for (uint32_t i = SLHDSA_N; i < 32; i++) pair_input[i] = 0;

            if ((tree_idx >> layer) & 1) {
                for (uint32_t i = 0; i < SLHDSA_N; i++) pair_input[32 + i] = sibling[i];
                for (uint32_t i = 0; i < SLHDSA_N; i++) pair_input[32 + SLHDSA_N + i] = node[i];
            } else {
                for (uint32_t i = 0; i < SLHDSA_N; i++) pair_input[32 + i] = node[i];
                for (uint32_t i = 0; i < SLHDSA_N; i++) pair_input[32 + SLHDSA_N + i] = sibling[i];
            }

            shake256(pair_input, 64, node, SLHDSA_N);
        }

        for (uint32_t i = 0; i < SLHDSA_N; i++) fors_roots[tree][i] = node[i];
    }

    // -- Compute FORS public key hash from roots --
    uint8_t fors_pk_input[SLHDSA_N + SLHDSA_K * SLHDSA_N];
    for (uint32_t i = 0; i < SLHDSA_N; i++) fors_pk_input[i] = pk_seed[i];
    for (uint32_t t = 0; t < SLHDSA_K; t++) {
        for (uint32_t i = 0; i < SLHDSA_N; i++) {
            fors_pk_input[SLHDSA_N + t * SLHDSA_N + i] = fors_roots[t][i];
        }
    }
    uint8_t fors_pk[SLHDSA_N];
    shake256(fors_pk_input, SLHDSA_N + SLHDSA_K * SLHDSA_N, fors_pk, SLHDSA_N);

    // -- Hypertree verification --
    uint8_t current_node[SLHDSA_N];
    for (uint32_t i = 0; i < SLHDSA_N; i++) current_node[i] = fors_pk[i];

    uint32_t ht_offset = fors_offset + SLHDSA_K * (SLHDSA_N + SLHDSA_A * SLHDSA_N);

    for (uint32_t layer = 0; layer < SLHDSA_D; layer++) {
        // Extract WOTS+ signature for this layer
        uint8_t wots_sig[SLHDSA_LEN][SLHDSA_N];
        for (uint32_t i = 0; i < SLHDSA_LEN; i++) {
            for (uint32_t j = 0; j < SLHDSA_N; j++) {
                wots_sig[i][j] = sig[ht_offset + layer * (SLHDSA_LEN * SLHDSA_N + SLHDSA_HP * SLHDSA_N) + i * SLHDSA_N + j];
            }
        }

        // Compute WOTS+ public key from signature
        uint8_t adrs[32];
        for (uint32_t i = 0; i < 32; i++) adrs[i] = 0;
        adrs[4] = (uint8_t)layer;

        uint8_t wots_pk_parts[SLHDSA_LEN][SLHDSA_N];
        for (uint32_t i = 0; i < SLHDSA_LEN; i++) {
            uint32_t msg_byte = i < SLHDSA_N ? current_node[i] : 0;
            uint32_t chain_start, chain_len;

            if (i < SLHDSA_LEN1) {
                uint32_t digit = (msg_byte >> ((i % 2) * 4)) & 0x0F;
                chain_start = digit;
                chain_len = SLHDSA_W - 1 - digit;
            } else {
                chain_start = 0;
                chain_len = SLHDSA_W - 1;
            }

            adrs[20] = (uint8_t)(i >> 8);
            adrs[21] = (uint8_t)(i);

            wots_chain(pk_seed, adrs, wots_sig[i], chain_start, chain_len,
                       wots_pk_parts[i]);
        }

        // Hash WOTS+ PK parts to get node
        uint8_t wots_pk_input[SLHDSA_N + SLHDSA_LEN * SLHDSA_N];
        for (uint32_t i = 0; i < SLHDSA_N; i++) wots_pk_input[i] = pk_seed[i];
        for (uint32_t i = 0; i < SLHDSA_LEN; i++) {
            for (uint32_t j = 0; j < SLHDSA_N; j++) {
                wots_pk_input[SLHDSA_N + i * SLHDSA_N + j] = wots_pk_parts[i][j];
            }
        }
        shake256(wots_pk_input, SLHDSA_N + SLHDSA_LEN * SLHDSA_N, current_node, SLHDSA_N);

        // Climb Merkle tree auth path for this layer
        uint32_t auth_base = ht_offset + layer * (SLHDSA_LEN * SLHDSA_N + SLHDSA_HP * SLHDSA_N)
                           + SLHDSA_LEN * SLHDSA_N;

        for (uint32_t h = 0; h < SLHDSA_HP; h++) {
            uint8_t sibling[SLHDSA_N];
            for (uint32_t i = 0; i < SLHDSA_N; i++) {
                sibling[i] = sig[auth_base + h * SLHDSA_N + i];
            }

            uint8_t pair_input[64];
            for (uint32_t i = 0; i < SLHDSA_N; i++) pair_input[i] = pk_seed[i];
            for (uint32_t i = SLHDSA_N; i < 32; i++) pair_input[i] = 0;
            for (uint32_t i = 0; i < SLHDSA_N; i++) pair_input[32 + i] = current_node[i];
            for (uint32_t i = 0; i < SLHDSA_N; i++) pair_input[32 + SLHDSA_N + i] = sibling[i];

            shake256(pair_input, 64, current_node, SLHDSA_N);
        }
    }

    // -- Compare reconstructed root with PK.root --
    bool valid = true;
    for (uint32_t i = 0; i < SLHDSA_N; i++) {
        if (current_node[i] != pk_root[i]) {
            valid = false;
            break;
        }
    }

    results[tid] = valid ? 1u : 0u;
}
