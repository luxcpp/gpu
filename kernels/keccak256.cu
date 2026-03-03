// Keccak-256 batch hashing — CUDA implementation
// Matches keccak256.metal output byte-for-byte
// One thread per hash

#include <cstdint>

#ifndef __CUDA_ARCH__
#define __device__
#define __global__
#define __shared__
struct dim3 { unsigned x, y, z; };
static dim3 blockIdx, blockDim, threadIdx;
#endif

__device__ static const uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ static const int ROTC[24] = {
    1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44
};

__device__ static const int PI[24] = {
    10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1
};

__device__ void keccak_f1600(uint64_t* state) {
    for (int round = 0; round < 24; round++) {
        // Theta
        uint64_t C[5], D[5];
        for (int x = 0; x < 5; x++)
            C[x] = state[x] ^ state[x+5] ^ state[x+10] ^ state[x+15] ^ state[x+20];
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x+4)%5] ^ ((C[(x+1)%5] << 1) | (C[(x+1)%5] >> 63));
            for (int y = 0; y < 25; y += 5)
                state[y+x] ^= D[x];
        }
        // Rho + Pi
        uint64_t t = state[1];
        for (int i = 0; i < 24; i++) {
            int j = PI[i];
            uint64_t tmp = state[j];
            state[j] = (t << ROTC[i]) | (t >> (64-ROTC[i]));
            t = tmp;
        }
        // Chi
        for (int y = 0; y < 25; y += 5) {
            uint64_t t0 = state[y], t1 = state[y+1], t2 = state[y+2],
                     t3 = state[y+3], t4 = state[y+4];
            state[y]   = t0 ^ (~t1 & t2);
            state[y+1] = t1 ^ (~t2 & t3);
            state[y+2] = t2 ^ (~t3 & t4);
            state[y+3] = t3 ^ (~t4 & t0);
            state[y+4] = t4 ^ (~t0 & t1);
        }
        // Iota
        state[0] ^= RC[round];
    }
}

extern "C" __global__ void keccak256_batch(
    const uint8_t* __restrict__ data,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ lengths,
    uint8_t* __restrict__ outputs,
    uint32_t num_inputs)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_inputs) return;

    uint64_t state[25] = {0};
    const uint8_t* input = data + offsets[tid];
    uint32_t len = lengths[tid];

    // Absorb (rate = 136 bytes = 17 uint64s)
    uint32_t pos = 0;
    while (pos + 136 <= len) {
        for (int i = 0; i < 17; i++) {
            uint64_t word = 0;
            for (int b = 0; b < 8; b++)
                word |= (uint64_t)input[pos + i*8 + b] << (b*8);
            state[i] ^= word;
        }
        keccak_f1600(state);
        pos += 136;
    }

    // Pad last block (Keccak padding: 0x01...0x80)
    uint8_t block[136] = {0};
    uint32_t rem = len - pos;
    for (uint32_t i = 0; i < rem; i++)
        block[i] = input[pos + i];
    block[rem] = 0x01;
    block[135] = 0x80;

    for (int i = 0; i < 17; i++) {
        uint64_t word = 0;
        for (int b = 0; b < 8; b++)
            word |= (uint64_t)block[i*8 + b] << (b*8);
        state[i] ^= word;
    }
    keccak_f1600(state);

    // Squeeze 32 bytes
    uint8_t* out = outputs + tid * 32;
    for (int i = 0; i < 4; i++)
        for (int b = 0; b < 8; b++)
            out[i*8 + b] = (state[i] >> (b*8)) & 0xFF;
}
