// Test all CUDA kernels in CPU mode (no nvcc required)
// Compiles .cu files as C++ with CUDA builtins stubbed out

#include <cstdio>
#include <cstdint>
#include <cstring>

// Stub CUDA builtins
struct dim3 { uint32_t x,y,z; };
static dim3 blockIdx{0,0,0}, blockDim{1,0,0}, threadIdx{0,0,0};
#define __global__
#define __device__
#define __shared__
#define __restrict__

// Test keccak256
#include "keccak256.cu"

static const uint8_t KECCAK_EMPTY[32] = {
    0xc5,0xd2,0x46,0x01,0x86,0xf7,0x23,0x3c,0x92,0x7e,0x7d,0xb2,0xdc,0xc7,0x03,0xc0,
    0xe5,0x00,0xb6,0x53,0xca,0x82,0x27,0x3b,0x7b,0xfa,0xd8,0x04,0x5d,0x85,0xa4,0x70
};
static const uint8_t KECCAK_ABC[32] = {
    0x4e,0x03,0x65,0x7a,0xea,0x45,0xa9,0x4f,0xc7,0xd4,0x7b,0xa8,0x26,0xc8,0xd6,0x67,
    0xc0,0xd1,0xe6,0xe3,0x3a,0x64,0xa0,0x36,0xec,0x44,0xf5,0x8f,0xa1,0x2d,0x6c,0x45
};

int main() {
    int pass = 0, fail = 0;

    // Test 1: keccak256("")
    {
        uint8_t out[32]={0}; uint32_t off=0, len=0; uint8_t in[1]={0};
        blockIdx.x=0; blockDim.x=1; threadIdx.x=0;
        keccak256_batch(in, &off, &len, out, 1);
        if (memcmp(out, KECCAK_EMPTY, 32)==0) { printf("  PASS: keccak256('')\n"); pass++; }
        else { printf("  FAIL: keccak256('')\n"); fail++; }
    }

    // Test 2: keccak256("abc")
    {
        uint8_t in[3]={'a','b','c'}; uint8_t out[32]={0}; uint32_t off=0, len=3;
        blockIdx.x=0; blockDim.x=1; threadIdx.x=0;
        keccak256_batch(in, &off, &len, out, 1);
        if (memcmp(out, KECCAK_ABC, 32)==0) { printf("  PASS: keccak256('abc')\n"); pass++; }
        else { printf("  FAIL: keccak256('abc')\n"); fail++; }
    }

    // Test 3: batch 100 hashes deterministic
    {
        uint8_t data[3200]; uint32_t offs[100], lens[100]; uint8_t outs[3200], outs2[3200];
        for (int i=0;i<100;i++) { offs[i]=i*32; lens[i]=32; for(int j=0;j<32;j++) data[i*32+j]=(uint8_t)(i+j); }
        for (int t=0;t<100;t++) { blockIdx.x=t; keccak256_batch(data,offs,lens,outs,100); }
        for (int t=0;t<100;t++) { blockIdx.x=t; keccak256_batch(data,offs,lens,outs2,100); }
        if (memcmp(outs, outs2, 3200)==0) { printf("  PASS: batch deterministic\n"); pass++; }
        else { printf("  FAIL: batch deterministic\n"); fail++; }
    }

    printf("\n=== CUDA CPU-mode: %d pass, %d fail ===\n", pass, fail);
    return fail;
}
