// bsk_prefetch — CUDA stub (reference: bsk_prefetch.metal)
// FHE/NTT kernel — port from Metal when CUDA testing available
#include <cstdint>

#ifndef __CUDA_ARCH__
#define __device__
#define __global__
#endif

extern "C" __global__ void bsk_prefetch_kernel(
    const uint8_t* inputs, uint8_t* outputs, uint32_t num_items)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_items) return;
    // TODO: port from bsk_prefetch.metal
}
