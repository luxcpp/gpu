// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// CUDA Backend Plugin - NVIDIA GPU acceleration for lux-gpu
//
// Implements the backend vtable. keccak256 and ecrecover fall back to CPU
// until .cu kernels are added. Buffer management uses cudaMalloc/cudaFree.
//
// Guarded by LUX_GPU_CUDA — compiles to a no-op on non-CUDA platforms.

#include "lux/gpu/backend_plugin.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>

#ifdef LUX_GPU_CUDA
#include <cuda_runtime.h>
#endif

// =============================================================================
// Context & Buffer Structures
// =============================================================================

struct LuxBackendContext {
    int device_index;
    std::string device_name;
#ifdef LUX_GPU_CUDA
    cudaDeviceProp props;
#endif
};

struct LuxBackendBuffer {
    void* device_ptr;
    void* host_ptr;   // For staging
    size_t size;
};

struct LuxBackendKernel {
    std::string entry_point;
    // Kernel module handle would go here when .cu files are compiled
};

// =============================================================================
// CPU fallback: Keccak-256 (used when CUDA kernels not yet available)
// =============================================================================

namespace {

static constexpr uint64_t KECCAK_RC[24] = {
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

static inline uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

static void keccak_f1600(uint64_t st[25]) {
    static constexpr int PI_LANE[24] = {
        10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
        15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
    };
    static constexpr int RHO[24] = {
         1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
        27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
    };

    for (int round = 0; round < 24; ++round) {
        uint64_t C[5];
        for (int x = 0; x < 5; ++x)
            C[x] = st[x] ^ st[x+5] ^ st[x+10] ^ st[x+15] ^ st[x+20];
        for (int x = 0; x < 5; ++x) {
            uint64_t d = C[(x+4)%5] ^ rotl64(C[(x+1)%5], 1);
            for (int y = 0; y < 5; ++y) st[x + 5*y] ^= d;
        }
        uint64_t t = st[1];
        for (int i = 0; i < 24; ++i) {
            uint64_t tmp = st[PI_LANE[i]];
            st[PI_LANE[i]] = rotl64(t, RHO[i]);
            t = tmp;
        }
        for (int y = 0; y < 5; ++y) {
            uint64_t row[5];
            for (int x = 0; x < 5; ++x) row[x] = st[x + 5*y];
            for (int x = 0; x < 5; ++x)
                st[x + 5*y] = row[x] ^ ((~row[(x+1)%5]) & row[(x+2)%5]);
        }
        st[0] ^= KECCAK_RC[round];
    }
}

static void keccak256_single(const uint8_t* data, size_t length, uint8_t out[32]) {
    constexpr size_t rate = 136;
    uint64_t state[25] = {};

    size_t absorbed = 0;
    while (absorbed + rate <= length) {
        for (size_t w = 0; w < rate / 8; ++w) {
            uint64_t lane = 0;
            for (size_t b = 0; b < 8; ++b)
                lane |= uint64_t(data[absorbed + w*8 + b]) << (b * 8);
            state[w] ^= lane;
        }
        keccak_f1600(state);
        absorbed += rate;
    }

    uint8_t padded[136] = {};
    size_t remaining = length - absorbed;
    std::memcpy(padded, data + absorbed, remaining);
    padded[remaining] = 0x01;
    padded[rate - 1] |= 0x80;

    for (size_t w = 0; w < rate / 8; ++w) {
        uint64_t lane = 0;
        for (size_t b = 0; b < 8; ++b)
            lane |= uint64_t(padded[w*8 + b]) << (b * 8);
        state[w] ^= lane;
    }
    keccak_f1600(state);

    for (size_t w = 0; w < 4; ++w) {
        uint64_t lane = state[w];
        for (size_t b = 0; b < 8; ++b)
            out[w*8 + b] = static_cast<uint8_t>(lane >> (b * 8));
    }
}

} // anonymous namespace

// =============================================================================
// Lifecycle
// =============================================================================

static LuxBackendContext* cuda_create_context(int device_index) {
    auto* ctx = new LuxBackendContext();
    ctx->device_index = device_index;

#ifdef LUX_GPU_CUDA
    if (cudaSetDevice(device_index) != cudaSuccess) {
        delete ctx;
        return nullptr;
    }
    cudaGetDeviceProperties(&ctx->props, device_index);
    ctx->device_name = ctx->props.name;
#else
    ctx->device_name = "cuda-stub";
#endif

    return ctx;
}

static void cuda_destroy_context(LuxBackendContext* ctx) {
    delete ctx;
}

// =============================================================================
// Device Info
// =============================================================================

static LuxBackendError cuda_get_device_count(int* count) {
    if (!count) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
#ifdef LUX_GPU_CUDA
    cudaError_t err = cudaGetDeviceCount(count);
    if (err != cudaSuccess) { *count = 0; return LUX_BACKEND_ERROR_INTERNAL; }
#else
    *count = 0;
#endif
    return LUX_BACKEND_OK;
}

static LuxBackendError cuda_get_device_info(LuxBackendContext* ctx,
                                             LuxBackendDeviceInfo* info) {
    if (!ctx || !info) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    info->name = ctx->device_name.c_str();
    info->vendor = "NVIDIA";
    info->is_discrete = true;
    info->is_unified_memory = false;

#ifdef LUX_GPU_CUDA
    info->memory_total = ctx->props.totalGlobalMem;
    info->memory_available = info->memory_total;  // Approximate
    info->compute_units = ctx->props.multiProcessorCount;
    info->max_workgroup_size = ctx->props.maxThreadsPerBlock;
#else
    info->memory_total = 0;
    info->memory_available = 0;
    info->compute_units = 0;
    info->max_workgroup_size = 0;
#endif

    return LUX_BACKEND_OK;
}

// =============================================================================
// Sync
// =============================================================================

static LuxBackendError cuda_sync(LuxBackendContext*) {
#ifdef LUX_GPU_CUDA
    return (cudaDeviceSynchronize() == cudaSuccess)
           ? LUX_BACKEND_OK : LUX_BACKEND_ERROR_DEVICE_LOST;
#else
    return LUX_BACKEND_OK;
#endif
}

// =============================================================================
// Buffer Management
// =============================================================================

static LuxBackendBuffer* cuda_buffer_alloc(LuxBackendContext*, size_t bytes) {
    if (bytes == 0) return nullptr;
    auto* buf = new LuxBackendBuffer();
    buf->size = bytes;
    buf->host_ptr = nullptr;

#ifdef LUX_GPU_CUDA
    if (cudaMalloc(&buf->device_ptr, bytes) != cudaSuccess) {
        delete buf;
        return nullptr;
    }
    cudaMemset(buf->device_ptr, 0, bytes);
#else
    buf->device_ptr = std::calloc(1, bytes);
#endif

    return buf;
}

static LuxBackendBuffer* cuda_buffer_alloc_with_data(LuxBackendContext* ctx,
                                                      const void* data, size_t bytes) {
    auto* buf = cuda_buffer_alloc(ctx, bytes);
    if (!buf || !data) return buf;

#ifdef LUX_GPU_CUDA
    cudaMemcpy(buf->device_ptr, data, bytes, cudaMemcpyHostToDevice);
#else
    std::memcpy(buf->device_ptr, data, bytes);
#endif

    return buf;
}

static void cuda_buffer_free(LuxBackendContext*, LuxBackendBuffer* buf) {
    if (!buf) return;
#ifdef LUX_GPU_CUDA
    cudaFree(buf->device_ptr);
#else
    std::free(buf->device_ptr);
#endif
    std::free(buf->host_ptr);
    delete buf;
}

static LuxBackendError cuda_buffer_copy_to_host(LuxBackendContext*,
                                                 LuxBackendBuffer* buf,
                                                 void* dst, size_t bytes) {
    if (!buf || !dst) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    size_t n = std::min(bytes, buf->size);
#ifdef LUX_GPU_CUDA
    return (cudaMemcpy(dst, buf->device_ptr, n, cudaMemcpyDeviceToHost) == cudaSuccess)
           ? LUX_BACKEND_OK : LUX_BACKEND_ERROR_INTERNAL;
#else
    std::memcpy(dst, buf->device_ptr, n);
    return LUX_BACKEND_OK;
#endif
}

static LuxBackendError cuda_buffer_copy_from_host(LuxBackendContext*,
                                                   LuxBackendBuffer* buf,
                                                   const void* src, size_t bytes) {
    if (!buf || !src) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    size_t n = std::min(bytes, buf->size);
#ifdef LUX_GPU_CUDA
    return (cudaMemcpy(buf->device_ptr, src, n, cudaMemcpyHostToDevice) == cudaSuccess)
           ? LUX_BACKEND_OK : LUX_BACKEND_ERROR_INTERNAL;
#else
    std::memcpy(buf->device_ptr, src, n);
    return LUX_BACKEND_OK;
#endif
}

static void* cuda_buffer_get_host_ptr(LuxBackendContext*, LuxBackendBuffer* buf) {
    if (!buf) return nullptr;
#ifdef LUX_GPU_CUDA
    // CUDA doesn't have unified memory by default; return nullptr.
    // Caller should use buffer_copy_to_host instead.
    return nullptr;
#else
    return buf->device_ptr;
#endif
}

// =============================================================================
// Kernel Management (stubs)
// =============================================================================

static LuxBackendKernel* cuda_kernel_load(LuxBackendContext*, const char*,
                                           const char* entry_point) {
    if (!entry_point) return nullptr;
    auto* k = new LuxBackendKernel();
    k->entry_point = entry_point;
    return k;
}

static LuxBackendKernel* cuda_kernel_load_binary(LuxBackendContext*, const void*,
                                                   size_t, const char* entry_point) {
    if (!entry_point) return nullptr;
    auto* k = new LuxBackendKernel();
    k->entry_point = entry_point;
    return k;
}

static void cuda_kernel_destroy(LuxBackendContext*, LuxBackendKernel* k) {
    delete k;
}

static LuxBackendError cuda_kernel_dispatch(
    LuxBackendContext*, LuxBackendKernel*, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, LuxBackendBuffer**, int) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

// =============================================================================
// Tensor Ops — CPU fallback until .cu kernels exist
// =============================================================================

// Helper to get a float pointer from buffer (works in stub mode)
static float* buf_f32(LuxBackendBuffer* b) {
#ifdef LUX_GPU_CUDA
    // In real CUDA mode, we'd need device-side kernels.
    // For now this only works in stub mode.
    return nullptr;
#else
    return static_cast<float*>(b->device_ptr);
#endif
}

static LuxBackendError cuda_op_add_f32(LuxBackendContext*, LuxBackendBuffer* a,
                                        LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
#ifndef LUX_GPU_CUDA
    float *pa = buf_f32(a), *pb = buf_f32(b), *po = buf_f32(out);
    if (!pa || !pb || !po) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    for (size_t i = 0; i < n; i++) po[i] = pa[i] + pb[i];
    return LUX_BACKEND_OK;
#else
    (void)a; (void)b; (void)out; (void)n;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

static LuxBackendError cuda_op_sub_f32(LuxBackendContext*, LuxBackendBuffer* a,
                                        LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
#ifndef LUX_GPU_CUDA
    float *pa = buf_f32(a), *pb = buf_f32(b), *po = buf_f32(out);
    if (!pa || !pb || !po) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    for (size_t i = 0; i < n; i++) po[i] = pa[i] - pb[i];
    return LUX_BACKEND_OK;
#else
    (void)a; (void)b; (void)out; (void)n;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

static LuxBackendError cuda_op_mul_f32(LuxBackendContext*, LuxBackendBuffer* a,
                                        LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
#ifndef LUX_GPU_CUDA
    float *pa = buf_f32(a), *pb = buf_f32(b), *po = buf_f32(out);
    if (!pa || !pb || !po) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    for (size_t i = 0; i < n; i++) po[i] = pa[i] * pb[i];
    return LUX_BACKEND_OK;
#else
    (void)a; (void)b; (void)out; (void)n;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

static LuxBackendError cuda_op_div_f32(LuxBackendContext*, LuxBackendBuffer* a,
                                        LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
#ifndef LUX_GPU_CUDA
    float *pa = buf_f32(a), *pb = buf_f32(b), *po = buf_f32(out);
    if (!pa || !pb || !po) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    for (size_t i = 0; i < n; i++) po[i] = pa[i] / pb[i];
    return LUX_BACKEND_OK;
#else
    (void)a; (void)b; (void)out; (void)n;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
#endif
}

// Everything else returns NOT_SUPPORTED
static LuxBackendError cuda_not_supported_2buf(LuxBackendContext*, LuxBackendBuffer*,
                                                LuxBackendBuffer*, int, int) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_matmul(LuxBackendContext*, LuxBackendBuffer*,
                                                  LuxBackendBuffer*, LuxBackendBuffer*,
                                                  int, int, int) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_reduce(LuxBackendContext*, LuxBackendBuffer*,
                                                  LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_reduce_axis(LuxBackendContext*, LuxBackendBuffer*,
                                                       LuxBackendBuffer*, size_t, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_softmax(LuxBackendContext*, LuxBackendBuffer*,
                                                   LuxBackendBuffer*, size_t, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_unary(LuxBackendContext*, LuxBackendBuffer*,
                                                 LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_norm(LuxBackendContext*, LuxBackendBuffer*,
                                                LuxBackendBuffer*, LuxBackendBuffer*,
                                                LuxBackendBuffer*, size_t, size_t, float) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_rms(LuxBackendContext*, LuxBackendBuffer*,
                                               LuxBackendBuffer*, LuxBackendBuffer*,
                                               size_t, size_t, float) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_ntt(LuxBackendContext*, uint64_t*, size_t, uint64_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_msm(LuxBackendContext*, const void*, const void*,
                                               void*, size_t, int) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_poly_mul(LuxBackendContext*, const uint64_t*,
                                                    const uint64_t*, uint64_t*, size_t, uint64_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_bootstrap(LuxBackendContext*, const uint64_t*,
                                                     uint64_t*, const uint64_t*,
                                                     const uint64_t*, uint32_t, uint32_t,
                                                     uint32_t, uint32_t, uint64_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_keyswitch(LuxBackendContext*, const uint64_t*,
                                                     uint64_t*, const uint64_t*, uint32_t,
                                                     uint32_t, uint32_t, uint32_t, uint64_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_blind_rotate(LuxBackendContext*, uint64_t*,
                                                        const uint64_t*, const uint64_t*,
                                                        uint32_t, uint32_t, uint32_t,
                                                        uint32_t, uint64_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_sample_extract(LuxBackendContext*, const uint64_t*,
                                                          uint64_t*, uint32_t, uint32_t,
                                                          uint64_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_sample_ntt(LuxBackendContext*, uint64_t*, size_t,
                                                      uint64_t, double, uint64_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_hash(LuxBackendContext*, const uint64_t*,
                                                uint64_t*, size_t, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_blake3(LuxBackendContext*, const uint8_t*,
                                                  uint8_t*, const size_t*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_curve_add(LuxBackendContext*, const void*,
                                                     const void*, void*, size_t, bool) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_curve_mul(LuxBackendContext*, const void*,
                                                     const void*, void*, size_t, bool) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_pairing(LuxBackendContext*, const void*,
                                                   const void*, void*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_kzg_commit(LuxBackendContext*, const void*,
                                                      const void*, void*, size_t, int) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_kzg_open(LuxBackendContext*, const void*,
                                                    const void*, const void*, void*,
                                                    size_t, int) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError cuda_not_supported_kzg_verify(LuxBackendContext*, const void*,
                                                      const void*, const void*, const void*,
                                                      const void*, bool*, int) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

// =============================================================================
// Keccak-256 — CPU fallback until .cu kernel is written
// =============================================================================

static LuxBackendError cuda_op_keccak256_hash(
    LuxBackendContext*,
    const uint8_t* inputs, uint8_t* outputs,
    const size_t* input_lens, size_t num_inputs) {

    if (!inputs || !outputs || !input_lens)
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    size_t offset = 0;
    for (size_t i = 0; i < num_inputs; i++) {
        keccak256_single(inputs + offset, input_lens[i], outputs + i * 32);
        offset += input_lens[i];
    }
    return LUX_BACKEND_OK;
}

// =============================================================================
// ecrecover — CPU fallback (returns invalid for now; full impl needs secp256k1)
// =============================================================================

static LuxBackendError cuda_op_ecrecover_batch(
    LuxBackendContext*,
    const void*,
    void* addresses,
    size_t num_signatures) {

    if (!addresses) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    // Zero all outputs (valid=0), indicating recovery not yet implemented
    std::memset(addresses, 0, num_signatures * 32);
    return LUX_BACKEND_OK;
}

// =============================================================================
// CUDA Backend VTable
// =============================================================================

static const lux_gpu_backend_vtbl cuda_vtbl = {
    .create_context = cuda_create_context,
    .destroy_context = cuda_destroy_context,
    .get_device_count = cuda_get_device_count,
    .get_device_info = cuda_get_device_info,
    .sync = cuda_sync,
    .buffer_alloc = cuda_buffer_alloc,
    .buffer_alloc_with_data = cuda_buffer_alloc_with_data,
    .buffer_free = cuda_buffer_free,
    .buffer_copy_to_host = cuda_buffer_copy_to_host,
    .buffer_copy_from_host = cuda_buffer_copy_from_host,
    .buffer_get_host_ptr = cuda_buffer_get_host_ptr,
    .kernel_load = cuda_kernel_load,
    .kernel_load_binary = cuda_kernel_load_binary,
    .kernel_destroy = cuda_kernel_destroy,
    .kernel_dispatch = cuda_kernel_dispatch,
    .op_add_f32 = cuda_op_add_f32,
    .op_sub_f32 = cuda_op_sub_f32,
    .op_mul_f32 = cuda_op_mul_f32,
    .op_div_f32 = cuda_op_div_f32,
    .op_matmul_f32 = cuda_not_supported_matmul,
    .op_transpose_f32 = cuda_not_supported_2buf,
    .op_reduce_sum_f32 = cuda_not_supported_reduce,
    .op_reduce_max_f32 = cuda_not_supported_reduce,
    .op_reduce_min_f32 = cuda_not_supported_reduce,
    .op_reduce_mean_f32 = cuda_not_supported_reduce,
    .op_reduce_sum_axis_f32 = cuda_not_supported_reduce_axis,
    .op_reduce_max_axis_f32 = cuda_not_supported_reduce_axis,
    .op_softmax_f32 = cuda_not_supported_softmax,
    .op_log_softmax_f32 = cuda_not_supported_softmax,
    .op_exp_f32 = cuda_not_supported_unary,
    .op_log_f32 = cuda_not_supported_unary,
    .op_sqrt_f32 = cuda_not_supported_unary,
    .op_neg_f32 = cuda_not_supported_unary,
    .op_abs_f32 = cuda_not_supported_unary,
    .op_tanh_f32 = cuda_not_supported_unary,
    .op_sigmoid_f32 = cuda_not_supported_unary,
    .op_relu_f32 = cuda_not_supported_unary,
    .op_gelu_f32 = cuda_not_supported_unary,
    .op_copy_f32 = cuda_not_supported_unary,
    .op_layer_norm_f32 = cuda_not_supported_norm,
    .op_rms_norm_f32 = cuda_not_supported_rms,
    .op_ntt_forward = cuda_not_supported_ntt,
    .op_ntt_inverse = cuda_not_supported_ntt,
    .op_msm = cuda_not_supported_msm,
    .op_poly_mul = cuda_not_supported_poly_mul,
    .op_tfhe_bootstrap = cuda_not_supported_bootstrap,
    .op_tfhe_keyswitch = cuda_not_supported_keyswitch,
    .op_blind_rotate = cuda_not_supported_blind_rotate,
    .op_sample_extract = cuda_not_supported_sample_extract,
    .op_sample_ntt = cuda_not_supported_sample_ntt,
    .op_poseidon2_hash = cuda_not_supported_hash,
    .op_blake3_hash = cuda_not_supported_blake3,
    .op_keccak256_hash = cuda_op_keccak256_hash,
    .op_bls12_381_add = cuda_not_supported_curve_add,
    .op_bls12_381_mul = cuda_not_supported_curve_mul,
    .op_bls12_381_pairing = cuda_not_supported_pairing,
    .op_bn254_add = cuda_not_supported_curve_add,
    .op_bn254_mul = cuda_not_supported_curve_mul,
    .op_kzg_commit = cuda_not_supported_kzg_commit,
    .op_kzg_open = cuda_not_supported_kzg_open,
    .op_kzg_verify = cuda_not_supported_kzg_verify,
    .op_ecrecover_batch = cuda_op_ecrecover_batch,
    ._reserved = {nullptr, nullptr, nullptr},
};

// =============================================================================
// Entry Point
// =============================================================================

static bool cuda_init(lux_gpu_backend_desc* out) {
    if (!out) return false;

#ifdef LUX_GPU_CUDA
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        return false;
#endif

    out->abi_version = LUX_GPU_BACKEND_ABI_VERSION;
    out->backend_name = "cuda";
    out->backend_version = "0.2.0";
    out->capabilities = LUX_CAP_TENSOR_OPS | LUX_CAP_KECCAK256;
    out->vtbl = &cuda_vtbl;
    return true;
}

LUX_GPU_DECLARE_BACKEND(cuda_init)
