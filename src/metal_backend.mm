// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Metal Backend Plugin - Apple GPU acceleration for lux-gpu
//
// Implements the full backend vtable via Metal compute shaders.
// Priority ops: keccak256, ecrecover (EVM-critical).
// Uses MTLResourceStorageModeShared for unified memory on Apple Silicon.

#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "lux/gpu/backend_plugin.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <unordered_map>
#include <vector>

// =============================================================================
// Metal Context & Buffer Structures
// =============================================================================

struct MetalKernel {
    id<MTLComputePipelineState> pipeline;
    NSString* name;
};

struct LuxBackendContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;

    // Compiled pipelines keyed by function name
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

    // Libraries compiled from .metal source
    id<MTLLibrary> keccak_lib;
    id<MTLLibrary> ecrecover_lib;

    std::string device_name;
    int device_index;
};

struct LuxBackendBuffer {
    id<MTLBuffer> mtl;
    size_t size;
    // On Apple Silicon with unified memory, the CPU pointer is the same as the
    // GPU pointer. No copy needed. Just use [mtl contents].
};

struct LuxBackendKernel {
    id<MTLComputePipelineState> pipeline;
    std::string entry_point;
};

// =============================================================================
// Shader source paths and loading
// =============================================================================

namespace {

// HashInput must match the Metal shader struct exactly.
struct alignas(4) GPUHashInput {
    uint32_t offset;
    uint32_t length;
};
static_assert(sizeof(GPUHashInput) == 8);

// Ecrecover I/O must match gpu.h LuxEcrecoverInput / LuxEcrecoverOutput layout
// These are passed through to the Metal shader as raw bytes.

/// Find a .metal source file relative to known search paths.
static NSString* find_metal_source(const char* name) {
    const char* env = std::getenv("LUX_GPU_KERNEL_PATH");
    std::vector<std::string> search_dirs;

    if (env && *env) {
        search_dirs.push_back(env);
    }

    // Find the directory containing this loaded dylib (dladdr on our own symbol)
    Dl_info dl_info;
    if (dladdr(reinterpret_cast<void*>(&find_metal_source), &dl_info) && dl_info.dli_fname) {
        std::string dylib_path(dl_info.dli_fname);
        auto slash = dylib_path.rfind('/');
        if (slash != std::string::npos) {
            search_dirs.push_back(dylib_path.substr(0, slash));
        }
    }

    // Relative to this source file (compile-time path)
    std::string this_file(__FILE__);
    auto last_slash = this_file.rfind('/');
    if (last_slash != std::string::npos) {
        std::string src_dir = this_file.substr(0, last_slash);
        search_dirs.push_back(src_dir + "/../kernels");
    }

    search_dirs.push_back("/usr/local/share/lux-gpu/kernels");
    search_dirs.push_back("/opt/homebrew/share/lux-gpu/kernels");

    NSError* error = nil;
    for (const auto& dir : search_dirs) {
        std::string path = dir + "/" + name;
        NSString* ns_path = [NSString stringWithUTF8String:path.c_str()];
        NSString* source = [NSString stringWithContentsOfFile:ns_path
                                     encoding:NSUTF8StringEncoding
                                     error:&error];
        if (source) {
            return source;
        }
    }
    return nil;
}

/// Compile a .metal source string into a library.
static id<MTLLibrary> compile_metal_source(id<MTLDevice> device, NSString* source) {
    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.mathMode = MTLMathModeFast;
    opts.languageVersion = MTLLanguageVersion3_0;

    id<MTLLibrary> lib = [device newLibraryWithSource:source options:opts error:&error];
    if (!lib) {
        fprintf(stderr, "lux-gpu metal: shader compilation failed: %s\n",
                [[error localizedDescription] UTF8String]);
    }
    return lib;
}

/// Get or create a pipeline for a given function name in a library.
static id<MTLComputePipelineState> get_pipeline(LuxBackendContext* ctx,
                                                 id<MTLLibrary> lib,
                                                 const char* func_name) {
    auto it = ctx->pipelines.find(func_name);
    if (it != ctx->pipelines.end()) {
        return it->second;
    }

    NSString* name = [NSString stringWithUTF8String:func_name];
    id<MTLFunction> func = [lib newFunctionWithName:name];
    if (!func) {
        fprintf(stderr, "lux-gpu metal: function '%s' not found\n", func_name);
        return nil;
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [ctx->device newComputePipelineStateWithFunction:func error:&error];
    if (!pipeline) {
        fprintf(stderr, "lux-gpu metal: pipeline creation failed for '%s': %s\n",
                func_name, [[error localizedDescription] UTF8String]);
        return nil;
    }

    ctx->pipelines[func_name] = pipeline;
    return pipeline;
}

/// Dispatch a compute kernel with the given buffers and thread count.
static LuxBackendError dispatch_1d(LuxBackendContext* ctx,
                                    id<MTLComputePipelineState> pipeline,
                                    id<MTLBuffer>* buffers, int num_buffers,
                                    NSUInteger thread_count) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        if (!cmd) return LUX_BACKEND_ERROR_INTERNAL;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];

        for (int i = 0; i < num_buffers; i++) {
            [enc setBuffer:buffers[i] offset:0 atIndex:static_cast<NSUInteger>(i)];
        }

        NSUInteger tpg = pipeline.maxTotalThreadsPerThreadgroup;
        if (tpg > thread_count) tpg = thread_count;
        if (tpg == 0) tpg = 1;

        MTLSize grid = MTLSizeMake(thread_count, 1, 1);
        MTLSize group = MTLSizeMake(tpg, 1, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd error]) {
            fprintf(stderr, "lux-gpu metal: command failed: %s\n",
                    [[[cmd error] localizedDescription] UTF8String]);
            return LUX_BACKEND_ERROR_DEVICE_LOST;
        }
    }
    return LUX_BACKEND_OK;
}

// =============================================================================
// Inline Metal shader source for basic tensor ops
// =============================================================================

static NSString* const kTensorOpsSource = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void add_f32(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* out     [[buffer(2)]],
                    uint tid              [[thread_position_in_grid]]) {
    out[tid] = a[tid] + b[tid];
}

kernel void sub_f32(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* out     [[buffer(2)]],
                    uint tid              [[thread_position_in_grid]]) {
    out[tid] = a[tid] - b[tid];
}

kernel void mul_f32(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* out     [[buffer(2)]],
                    uint tid              [[thread_position_in_grid]]) {
    out[tid] = a[tid] * b[tid];
}

kernel void div_f32(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* out     [[buffer(2)]],
                    uint tid              [[thread_position_in_grid]]) {
    out[tid] = a[tid] / b[tid];
}

kernel void exp_f32(device const float* in [[buffer(0)]],
                    device float* out      [[buffer(1)]],
                    uint tid               [[thread_position_in_grid]]) {
    out[tid] = exp(in[tid]);
}

kernel void log_f32(device const float* in [[buffer(0)]],
                    device float* out      [[buffer(1)]],
                    uint tid               [[thread_position_in_grid]]) {
    out[tid] = log(in[tid]);
}

kernel void sqrt_f32(device const float* in [[buffer(0)]],
                     device float* out      [[buffer(1)]],
                     uint tid               [[thread_position_in_grid]]) {
    out[tid] = sqrt(in[tid]);
}

kernel void neg_f32(device const float* in [[buffer(0)]],
                    device float* out      [[buffer(1)]],
                    uint tid               [[thread_position_in_grid]]) {
    out[tid] = -in[tid];
}

kernel void abs_f32(device const float* in [[buffer(0)]],
                    device float* out      [[buffer(1)]],
                    uint tid               [[thread_position_in_grid]]) {
    out[tid] = abs(in[tid]);
}

kernel void tanh_f32(device const float* in [[buffer(0)]],
                     device float* out      [[buffer(1)]],
                     uint tid               [[thread_position_in_grid]]) {
    out[tid] = tanh(in[tid]);
}

kernel void sigmoid_f32(device const float* in [[buffer(0)]],
                        device float* out      [[buffer(1)]],
                        uint tid               [[thread_position_in_grid]]) {
    out[tid] = 1.0f / (1.0f + exp(-in[tid]));
}

kernel void relu_f32(device const float* in [[buffer(0)]],
                     device float* out      [[buffer(1)]],
                     uint tid               [[thread_position_in_grid]]) {
    out[tid] = max(in[tid], 0.0f);
}

kernel void gelu_f32(device const float* in [[buffer(0)]],
                     device float* out      [[buffer(1)]],
                     uint tid               [[thread_position_in_grid]]) {
    float x = in[tid];
    out[tid] = 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

kernel void copy_f32(device const float* src [[buffer(0)]],
                     device float* dst       [[buffer(1)]],
                     uint tid                [[thread_position_in_grid]]) {
    dst[tid] = src[tid];
}
)";

} // anonymous namespace

// =============================================================================
// Lifecycle
// =============================================================================

static LuxBackendContext* metal_create_context(int device_index) {
    @autoreleasepool {
        id<MTLDevice> device = nil;

        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices && (int)devices.count > device_index && device_index >= 0) {
            device = devices[device_index];
        } else {
            device = MTLCreateSystemDefaultDevice();
        }

        if (!device) return nullptr;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return nullptr;

        auto* ctx = new LuxBackendContext();
        ctx->device = device;
        ctx->queue = queue;
        ctx->device_index = device_index;
        ctx->device_name = std::string([[device name] UTF8String]);

        // Compile inline tensor ops
        id<MTLLibrary> tensor_lib = compile_metal_source(device, kTensorOpsSource);
        if (tensor_lib) {
            // Pre-warm commonly used pipelines
            NSArray* funcs = @[@"add_f32", @"sub_f32", @"mul_f32", @"div_f32",
                               @"exp_f32", @"log_f32", @"sqrt_f32", @"neg_f32",
                               @"abs_f32", @"tanh_f32", @"sigmoid_f32", @"relu_f32",
                               @"gelu_f32", @"copy_f32"];
            for (NSString* name in funcs) {
                id<MTLFunction> func = [tensor_lib newFunctionWithName:name];
                if (func) {
                    NSError* err = nil;
                    id<MTLComputePipelineState> p =
                        [device newComputePipelineStateWithFunction:func error:&err];
                    if (p) {
                        ctx->pipelines[[name UTF8String]] = p;
                    }
                }
            }
        }

        // Load keccak256 kernel from .metal file
        NSString* keccak_src = find_metal_source("keccak256.metal");
        if (keccak_src) {
            ctx->keccak_lib = compile_metal_source(device, keccak_src);
        }

        // Load secp256k1 ecrecover kernel from .metal file
        NSString* ecrecover_src = find_metal_source("secp256k1_recover.metal");
        if (ecrecover_src) {
            ctx->ecrecover_lib = compile_metal_source(device, ecrecover_src);
        }

        return ctx;
    }
}

static void metal_destroy_context(LuxBackendContext* ctx) {
    if (!ctx) return;
    ctx->pipelines.clear();
    ctx->keccak_lib = nil;
    ctx->ecrecover_lib = nil;
    ctx->queue = nil;
    ctx->device = nil;
    delete ctx;
}

// =============================================================================
// Device Info
// =============================================================================

static LuxBackendError metal_get_device_count(int* count) {
    @autoreleasepool {
        if (!count) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        *count = devices ? (int)devices.count : 0;
        if (*count == 0) {
            // MTLCopyAllDevices can return empty on some systems; try default
            if (MTLCreateSystemDefaultDevice()) *count = 1;
        }
        return LUX_BACKEND_OK;
    }
}

static LuxBackendError metal_get_device_info(LuxBackendContext* ctx,
                                              LuxBackendDeviceInfo* info) {
    if (!ctx || !info) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    info->name = ctx->device_name.c_str();
    info->vendor = "Apple";
    info->memory_total = [ctx->device recommendedMaxWorkingSetSize];
    info->memory_available = info->memory_total;
    info->compute_units = 0;  // Metal doesn't expose this directly
    info->max_workgroup_size = (int)[ctx->device maxThreadsPerThreadgroup].width;
    info->is_discrete = ![ctx->device hasUnifiedMemory];
    info->is_unified_memory = [ctx->device hasUnifiedMemory];

    return LUX_BACKEND_OK;
}

// =============================================================================
// Synchronization
// =============================================================================

static LuxBackendError metal_sync(LuxBackendContext* ctx) {
    if (!ctx) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    return LUX_BACKEND_OK;
}

// =============================================================================
// Buffer Management
// =============================================================================

static LuxBackendBuffer* metal_buffer_alloc(LuxBackendContext* ctx, size_t bytes) {
    if (!ctx || bytes == 0) return nullptr;
    @autoreleasepool {
        // MTLResourceStorageModeShared: CPU and GPU share the same physical memory
        // on Apple Silicon. No copy needed.
        id<MTLBuffer> mtl = [ctx->device newBufferWithLength:bytes
                                         options:MTLResourceStorageModeShared];
        if (!mtl) return nullptr;

        auto* buf = new LuxBackendBuffer();
        buf->mtl = mtl;
        buf->size = bytes;
        return buf;
    }
}

static LuxBackendBuffer* metal_buffer_alloc_with_data(LuxBackendContext* ctx,
                                                       const void* data, size_t bytes) {
    if (!ctx || !data || bytes == 0) return nullptr;
    @autoreleasepool {
        id<MTLBuffer> mtl = [ctx->device newBufferWithBytes:data
                                         length:bytes
                                         options:MTLResourceStorageModeShared];
        if (!mtl) return nullptr;

        auto* buf = new LuxBackendBuffer();
        buf->mtl = mtl;
        buf->size = bytes;
        return buf;
    }
}

static void metal_buffer_free(LuxBackendContext*, LuxBackendBuffer* buf) {
    if (!buf) return;
    buf->mtl = nil;
    delete buf;
}

static LuxBackendError metal_buffer_copy_to_host(LuxBackendContext*,
                                                  LuxBackendBuffer* buf,
                                                  void* dst, size_t bytes) {
    if (!buf || !dst) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    size_t to_copy = std::min(bytes, buf->size);
    std::memcpy(dst, [buf->mtl contents], to_copy);
    return LUX_BACKEND_OK;
}

static LuxBackendError metal_buffer_copy_from_host(LuxBackendContext*,
                                                    LuxBackendBuffer* buf,
                                                    const void* src, size_t bytes) {
    if (!buf || !src) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    size_t to_copy = std::min(bytes, buf->size);
    std::memcpy([buf->mtl contents], src, to_copy);
    return LUX_BACKEND_OK;
}

static void* metal_buffer_get_host_ptr(LuxBackendContext*, LuxBackendBuffer* buf) {
    if (!buf) return nullptr;
    // On Apple Silicon unified memory, this IS the GPU pointer too.
    return [buf->mtl contents];
}

// =============================================================================
// Custom Kernel Management
// =============================================================================

static LuxBackendKernel* metal_kernel_load(LuxBackendContext* ctx,
                                            const char* source,
                                            const char* entry_point) {
    if (!ctx || !source || !entry_point) return nullptr;
    @autoreleasepool {
        NSString* src = [NSString stringWithUTF8String:source];
        id<MTLLibrary> lib = compile_metal_source(ctx->device, src);
        if (!lib) return nullptr;

        NSString* name = [NSString stringWithUTF8String:entry_point];
        id<MTLFunction> func = [lib newFunctionWithName:name];
        if (!func) return nullptr;

        NSError* error = nil;
        id<MTLComputePipelineState> pipeline =
            [ctx->device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline) return nullptr;

        auto* kernel = new LuxBackendKernel();
        kernel->pipeline = pipeline;
        kernel->entry_point = entry_point;
        return kernel;
    }
}

static LuxBackendKernel* metal_kernel_load_binary(LuxBackendContext* ctx,
                                                    const void* binary, size_t size,
                                                    const char* entry_point) {
    if (!ctx || !binary || !entry_point || size == 0) return nullptr;
    @autoreleasepool {
        dispatch_data_t data = dispatch_data_create(binary, size,
                                                     dispatch_get_main_queue(),
                                                     DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NSError* error = nil;
        id<MTLLibrary> lib = [ctx->device newLibraryWithData:data error:&error];
        if (!lib) return nullptr;

        NSString* name = [NSString stringWithUTF8String:entry_point];
        id<MTLFunction> func = [lib newFunctionWithName:name];
        if (!func) return nullptr;

        id<MTLComputePipelineState> pipeline =
            [ctx->device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline) return nullptr;

        auto* kernel = new LuxBackendKernel();
        kernel->pipeline = pipeline;
        kernel->entry_point = entry_point;
        return kernel;
    }
}

static void metal_kernel_destroy(LuxBackendContext*, LuxBackendKernel* kernel) {
    if (!kernel) return;
    kernel->pipeline = nil;
    delete kernel;
}

// =============================================================================
// Kernel Dispatch
// =============================================================================

static LuxBackendError metal_kernel_dispatch(
    LuxBackendContext* ctx,
    LuxBackendKernel* kernel,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t block_x, uint32_t block_y, uint32_t block_z,
    LuxBackendBuffer** buffers, int num_buffers) {

    if (!ctx || !kernel) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:kernel->pipeline];

        for (int i = 0; i < num_buffers; i++) {
            if (buffers[i]) {
                [enc setBuffer:buffers[i]->mtl offset:0 atIndex:static_cast<NSUInteger>(i)];
            }
        }

        MTLSize grid = MTLSizeMake(grid_x, grid_y, grid_z);
        MTLSize group = MTLSizeMake(block_x, block_y, block_z);

        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd error]) return LUX_BACKEND_ERROR_DEVICE_LOST;
    }
    return LUX_BACKEND_OK;
}

// =============================================================================
// Tensor Ops - dispatch inline Metal shaders
// =============================================================================

// Helper: dispatch a binary op (add, sub, mul, div)
static LuxBackendError metal_binary_op(LuxBackendContext* ctx,
                                        const char* op_name,
                                        LuxBackendBuffer* a, LuxBackendBuffer* b,
                                        LuxBackendBuffer* out, size_t n) {
    if (!ctx || !a || !b || !out || n == 0) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    auto it = ctx->pipelines.find(op_name);
    if (it == ctx->pipelines.end()) return LUX_BACKEND_ERROR_NOT_SUPPORTED;

    id<MTLBuffer> bufs[3] = { a->mtl, b->mtl, out->mtl };
    return dispatch_1d(ctx, it->second, bufs, 3, n);
}

// Helper: dispatch a unary op
static LuxBackendError metal_unary_op(LuxBackendContext* ctx,
                                       const char* op_name,
                                       LuxBackendBuffer* in, LuxBackendBuffer* out,
                                       size_t n) {
    if (!ctx || !in || !out || n == 0) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    auto it = ctx->pipelines.find(op_name);
    if (it == ctx->pipelines.end()) return LUX_BACKEND_ERROR_NOT_SUPPORTED;

    id<MTLBuffer> bufs[2] = { in->mtl, out->mtl };
    return dispatch_1d(ctx, it->second, bufs, 2, n);
}

static LuxBackendError metal_op_add_f32(LuxBackendContext* ctx, LuxBackendBuffer* a,
                                         LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
    return metal_binary_op(ctx, "add_f32", a, b, out, n);
}
static LuxBackendError metal_op_sub_f32(LuxBackendContext* ctx, LuxBackendBuffer* a,
                                         LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
    return metal_binary_op(ctx, "sub_f32", a, b, out, n);
}
static LuxBackendError metal_op_mul_f32(LuxBackendContext* ctx, LuxBackendBuffer* a,
                                         LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
    return metal_binary_op(ctx, "mul_f32", a, b, out, n);
}
static LuxBackendError metal_op_div_f32(LuxBackendContext* ctx, LuxBackendBuffer* a,
                                         LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
    return metal_binary_op(ctx, "div_f32", a, b, out, n);
}

// Matmul: naive for now, tiled version later
static LuxBackendError metal_op_matmul_f32(LuxBackendContext* ctx,
                                            LuxBackendBuffer* a, LuxBackendBuffer* b,
                                            LuxBackendBuffer* out,
                                            int M, int K, int N) {
    (void)ctx; (void)a; (void)b; (void)out; (void)M; (void)K; (void)N;
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_transpose_f32(LuxBackendContext*, LuxBackendBuffer*,
                                               LuxBackendBuffer*, int, int) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

// Reductions: not yet implemented on GPU
static LuxBackendError metal_op_reduce_sum_f32(LuxBackendContext*, LuxBackendBuffer*,
                                                LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_op_reduce_max_f32(LuxBackendContext*, LuxBackendBuffer*,
                                                LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_op_reduce_min_f32(LuxBackendContext*, LuxBackendBuffer*,
                                                LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_op_reduce_mean_f32(LuxBackendContext*, LuxBackendBuffer*,
                                                 LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_op_reduce_sum_axis_f32(LuxBackendContext*, LuxBackendBuffer*,
                                                     LuxBackendBuffer*, size_t, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_op_reduce_max_axis_f32(LuxBackendContext*, LuxBackendBuffer*,
                                                     LuxBackendBuffer*, size_t, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

// Softmax
static LuxBackendError metal_op_softmax_f32(LuxBackendContext*, LuxBackendBuffer*,
                                             LuxBackendBuffer*, size_t, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_op_log_softmax_f32(LuxBackendContext*, LuxBackendBuffer*,
                                                 LuxBackendBuffer*, size_t, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

// Unary ops
static LuxBackendError metal_op_exp_f32(LuxBackendContext* ctx, LuxBackendBuffer* in,
                                         LuxBackendBuffer* out, size_t n) {
    return metal_unary_op(ctx, "exp_f32", in, out, n);
}
static LuxBackendError metal_op_log_f32(LuxBackendContext* ctx, LuxBackendBuffer* in,
                                         LuxBackendBuffer* out, size_t n) {
    return metal_unary_op(ctx, "log_f32", in, out, n);
}
static LuxBackendError metal_op_sqrt_f32(LuxBackendContext* ctx, LuxBackendBuffer* in,
                                          LuxBackendBuffer* out, size_t n) {
    return metal_unary_op(ctx, "sqrt_f32", in, out, n);
}
static LuxBackendError metal_op_neg_f32(LuxBackendContext* ctx, LuxBackendBuffer* in,
                                         LuxBackendBuffer* out, size_t n) {
    return metal_unary_op(ctx, "neg_f32", in, out, n);
}
static LuxBackendError metal_op_abs_f32(LuxBackendContext* ctx, LuxBackendBuffer* in,
                                         LuxBackendBuffer* out, size_t n) {
    return metal_unary_op(ctx, "abs_f32", in, out, n);
}
static LuxBackendError metal_op_tanh_f32(LuxBackendContext* ctx, LuxBackendBuffer* in,
                                          LuxBackendBuffer* out, size_t n) {
    return metal_unary_op(ctx, "tanh_f32", in, out, n);
}
static LuxBackendError metal_op_sigmoid_f32(LuxBackendContext* ctx, LuxBackendBuffer* in,
                                             LuxBackendBuffer* out, size_t n) {
    return metal_unary_op(ctx, "sigmoid_f32", in, out, n);
}
static LuxBackendError metal_op_relu_f32(LuxBackendContext* ctx, LuxBackendBuffer* in,
                                          LuxBackendBuffer* out, size_t n) {
    return metal_unary_op(ctx, "relu_f32", in, out, n);
}
static LuxBackendError metal_op_gelu_f32(LuxBackendContext* ctx, LuxBackendBuffer* in,
                                          LuxBackendBuffer* out, size_t n) {
    return metal_unary_op(ctx, "gelu_f32", in, out, n);
}

// Copy
static LuxBackendError metal_op_copy_f32(LuxBackendContext* ctx, LuxBackendBuffer* src,
                                          LuxBackendBuffer* dst, size_t n) {
    return metal_unary_op(ctx, "copy_f32", src, dst, n);
}

// Normalization
static LuxBackendError metal_op_layer_norm_f32(LuxBackendContext*, LuxBackendBuffer*,
                                                LuxBackendBuffer*, LuxBackendBuffer*,
                                                LuxBackendBuffer*, size_t, size_t, float) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_op_rms_norm_f32(LuxBackendContext*, LuxBackendBuffer*,
                                              LuxBackendBuffer*, LuxBackendBuffer*,
                                              size_t, size_t, float) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

// NTT / MSM / FHE / BLS / BN254 / KZG: return NOT_SUPPORTED (use CPU fallback)
static LuxBackendError metal_not_supported_ntt(LuxBackendContext*, uint64_t*, size_t, uint64_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_not_supported_msm(LuxBackendContext*, const void*, const void*,
                                                void*, size_t, int) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_not_supported_poly_mul(LuxBackendContext*, const uint64_t*,
                                                     const uint64_t*, uint64_t*, size_t, uint64_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_not_supported_bootstrap(LuxBackendContext*, const uint64_t*,
                                                      uint64_t*, const uint64_t*,
                                                      const uint64_t*, uint32_t, uint32_t,
                                                      uint32_t, uint32_t, uint64_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_not_supported_keyswitch(LuxBackendContext*, const uint64_t*,
                                                      uint64_t*, const uint64_t*, uint32_t,
                                                      uint32_t, uint32_t, uint32_t, uint64_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_not_supported_blind_rotate(LuxBackendContext*, uint64_t*,
                                                         const uint64_t*, const uint64_t*,
                                                         uint32_t, uint32_t, uint32_t,
                                                         uint32_t, uint64_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_not_supported_sample_extract(LuxBackendContext*, const uint64_t*,
                                                           uint64_t*, uint32_t, uint32_t,
                                                           uint64_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_not_supported_sample_ntt(LuxBackendContext*, uint64_t*, size_t,
                                                       uint64_t, double, uint64_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_not_supported_poseidon2(LuxBackendContext*, const uint64_t*,
                                                      uint64_t*, size_t, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_not_supported_blake3(LuxBackendContext*, const uint8_t*,
                                                   uint8_t*, const size_t*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_not_supported_bls_add(LuxBackendContext*, const void*,
                                                    const void*, void*, size_t, bool) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_not_supported_bls_mul(LuxBackendContext*, const void*,
                                                    const void*, void*, size_t, bool) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_not_supported_bls_pairing(LuxBackendContext*, const void*,
                                                        const void*, void*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_not_supported_bn254_add(LuxBackendContext*, const void*,
                                                      const void*, void*, size_t, bool) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_not_supported_bn254_mul(LuxBackendContext*, const void*,
                                                      const void*, void*, size_t, bool) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_not_supported_kzg_commit(LuxBackendContext*, const void*,
                                                       const void*, void*, size_t, int) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_not_supported_kzg_open(LuxBackendContext*, const void*,
                                                     const void*, const void*, void*,
                                                     size_t, int) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}
static LuxBackendError metal_not_supported_kzg_verify(LuxBackendContext*, const void*,
                                                       const void*, const void*, const void*,
                                                       const void*, bool*, int) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

// =============================================================================
// Keccak-256 Hash — dispatches keccak256.metal kernel
// =============================================================================

static LuxBackendError metal_op_keccak256_hash(
    LuxBackendContext* ctx,
    const uint8_t* inputs,
    uint8_t* outputs,
    const size_t* input_lens,
    size_t num_inputs) {

    if (!ctx || !inputs || !outputs || !input_lens || num_inputs == 0)
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    if (!ctx->keccak_lib) {
        // Shader not loaded — cannot dispatch
        return LUX_BACKEND_ERROR_NOT_SUPPORTED;
    }

    id<MTLComputePipelineState> pipeline = get_pipeline(ctx, ctx->keccak_lib, "keccak256_batch");
    if (!pipeline) return LUX_BACKEND_ERROR_INTERNAL;

    @autoreleasepool {
        // Build GPU input descriptors and total data size
        size_t total_data = 0;
        for (size_t i = 0; i < num_inputs; i++)
            total_data += input_lens[i];

        const size_t desc_size = num_inputs * sizeof(GPUHashInput);
        const size_t out_size = num_inputs * 32;

        // Allocate Metal buffers (shared memory — no copy on Apple Silicon)
        id<MTLBuffer> desc_buf = [ctx->device newBufferWithLength:desc_size
                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> data_buf = [ctx->device newBufferWithLength:(total_data > 0 ? total_data : 1)
                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf = [ctx->device newBufferWithLength:out_size
                                             options:MTLResourceStorageModeShared];

        if (!desc_buf || !data_buf || !out_buf) return LUX_BACKEND_ERROR_OUT_OF_MEMORY;

        // Fill descriptor and data buffers
        auto* gpu_descs = static_cast<GPUHashInput*>([desc_buf contents]);
        auto* gpu_data = static_cast<uint8_t*>([data_buf contents]);

        uint32_t offset = 0;
        for (size_t i = 0; i < num_inputs; i++) {
            gpu_descs[i].offset = offset;
            gpu_descs[i].length = static_cast<uint32_t>(input_lens[i]);
            if (input_lens[i] > 0)
                std::memcpy(gpu_data + offset, inputs + offset, input_lens[i]);
            offset += static_cast<uint32_t>(input_lens[i]);
        }

        // Dispatch
        id<MTLBuffer> bufs[3] = { desc_buf, data_buf, out_buf };
        LuxBackendError err = dispatch_1d(ctx, pipeline, bufs, 3, num_inputs);
        if (err != LUX_BACKEND_OK) return err;

        // Read back results (on unified memory this is just a memcpy from same physical page)
        std::memcpy(outputs, [out_buf contents], out_size);
    }
    return LUX_BACKEND_OK;
}

// =============================================================================
// secp256k1 ECDSA Recovery — dispatches secp256k1_recover.metal kernel
// =============================================================================

static LuxBackendError metal_op_ecrecover_batch(
    LuxBackendContext* ctx,
    const void* signatures,
    void* addresses,
    size_t num_signatures) {

    if (!ctx || !signatures || !addresses)
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    if (num_signatures == 0)
        return LUX_BACKEND_OK;

    if (!ctx->ecrecover_lib) {
        return LUX_BACKEND_ERROR_NOT_SUPPORTED;
    }

    id<MTLComputePipelineState> pipeline =
        get_pipeline(ctx, ctx->ecrecover_lib, "secp256k1_ecrecover_batch");
    if (!pipeline) return LUX_BACKEND_ERROR_INTERNAL;

    @autoreleasepool {
        const size_t in_size = num_signatures * 128;   // LuxEcrecoverInput = 128 bytes
        const size_t out_size = num_signatures * 32;    // LuxEcrecoverOutput = 32 bytes

        id<MTLBuffer> in_buf = [ctx->device newBufferWithBytes:signatures
                                            length:in_size
                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf = [ctx->device newBufferWithLength:out_size
                                             options:MTLResourceStorageModeShared];

        if (!in_buf || !out_buf) return LUX_BACKEND_ERROR_OUT_OF_MEMORY;

        // Zero the output buffer
        std::memset([out_buf contents], 0, out_size);

        id<MTLBuffer> bufs[2] = { in_buf, out_buf };
        LuxBackendError err = dispatch_1d(ctx, pipeline, bufs, 2, num_signatures);
        if (err != LUX_BACKEND_OK) return err;

        std::memcpy(addresses, [out_buf contents], out_size);
    }
    return LUX_BACKEND_OK;
}

// =============================================================================
// Metal Backend VTable
// =============================================================================

static const lux_gpu_backend_vtbl metal_vtbl = {
    // Lifecycle
    .create_context = metal_create_context,
    .destroy_context = metal_destroy_context,

    // Device info
    .get_device_count = metal_get_device_count,
    .get_device_info = metal_get_device_info,

    // Sync
    .sync = metal_sync,

    // Buffer management
    .buffer_alloc = metal_buffer_alloc,
    .buffer_alloc_with_data = metal_buffer_alloc_with_data,
    .buffer_free = metal_buffer_free,
    .buffer_copy_to_host = metal_buffer_copy_to_host,
    .buffer_copy_from_host = metal_buffer_copy_from_host,
    .buffer_get_host_ptr = metal_buffer_get_host_ptr,

    // Custom kernels
    .kernel_load = metal_kernel_load,
    .kernel_load_binary = metal_kernel_load_binary,
    .kernel_destroy = metal_kernel_destroy,
    .kernel_dispatch = metal_kernel_dispatch,

    // Tensor ops
    .op_add_f32 = metal_op_add_f32,
    .op_sub_f32 = metal_op_sub_f32,
    .op_mul_f32 = metal_op_mul_f32,
    .op_div_f32 = metal_op_div_f32,

    // Matrix ops
    .op_matmul_f32 = metal_op_matmul_f32,
    .op_transpose_f32 = metal_op_transpose_f32,

    // Reductions
    .op_reduce_sum_f32 = metal_op_reduce_sum_f32,
    .op_reduce_max_f32 = metal_op_reduce_max_f32,
    .op_reduce_min_f32 = metal_op_reduce_min_f32,
    .op_reduce_mean_f32 = metal_op_reduce_mean_f32,
    .op_reduce_sum_axis_f32 = metal_op_reduce_sum_axis_f32,
    .op_reduce_max_axis_f32 = metal_op_reduce_max_axis_f32,

    // Softmax
    .op_softmax_f32 = metal_op_softmax_f32,
    .op_log_softmax_f32 = metal_op_log_softmax_f32,

    // Unary ops
    .op_exp_f32 = metal_op_exp_f32,
    .op_log_f32 = metal_op_log_f32,
    .op_sqrt_f32 = metal_op_sqrt_f32,
    .op_neg_f32 = metal_op_neg_f32,
    .op_abs_f32 = metal_op_abs_f32,
    .op_tanh_f32 = metal_op_tanh_f32,
    .op_sigmoid_f32 = metal_op_sigmoid_f32,
    .op_relu_f32 = metal_op_relu_f32,
    .op_gelu_f32 = metal_op_gelu_f32,

    // Copy
    .op_copy_f32 = metal_op_copy_f32,

    // Normalization
    .op_layer_norm_f32 = metal_op_layer_norm_f32,
    .op_rms_norm_f32 = metal_op_rms_norm_f32,

    // NTT
    .op_ntt_forward = metal_not_supported_ntt,
    .op_ntt_inverse = metal_not_supported_ntt,

    // MSM
    .op_msm = metal_not_supported_msm,

    // FHE
    .op_poly_mul = metal_not_supported_poly_mul,
    .op_tfhe_bootstrap = metal_not_supported_bootstrap,
    .op_tfhe_keyswitch = metal_not_supported_keyswitch,
    .op_blind_rotate = metal_not_supported_blind_rotate,
    .op_sample_extract = metal_not_supported_sample_extract,
    .op_sample_ntt = metal_not_supported_sample_ntt,

    // Crypto hashes
    .op_poseidon2_hash = metal_not_supported_poseidon2,
    .op_blake3_hash = metal_not_supported_blake3,
    .op_keccak256_hash = metal_op_keccak256_hash,

    // BLS12-381
    .op_bls12_381_add = metal_not_supported_bls_add,
    .op_bls12_381_mul = metal_not_supported_bls_mul,
    .op_bls12_381_pairing = metal_not_supported_bls_pairing,

    // BN254
    .op_bn254_add = metal_not_supported_bn254_add,
    .op_bn254_mul = metal_not_supported_bn254_mul,

    // KZG
    .op_kzg_commit = metal_not_supported_kzg_commit,
    .op_kzg_open = metal_not_supported_kzg_open,
    .op_kzg_verify = metal_not_supported_kzg_verify,

    // secp256k1 ecrecover
    .op_ecrecover_batch = metal_op_ecrecover_batch,

    // Reserved
    ._reserved = {nullptr, nullptr, nullptr},
};

// =============================================================================
// Entry Point
// =============================================================================

static bool metal_init(lux_gpu_backend_desc* out) {
    if (!out) return false;

    // Verify Metal is actually available on this system
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return false;
    }

    out->abi_version = LUX_GPU_BACKEND_ABI_VERSION;
    out->backend_name = "metal";
    out->backend_version = "0.2.0";
    out->capabilities = LUX_CAP_TENSOR_OPS | LUX_CAP_UNIFIED_MEMORY
                       | LUX_CAP_CUSTOM_KERNELS | LUX_CAP_UNARY
                       | LUX_CAP_KECCAK256 | LUX_CAP_ECRECOVER;
    out->vtbl = &metal_vtbl;
    return true;
}

LUX_GPU_DECLARE_BACKEND(metal_init)

#endif // __APPLE__
