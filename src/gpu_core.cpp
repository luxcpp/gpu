// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Core GPU Library - Plugin-based backend management

#include "lux/gpu.h"
#include "lux/gpu/backend_plugin.h"
#include "plugin_loader.hpp"
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>

// =============================================================================
// Built-in CPU backend declaration
// =============================================================================

extern "C" bool cpu_backend_init(lux_gpu_backend_desc* out);

// =============================================================================
// GPU Context Implementation
// =============================================================================

struct LuxGPU {
    std::string backend_name;
    const lux_gpu_backend_vtbl* vtbl = nullptr;
    LuxBackendContext* ctx = nullptr;
    std::string last_error;
    std::mutex mutex;

    ~LuxGPU() {
        if (ctx && vtbl && vtbl->destroy_context) {
            vtbl->destroy_context(ctx);
        }
    }

    void set_error(const char* msg) {
        std::lock_guard<std::mutex> lock(mutex);
        last_error = msg ? msg : "";
    }
};

// =============================================================================
// Tensor wrapper (bridges plugin buffers to public API)
// =============================================================================

struct LuxTensor {
    std::vector<int64_t> shape;
    LuxDtype dtype;
    std::vector<uint8_t> host_data;
    LuxBackendBuffer* device_buffer = nullptr;
    const lux_gpu_backend_vtbl* vtbl = nullptr;
    LuxBackendContext* ctx = nullptr;

    int64_t size() const {
        int64_t s = 1;
        for (auto d : shape) s *= d;
        return s;
    }

    size_t element_size() const {
        switch (dtype) {
            case LUX_FLOAT32: case LUX_INT32: case LUX_UINT32: return 4;
            case LUX_FLOAT16: case LUX_BFLOAT16: return 2;
            case LUX_INT64: case LUX_UINT64: return 8;
            case LUX_BOOL: return 1;
            default: return 0;
        }
    }

    ~LuxTensor() {
        if (device_buffer && vtbl && vtbl->buffer_free && ctx) {
            vtbl->buffer_free(ctx, device_buffer);
        }
    }
};

// =============================================================================
// Global initialization
// =============================================================================

static std::once_flag g_init_flag;
static lux_gpu_backend_desc g_cpu_backend = {};

static void global_init() {
    // Initialize built-in CPU backend
    cpu_backend_init(&g_cpu_backend);

    // Scan for plugins in all search paths
    auto& loader = lux::gpu::PluginLoader::instance();

    // Try to load each backend (will search all paths)
    loader.load_backend("metal");
    loader.load_backend("cuda");
    loader.load_backend("webgpu");
}

// =============================================================================
// C API Implementation
// =============================================================================

extern "C" {

LuxGPU* lux_gpu_create(void) {
    return lux_gpu_create_with_backend(LUX_BACKEND_AUTO);
}

LuxGPU* lux_gpu_create_with_backend(LuxBackend backend) {
    return lux_gpu_create_with_device(backend, 0);
}

LuxGPU* lux_gpu_create_with_device(LuxBackend backend, int device_index) {
    std::call_once(g_init_flag, global_init);

    auto gpu = new LuxGPU();
    auto& loader = lux::gpu::PluginLoader::instance();

    const lux_gpu_backend_vtbl* vtbl = nullptr;
    std::string name;

    if (backend == LUX_BACKEND_AUTO) {
        // Try to find the best available backend
        if (auto* best = loader.get_best_backend()) {
            vtbl = best->desc.vtbl;
            name = best->name;
        }
        // Fall back to CPU
        if (!vtbl) {
            vtbl = g_cpu_backend.vtbl;
            name = "cpu";
        }
    } else {
        // Specific backend requested
        switch (backend) {
            case LUX_BACKEND_CPU:
                vtbl = g_cpu_backend.vtbl;
                name = "cpu";
                break;

            case LUX_BACKEND_METAL:
                if (!loader.is_available("metal")) {
                    loader.load_backend("metal");
                }
                if (auto* b = loader.get_backend("metal")) {
                    vtbl = b->desc.vtbl;
                    name = "metal";
                }
                break;

            case LUX_BACKEND_CUDA:
                if (!loader.is_available("cuda")) {
                    loader.load_backend("cuda");
                }
                if (auto* b = loader.get_backend("cuda")) {
                    vtbl = b->desc.vtbl;
                    name = "cuda";
                }
                break;

            case LUX_BACKEND_DAWN:
                if (!loader.is_available("webgpu")) {
                    loader.load_backend("webgpu");
                }
                if (auto* b = loader.get_backend("webgpu")) {
                    vtbl = b->desc.vtbl;
                    name = "webgpu";
                }
                break;

            default:
                break;
        }
    }

    if (!vtbl) {
        // Fall back to CPU
        vtbl = g_cpu_backend.vtbl;
        name = "cpu";
    }

    gpu->vtbl = vtbl;
    gpu->backend_name = name;

    // Create context
    if (vtbl && vtbl->create_context) {
        gpu->ctx = vtbl->create_context(device_index);
        if (!gpu->ctx) {
            gpu->set_error("Failed to create backend context");
            // Fall back to CPU
            gpu->vtbl = g_cpu_backend.vtbl;
            gpu->backend_name = "cpu";
            gpu->ctx = g_cpu_backend.vtbl->create_context(0);
        }
    }

    return gpu;
}

void lux_gpu_destroy(LuxGPU* gpu) {
    delete gpu;
}

LuxBackend lux_gpu_backend(LuxGPU* gpu) {
    if (!gpu) return LUX_BACKEND_CPU;
    if (gpu->backend_name == "cpu") return LUX_BACKEND_CPU;
    if (gpu->backend_name == "metal") return LUX_BACKEND_METAL;
    if (gpu->backend_name == "cuda") return LUX_BACKEND_CUDA;
    if (gpu->backend_name == "webgpu") return LUX_BACKEND_DAWN;
    return LUX_BACKEND_CPU;
}

const char* lux_gpu_backend_name(LuxGPU* gpu) {
    return gpu ? gpu->backend_name.c_str() : "cpu";
}

LuxError lux_gpu_set_backend(LuxGPU* gpu, LuxBackend backend) {
    if (!gpu) return LUX_ERROR_INVALID_ARGUMENT;

    // Create new context for requested backend
    auto* new_gpu = lux_gpu_create_with_backend(backend);
    if (!new_gpu || new_gpu->backend_name == "cpu" && backend != LUX_BACKEND_CPU && backend != LUX_BACKEND_AUTO) {
        delete new_gpu;
        return LUX_ERROR_BACKEND_NOT_AVAILABLE;
    }

    // Swap internals
    std::swap(gpu->vtbl, new_gpu->vtbl);
    std::swap(gpu->ctx, new_gpu->ctx);
    std::swap(gpu->backend_name, new_gpu->backend_name);

    delete new_gpu;
    return LUX_OK;
}

LuxError lux_gpu_device_info(LuxGPU* gpu, LuxDeviceInfo* info) {
    if (!gpu || !gpu->vtbl || !info) return LUX_ERROR_INVALID_ARGUMENT;

    LuxBackendDeviceInfo binfo = {};
    LuxBackendError err = gpu->vtbl->get_device_info(gpu->ctx, &binfo);
    if (err != LUX_BACKEND_OK) return static_cast<LuxError>(err);

    info->backend = lux_gpu_backend(gpu);
    info->index = 0;
    info->name = binfo.name;
    info->vendor = binfo.vendor;
    info->memory_total = binfo.memory_total;
    info->memory_available = binfo.memory_available;
    info->compute_units = binfo.compute_units;
    info->max_workgroup_size = binfo.max_workgroup_size;
    info->is_discrete = binfo.is_discrete;
    info->is_unified_memory = binfo.is_unified_memory;

    return LUX_OK;
}

LuxError lux_gpu_sync(LuxGPU* gpu) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    return static_cast<LuxError>(gpu->vtbl->sync(gpu->ctx));
}

const char* lux_gpu_error(LuxGPU* gpu) {
    return gpu ? gpu->last_error.c_str() : "null gpu";
}

// Backend query
int lux_backend_count(void) {
    std::call_once(g_init_flag, global_init);
    return static_cast<int>(lux::gpu::PluginLoader::instance().available_backends().size());
}

bool lux_backend_available(LuxBackend backend) {
    std::call_once(g_init_flag, global_init);
    auto& loader = lux::gpu::PluginLoader::instance();

    switch (backend) {
        case LUX_BACKEND_CPU: return true;
        case LUX_BACKEND_METAL: return loader.is_available("metal") || loader.load_backend("metal");
        case LUX_BACKEND_CUDA: return loader.is_available("cuda") || loader.load_backend("cuda");
        case LUX_BACKEND_DAWN: return loader.is_available("webgpu") || loader.load_backend("webgpu");
        default: return false;
    }
}

const char* lux_backend_name(LuxBackend backend) {
    switch (backend) {
        case LUX_BACKEND_AUTO: return "auto";
        case LUX_BACKEND_CPU: return "cpu";
        case LUX_BACKEND_METAL: return "metal";
        case LUX_BACKEND_CUDA: return "cuda";
        case LUX_BACKEND_DAWN: return "webgpu";
        default: return "unknown";
    }
}

// =============================================================================
// Tensor Operations
// =============================================================================

LuxTensor* lux_tensor_zeros(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype) {
    if (!gpu || !gpu->vtbl || !shape || ndim <= 0) return nullptr;

    auto t = new LuxTensor();
    t->shape.assign(shape, shape + ndim);
    t->dtype = dtype;
    t->vtbl = gpu->vtbl;
    t->ctx = gpu->ctx;

    size_t bytes = t->size() * t->element_size();
    t->host_data.resize(bytes, 0);

    if (gpu->vtbl->buffer_alloc_with_data) {
        t->device_buffer = gpu->vtbl->buffer_alloc_with_data(gpu->ctx, t->host_data.data(), bytes);
    } else if (gpu->vtbl->buffer_alloc) {
        t->device_buffer = gpu->vtbl->buffer_alloc(gpu->ctx, bytes);
    }

    return t;
}

LuxTensor* lux_tensor_ones(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype) {
    return lux_tensor_full(gpu, shape, ndim, dtype, 1.0);
}

LuxTensor* lux_tensor_full(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype, double value) {
    if (!gpu || !gpu->vtbl || !shape || ndim <= 0) return nullptr;

    auto t = new LuxTensor();
    t->shape.assign(shape, shape + ndim);
    t->dtype = dtype;
    t->vtbl = gpu->vtbl;
    t->ctx = gpu->ctx;

    size_t bytes = t->size() * t->element_size();
    t->host_data.resize(bytes);

    // Fill host data
    if (dtype == LUX_FLOAT32) {
        float v = static_cast<float>(value);
        float* ptr = reinterpret_cast<float*>(t->host_data.data());
        for (int64_t i = 0; i < t->size(); i++) ptr[i] = v;
    }

    // Copy to device
    if (gpu->vtbl->buffer_alloc_with_data) {
        t->device_buffer = gpu->vtbl->buffer_alloc_with_data(gpu->ctx, t->host_data.data(), bytes);
    }

    return t;
}

LuxTensor* lux_tensor_from_data(LuxGPU* gpu, const void* data, const int64_t* shape, int ndim, LuxDtype dtype) {
    if (!gpu || !gpu->vtbl || !data || !shape || ndim <= 0) return nullptr;

    auto t = new LuxTensor();
    t->shape.assign(shape, shape + ndim);
    t->dtype = dtype;
    t->vtbl = gpu->vtbl;
    t->ctx = gpu->ctx;

    size_t bytes = t->size() * t->element_size();
    t->host_data.resize(bytes);
    std::memcpy(t->host_data.data(), data, bytes);

    if (gpu->vtbl->buffer_alloc_with_data) {
        t->device_buffer = gpu->vtbl->buffer_alloc_with_data(gpu->ctx, data, bytes);
    }

    return t;
}

void lux_tensor_destroy(LuxTensor* tensor) {
    delete tensor;
}

int lux_tensor_ndim(LuxTensor* tensor) {
    return tensor ? static_cast<int>(tensor->shape.size()) : 0;
}

int64_t lux_tensor_shape(LuxTensor* tensor, int dim) {
    return (tensor && dim >= 0 && dim < static_cast<int>(tensor->shape.size()))
        ? tensor->shape[dim] : 0;
}

int64_t lux_tensor_size(LuxTensor* tensor) {
    return tensor ? tensor->size() : 0;
}

LuxDtype lux_tensor_dtype(LuxTensor* tensor) {
    return tensor ? tensor->dtype : LUX_FLOAT32;
}

LuxError lux_tensor_to_host(LuxTensor* tensor, void* data, size_t size) {
    if (!tensor || !data) return LUX_ERROR_INVALID_ARGUMENT;

    size_t bytes = tensor->size() * tensor->element_size();
    if (size < bytes) return LUX_ERROR_INVALID_ARGUMENT;

    // If we have device buffer, sync from it
    if (tensor->device_buffer && tensor->vtbl && tensor->vtbl->buffer_copy_to_host) {
        LuxBackendError err = tensor->vtbl->buffer_copy_to_host(
            tensor->ctx, tensor->device_buffer, data, bytes
        );
        return static_cast<LuxError>(err);
    }

    // Otherwise copy from host data
    std::memcpy(data, tensor->host_data.data(), bytes);
    return LUX_OK;
}

// Binary operations helper
static LuxTensor* binary_op(LuxGPU* gpu, LuxTensor* a, LuxTensor* b,
                            LuxBackendError (*op)(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, LuxBackendBuffer*, size_t)) {
    if (!gpu || !a || !b || a->shape != b->shape) return nullptr;

    auto out = lux_tensor_zeros(gpu, a->shape.data(), static_cast<int>(a->shape.size()), a->dtype);
    if (!out) return nullptr;

    if (op && a->device_buffer && b->device_buffer && out->device_buffer) {
        LuxBackendError err = op(gpu->ctx, a->device_buffer, b->device_buffer, out->device_buffer, a->size());
        if (err != LUX_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

LuxTensor* lux_tensor_add(LuxGPU* gpu, LuxTensor* a, LuxTensor* b) {
    if (!gpu || !gpu->vtbl) return nullptr;
    return binary_op(gpu, a, b, gpu->vtbl->op_add_f32);
}

LuxTensor* lux_tensor_sub(LuxGPU* gpu, LuxTensor* a, LuxTensor* b) {
    if (!gpu || !gpu->vtbl) return nullptr;
    return binary_op(gpu, a, b, gpu->vtbl->op_sub_f32);
}

LuxTensor* lux_tensor_mul(LuxGPU* gpu, LuxTensor* a, LuxTensor* b) {
    if (!gpu || !gpu->vtbl) return nullptr;
    return binary_op(gpu, a, b, gpu->vtbl->op_mul_f32);
}

LuxTensor* lux_tensor_div(LuxGPU* gpu, LuxTensor* a, LuxTensor* b) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_div_f32) return nullptr;
    return binary_op(gpu, a, b, gpu->vtbl->op_div_f32);
}

LuxTensor* lux_tensor_matmul(LuxGPU* gpu, LuxTensor* a, LuxTensor* b) {
    if (!gpu || !gpu->vtbl || !a || !b) return nullptr;
    if (a->shape.size() != 2 || b->shape.size() != 2) return nullptr;
    if (a->shape[1] != b->shape[0]) return nullptr;

    int M = static_cast<int>(a->shape[0]);
    int K = static_cast<int>(a->shape[1]);
    int N = static_cast<int>(b->shape[1]);

    int64_t out_shape[2] = {M, N};
    auto out = lux_tensor_zeros(gpu, out_shape, 2, a->dtype);
    if (!out) return nullptr;

    if (gpu->vtbl->op_matmul_f32 && a->device_buffer && b->device_buffer && out->device_buffer) {
        LuxBackendError err = gpu->vtbl->op_matmul_f32(
            gpu->ctx, a->device_buffer, b->device_buffer, out->device_buffer, M, K, N
        );
        if (err != LUX_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

// =============================================================================
// Unary operations helper
// =============================================================================

static LuxTensor* unary_op(LuxGPU* gpu, LuxTensor* t,
                           LuxBackendError (*op)(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t)) {
    if (!gpu || !gpu->vtbl || !t || !op) return nullptr;

    auto out = lux_tensor_zeros(gpu, t->shape.data(), static_cast<int>(t->shape.size()), t->dtype);
    if (!out) return nullptr;

    if (t->device_buffer && out->device_buffer) {
        LuxBackendError err = op(gpu->ctx, t->device_buffer, out->device_buffer, t->size());
        if (err != LUX_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

LuxTensor* lux_tensor_neg(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_neg_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_neg_f32);
}

LuxTensor* lux_tensor_exp(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_exp_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_exp_f32);
}

LuxTensor* lux_tensor_log(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_log_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_log_f32);
}

LuxTensor* lux_tensor_sqrt(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_sqrt_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_sqrt_f32);
}

LuxTensor* lux_tensor_abs(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_abs_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_abs_f32);
}

LuxTensor* lux_tensor_tanh(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_tanh_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_tanh_f32);
}

LuxTensor* lux_tensor_sigmoid(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_sigmoid_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_sigmoid_f32);
}

LuxTensor* lux_tensor_relu(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_relu_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_relu_f32);
}

LuxTensor* lux_tensor_gelu(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_gelu_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_gelu_f32);
}

// =============================================================================
// Scalar reductions
// =============================================================================

float lux_tensor_reduce_sum(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_reduce_sum_f32) return 0.0f;

    int64_t one = 1;
    auto out = lux_tensor_zeros(gpu, &one, 1, LUX_FLOAT32);
    if (!out) return 0.0f;

    if (t->device_buffer && out->device_buffer) {
        gpu->vtbl->op_reduce_sum_f32(gpu->ctx, t->device_buffer, out->device_buffer, t->size());
    }

    float result = 0.0f;
    lux_tensor_to_host(out, &result, sizeof(float));
    lux_tensor_destroy(out);
    return result;
}

float lux_tensor_reduce_max(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_reduce_max_f32) return 0.0f;

    int64_t one = 1;
    auto out = lux_tensor_zeros(gpu, &one, 1, LUX_FLOAT32);
    if (!out) return 0.0f;

    if (t->device_buffer && out->device_buffer) {
        gpu->vtbl->op_reduce_max_f32(gpu->ctx, t->device_buffer, out->device_buffer, t->size());
    }

    float result = 0.0f;
    lux_tensor_to_host(out, &result, sizeof(float));
    lux_tensor_destroy(out);
    return result;
}

float lux_tensor_reduce_min(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_reduce_min_f32) return 0.0f;

    int64_t one = 1;
    auto out = lux_tensor_zeros(gpu, &one, 1, LUX_FLOAT32);
    if (!out) return 0.0f;

    if (t->device_buffer && out->device_buffer) {
        gpu->vtbl->op_reduce_min_f32(gpu->ctx, t->device_buffer, out->device_buffer, t->size());
    }

    float result = 0.0f;
    lux_tensor_to_host(out, &result, sizeof(float));
    lux_tensor_destroy(out);
    return result;
}

float lux_tensor_reduce_mean(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_reduce_mean_f32) return 0.0f;

    int64_t one = 1;
    auto out = lux_tensor_zeros(gpu, &one, 1, LUX_FLOAT32);
    if (!out) return 0.0f;

    if (t->device_buffer && out->device_buffer) {
        gpu->vtbl->op_reduce_mean_f32(gpu->ctx, t->device_buffer, out->device_buffer, t->size());
    }

    float result = 0.0f;
    lux_tensor_to_host(out, &result, sizeof(float));
    lux_tensor_destroy(out);
    return result;
}

// =============================================================================
// Axis Reduction Helpers
// =============================================================================

// Compute output shape after reducing along specified axes
static std::vector<int64_t> compute_reduced_shape(const std::vector<int64_t>& shape, const int* axes, int naxes) {
    std::vector<bool> reduce_axis(shape.size(), false);
    for (int i = 0; i < naxes; i++) {
        int ax = axes[i];
        if (ax < 0) ax += static_cast<int>(shape.size());
        if (ax >= 0 && ax < static_cast<int>(shape.size())) {
            reduce_axis[ax] = true;
        }
    }

    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < shape.size(); i++) {
        if (!reduce_axis[i]) {
            out_shape.push_back(shape[i]);
        }
    }
    if (out_shape.empty()) {
        out_shape.push_back(1);  // Scalar result
    }
    return out_shape;
}

// Compute outer_size and inner_size for contiguous last-axis reduction
// Returns true if reduction can be expressed as (outer_size, inner_size) -> outer_size
static bool can_use_axis_reduction(const std::vector<int64_t>& shape, const int* axes, int naxes,
                                    size_t* outer_size, size_t* inner_size) {
    if (naxes != 1 || shape.empty()) return false;

    int ax = axes[0];
    if (ax < 0) ax += static_cast<int>(shape.size());
    if (ax < 0 || ax >= static_cast<int>(shape.size())) return false;

    // Only support reduction along last axis for backend dispatch
    if (ax != static_cast<int>(shape.size()) - 1) return false;

    *inner_size = static_cast<size_t>(shape.back());
    *outer_size = 1;
    for (size_t i = 0; i < shape.size() - 1; i++) {
        *outer_size *= static_cast<size_t>(shape[i]);
    }
    return true;
}

// CPU fallback for sum reduction along axes
static void cpu_reduce_sum_axes(const float* in, float* out, const std::vector<int64_t>& shape,
                                 const int* axes, int naxes, const std::vector<int64_t>& out_shape) {
    std::vector<bool> reduce_axis(shape.size(), false);
    for (int i = 0; i < naxes; i++) {
        int ax = axes[i];
        if (ax < 0) ax += static_cast<int>(shape.size());
        if (ax >= 0 && ax < static_cast<int>(shape.size())) {
            reduce_axis[ax] = true;
        }
    }

    // Compute strides for input
    std::vector<size_t> strides(shape.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= static_cast<size_t>(shape[i]);
    }

    // Compute output size
    size_t out_size = 1;
    for (auto d : out_shape) out_size *= static_cast<size_t>(d);
    std::memset(out, 0, out_size * sizeof(float));

    // Iterate over input and accumulate
    size_t in_size = stride;
    for (size_t idx = 0; idx < in_size; idx++) {
        // Decompose linear index into multi-index
        size_t tmp = idx;
        size_t out_idx = 0;
        size_t out_stride = 1;
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; d--) {
            size_t coord = tmp % static_cast<size_t>(shape[d]);
            tmp /= static_cast<size_t>(shape[d]);
            if (!reduce_axis[d]) {
                // Find position in output
                int out_d = 0;
                for (int k = 0; k < d; k++) {
                    if (!reduce_axis[k]) out_d++;
                }
                // Compute contribution to out_idx
                size_t os = 1;
                for (size_t k = out_d + 1; k < out_shape.size(); k++) {
                    os *= static_cast<size_t>(out_shape[k]);
                }
                out_idx += coord * os;
            }
        }
        out[out_idx] += in[idx];
    }
}

// CPU fallback for max reduction along axes
static void cpu_reduce_max_axes(const float* in, float* out, const std::vector<int64_t>& shape,
                                 const int* axes, int naxes, const std::vector<int64_t>& out_shape) {
    std::vector<bool> reduce_axis(shape.size(), false);
    for (int i = 0; i < naxes; i++) {
        int ax = axes[i];
        if (ax < 0) ax += static_cast<int>(shape.size());
        if (ax >= 0 && ax < static_cast<int>(shape.size())) {
            reduce_axis[ax] = true;
        }
    }

    std::vector<size_t> strides(shape.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= static_cast<size_t>(shape[i]);
    }

    size_t out_size = 1;
    for (auto d : out_shape) out_size *= static_cast<size_t>(d);

    // Initialize with -inf
    for (size_t i = 0; i < out_size; i++) {
        out[i] = -std::numeric_limits<float>::infinity();
    }

    size_t in_size = stride;
    for (size_t idx = 0; idx < in_size; idx++) {
        size_t tmp = idx;
        size_t out_idx = 0;
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; d--) {
            size_t coord = tmp % static_cast<size_t>(shape[d]);
            tmp /= static_cast<size_t>(shape[d]);
            if (!reduce_axis[d]) {
                int out_d = 0;
                for (int k = 0; k < d; k++) {
                    if (!reduce_axis[k]) out_d++;
                }
                size_t os = 1;
                for (size_t k = out_d + 1; k < out_shape.size(); k++) {
                    os *= static_cast<size_t>(out_shape[k]);
                }
                out_idx += coord * os;
            }
        }
        if (in[idx] > out[out_idx]) {
            out[out_idx] = in[idx];
        }
    }
}

// CPU fallback for min reduction along axes
static void cpu_reduce_min_axes(const float* in, float* out, const std::vector<int64_t>& shape,
                                 const int* axes, int naxes, const std::vector<int64_t>& out_shape) {
    std::vector<bool> reduce_axis(shape.size(), false);
    for (int i = 0; i < naxes; i++) {
        int ax = axes[i];
        if (ax < 0) ax += static_cast<int>(shape.size());
        if (ax >= 0 && ax < static_cast<int>(shape.size())) {
            reduce_axis[ax] = true;
        }
    }

    std::vector<size_t> strides(shape.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= static_cast<size_t>(shape[i]);
    }

    size_t out_size = 1;
    for (auto d : out_shape) out_size *= static_cast<size_t>(d);

    for (size_t i = 0; i < out_size; i++) {
        out[i] = std::numeric_limits<float>::infinity();
    }

    size_t in_size = stride;
    for (size_t idx = 0; idx < in_size; idx++) {
        size_t tmp = idx;
        size_t out_idx = 0;
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; d--) {
            size_t coord = tmp % static_cast<size_t>(shape[d]);
            tmp /= static_cast<size_t>(shape[d]);
            if (!reduce_axis[d]) {
                int out_d = 0;
                for (int k = 0; k < d; k++) {
                    if (!reduce_axis[k]) out_d++;
                }
                size_t os = 1;
                for (size_t k = out_d + 1; k < out_shape.size(); k++) {
                    os *= static_cast<size_t>(out_shape[k]);
                }
                out_idx += coord * os;
            }
        }
        if (in[idx] < out[out_idx]) {
            out[out_idx] = in[idx];
        }
    }
}

// =============================================================================
// Axis Reduction API
// =============================================================================

LuxTensor* lux_tensor_sum(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes) {
    if (!gpu || !gpu->vtbl || !t || !axes || naxes <= 0) return nullptr;

    std::vector<int64_t> out_shape = compute_reduced_shape(t->shape, axes, naxes);
    auto out = lux_tensor_zeros(gpu, out_shape.data(), static_cast<int>(out_shape.size()), t->dtype);
    if (!out) return nullptr;

    // Try backend dispatch for single last-axis reduction
    size_t outer_size, inner_size;
    if (gpu->vtbl->op_reduce_sum_axis_f32 &&
        can_use_axis_reduction(t->shape, axes, naxes, &outer_size, &inner_size) &&
        t->device_buffer && out->device_buffer) {
        LuxBackendError err = gpu->vtbl->op_reduce_sum_axis_f32(
            gpu->ctx, t->device_buffer, out->device_buffer, outer_size, inner_size);
        if (err == LUX_BACKEND_OK) return out;
    }

    // CPU fallback: sync input to host, compute, sync output to device
    std::vector<float> in_data(t->size());
    lux_tensor_to_host(t, in_data.data(), in_data.size() * sizeof(float));

    std::vector<float> out_data(out->size());
    cpu_reduce_sum_axes(in_data.data(), out_data.data(), t->shape, axes, naxes, out_shape);

    // Copy result to output tensor host_data and device
    std::memcpy(out->host_data.data(), out_data.data(), out_data.size() * sizeof(float));
    if (out->device_buffer && gpu->vtbl->buffer_copy_from_host) {
        gpu->vtbl->buffer_copy_from_host(gpu->ctx, out->device_buffer,
                                          out_data.data(), out_data.size() * sizeof(float));
    }

    return out;
}

LuxTensor* lux_tensor_mean(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes) {
    if (!gpu || !gpu->vtbl || !t || !axes || naxes <= 0) return nullptr;

    // Compute sum first
    LuxTensor* sum_tensor = lux_tensor_sum(gpu, t, axes, naxes);
    if (!sum_tensor) return nullptr;

    // Compute reduction factor
    int64_t reduce_count = 1;
    for (int i = 0; i < naxes; i++) {
        int ax = axes[i];
        if (ax < 0) ax += static_cast<int>(t->shape.size());
        if (ax >= 0 && ax < static_cast<int>(t->shape.size())) {
            reduce_count *= t->shape[ax];
        }
    }

    // Divide by reduction count (in-place on host data)
    std::vector<float> data(sum_tensor->size());
    lux_tensor_to_host(sum_tensor, data.data(), data.size() * sizeof(float));

    float scale = 1.0f / static_cast<float>(reduce_count);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] *= scale;
    }

    std::memcpy(sum_tensor->host_data.data(), data.data(), data.size() * sizeof(float));
    if (sum_tensor->device_buffer && gpu->vtbl->buffer_copy_from_host) {
        gpu->vtbl->buffer_copy_from_host(gpu->ctx, sum_tensor->device_buffer,
                                          data.data(), data.size() * sizeof(float));
    }

    return sum_tensor;
}

LuxTensor* lux_tensor_max(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes) {
    if (!gpu || !gpu->vtbl || !t || !axes || naxes <= 0) return nullptr;

    std::vector<int64_t> out_shape = compute_reduced_shape(t->shape, axes, naxes);
    auto out = lux_tensor_zeros(gpu, out_shape.data(), static_cast<int>(out_shape.size()), t->dtype);
    if (!out) return nullptr;

    // Try backend dispatch for single last-axis reduction
    size_t outer_size, inner_size;
    if (gpu->vtbl->op_reduce_max_axis_f32 &&
        can_use_axis_reduction(t->shape, axes, naxes, &outer_size, &inner_size) &&
        t->device_buffer && out->device_buffer) {
        LuxBackendError err = gpu->vtbl->op_reduce_max_axis_f32(
            gpu->ctx, t->device_buffer, out->device_buffer, outer_size, inner_size);
        if (err == LUX_BACKEND_OK) return out;
    }

    // CPU fallback
    std::vector<float> in_data(t->size());
    lux_tensor_to_host(t, in_data.data(), in_data.size() * sizeof(float));

    std::vector<float> out_data(out->size());
    cpu_reduce_max_axes(in_data.data(), out_data.data(), t->shape, axes, naxes, out_shape);

    std::memcpy(out->host_data.data(), out_data.data(), out_data.size() * sizeof(float));
    if (out->device_buffer && gpu->vtbl->buffer_copy_from_host) {
        gpu->vtbl->buffer_copy_from_host(gpu->ctx, out->device_buffer,
                                          out_data.data(), out_data.size() * sizeof(float));
    }

    return out;
}

LuxTensor* lux_tensor_min(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes) {
    if (!gpu || !gpu->vtbl || !t || !axes || naxes <= 0) return nullptr;

    std::vector<int64_t> out_shape = compute_reduced_shape(t->shape, axes, naxes);
    auto out = lux_tensor_zeros(gpu, out_shape.data(), static_cast<int>(out_shape.size()), t->dtype);
    if (!out) return nullptr;

    // CPU fallback (no backend op_reduce_min_axis_f32 in vtable)
    std::vector<float> in_data(t->size());
    lux_tensor_to_host(t, in_data.data(), in_data.size() * sizeof(float));

    std::vector<float> out_data(out->size());
    cpu_reduce_min_axes(in_data.data(), out_data.data(), t->shape, axes, naxes, out_shape);

    std::memcpy(out->host_data.data(), out_data.data(), out_data.size() * sizeof(float));
    if (out->device_buffer && gpu->vtbl->buffer_copy_from_host) {
        gpu->vtbl->buffer_copy_from_host(gpu->ctx, out->device_buffer,
                                          out_data.data(), out_data.size() * sizeof(float));
    }

    return out;
}

// =============================================================================
// Softmax and normalization
// =============================================================================

LuxTensor* lux_tensor_softmax(LuxGPU* gpu, LuxTensor* t, int axis) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_softmax_f32) return nullptr;
    if (t->shape.size() < 1) return nullptr;

    // For now, only support last axis softmax
    (void)axis;
    size_t cols = static_cast<size_t>(t->shape.back());
    size_t rows = static_cast<size_t>(t->size() / cols);

    auto out = lux_tensor_zeros(gpu, t->shape.data(), static_cast<int>(t->shape.size()), t->dtype);
    if (!out) return nullptr;

    if (t->device_buffer && out->device_buffer) {
        LuxBackendError err = gpu->vtbl->op_softmax_f32(gpu->ctx, t->device_buffer, out->device_buffer, rows, cols);
        if (err != LUX_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

LuxTensor* lux_tensor_log_softmax(LuxGPU* gpu, LuxTensor* t, int axis) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_log_softmax_f32) return nullptr;
    if (t->shape.size() < 1) return nullptr;

    (void)axis;
    size_t cols = static_cast<size_t>(t->shape.back());
    size_t rows = static_cast<size_t>(t->size() / cols);

    auto out = lux_tensor_zeros(gpu, t->shape.data(), static_cast<int>(t->shape.size()), t->dtype);
    if (!out) return nullptr;

    if (t->device_buffer && out->device_buffer) {
        LuxBackendError err = gpu->vtbl->op_log_softmax_f32(gpu->ctx, t->device_buffer, out->device_buffer, rows, cols);
        if (err != LUX_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

LuxTensor* lux_tensor_layer_norm(LuxGPU* gpu, LuxTensor* t, LuxTensor* gamma, LuxTensor* beta, float eps) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_layer_norm_f32) return nullptr;
    if (t->shape.size() < 1) return nullptr;

    size_t dim = static_cast<size_t>(t->shape.back());
    size_t batch_size = static_cast<size_t>(t->size() / dim);

    auto out = lux_tensor_zeros(gpu, t->shape.data(), static_cast<int>(t->shape.size()), t->dtype);
    if (!out) return nullptr;

    LuxBackendBuffer* gamma_buf = gamma ? gamma->device_buffer : nullptr;
    LuxBackendBuffer* beta_buf = beta ? beta->device_buffer : nullptr;

    if (t->device_buffer && out->device_buffer) {
        LuxBackendError err = gpu->vtbl->op_layer_norm_f32(
            gpu->ctx, t->device_buffer, out->device_buffer,
            gamma_buf, beta_buf, batch_size, dim, eps
        );
        if (err != LUX_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

LuxTensor* lux_tensor_rms_norm(LuxGPU* gpu, LuxTensor* t, LuxTensor* weight, float eps) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_rms_norm_f32) return nullptr;
    if (t->shape.size() < 1) return nullptr;

    size_t dim = static_cast<size_t>(t->shape.back());
    size_t batch_size = static_cast<size_t>(t->size() / dim);

    auto out = lux_tensor_zeros(gpu, t->shape.data(), static_cast<int>(t->shape.size()), t->dtype);
    if (!out) return nullptr;

    LuxBackendBuffer* weight_buf = weight ? weight->device_buffer : nullptr;

    if (t->device_buffer && out->device_buffer) {
        LuxBackendError err = gpu->vtbl->op_rms_norm_f32(
            gpu->ctx, t->device_buffer, out->device_buffer,
            weight_buf, batch_size, dim, eps
        );
        if (err != LUX_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

// =============================================================================
// Transpose and copy
// =============================================================================

LuxTensor* lux_tensor_transpose(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_transpose_f32) return nullptr;
    if (t->shape.size() != 2) return nullptr;

    int rows = static_cast<int>(t->shape[0]);
    int cols = static_cast<int>(t->shape[1]);

    int64_t out_shape[2] = {cols, rows};
    auto out = lux_tensor_zeros(gpu, out_shape, 2, t->dtype);
    if (!out) return nullptr;

    if (t->device_buffer && out->device_buffer) {
        LuxBackendError err = gpu->vtbl->op_transpose_f32(gpu->ctx, t->device_buffer, out->device_buffer, rows, cols);
        if (err != LUX_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

LuxTensor* lux_tensor_copy(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_copy_f32) return nullptr;

    auto out = lux_tensor_zeros(gpu, t->shape.data(), static_cast<int>(t->shape.size()), t->dtype);
    if (!out) return nullptr;

    if (t->device_buffer && out->device_buffer) {
        LuxBackendError err = gpu->vtbl->op_copy_f32(gpu->ctx, t->device_buffer, out->device_buffer, t->size());
        if (err != LUX_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

// =============================================================================
// Stream/Event Management
// =============================================================================
//
// Streams provide ordered execution queues. For CPU backend, operations are
// synchronous so streams are lightweight handles that track parent context.
// Events mark points in stream execution for synchronization and timing.

struct LuxStream {
    LuxGPU* gpu;
    bool valid;

    explicit LuxStream(LuxGPU* g) : gpu(g), valid(true) {}
};

struct LuxEvent {
    LuxGPU* gpu;
    bool recorded;
    std::chrono::steady_clock::time_point timestamp;

    explicit LuxEvent(LuxGPU* g) : gpu(g), recorded(false), timestamp() {}
};

LuxStream* lux_stream_create(LuxGPU* gpu) {
    if (!gpu) return nullptr;
    return new LuxStream(gpu);
}

void lux_stream_destroy(LuxStream* stream) {
    delete stream;
}

LuxError lux_stream_sync(LuxStream* stream) {
    if (!stream || !stream->valid) return LUX_ERROR_INVALID_ARGUMENT;
    // For CPU backend, all operations are synchronous - nothing to wait for
    // For GPU backends, this would dispatch to backend sync
    if (stream->gpu && stream->gpu->vtbl && stream->gpu->vtbl->sync) {
        return static_cast<LuxError>(stream->gpu->vtbl->sync(stream->gpu->ctx));
    }
    return LUX_OK;
}

LuxEvent* lux_event_create(LuxGPU* gpu) {
    if (!gpu) return nullptr;
    return new LuxEvent(gpu);
}

void lux_event_destroy(LuxEvent* event) {
    delete event;
}

LuxError lux_event_record(LuxEvent* event, LuxStream* stream) {
    if (!event) return LUX_ERROR_INVALID_ARGUMENT;
    // Stream can be null (use default stream)
    // For CPU: record current time
    // For GPU: would insert marker into command queue
    event->timestamp = std::chrono::steady_clock::now();
    event->recorded = true;
    return LUX_OK;
}

LuxError lux_event_wait(LuxEvent* event, LuxStream* stream) {
    if (!event || !event->recorded) return LUX_ERROR_INVALID_ARGUMENT;
    // Stream can be null (use default stream)
    // For CPU: event is already complete (synchronous execution)
    // For GPU: would wait until event is signaled
    return LUX_OK;
}

float lux_event_elapsed(LuxEvent* start, LuxEvent* end) {
    if (!start || !end || !start->recorded || !end->recorded) return 0.0f;

    auto duration = end->timestamp - start->timestamp;
    // Return elapsed time in milliseconds
    return std::chrono::duration<float, std::milli>(duration).count();
}

// NTT Operations
LuxError lux_ntt_forward(LuxGPU* gpu, uint64_t* data, size_t n, uint64_t modulus) {
    if (!gpu || !gpu->vtbl || !data) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_ntt_forward) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_ntt_forward(gpu->ctx, data, n, modulus));
}

LuxError lux_ntt_inverse(LuxGPU* gpu, uint64_t* data, size_t n, uint64_t modulus) {
    if (!gpu || !gpu->vtbl || !data) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_ntt_inverse) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_ntt_inverse(gpu->ctx, data, n, modulus));
}

LuxError lux_ntt_batch(LuxGPU* gpu, uint64_t** polys, size_t count, size_t n, uint64_t modulus) {
    if (!gpu || !gpu->vtbl || !polys) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_ntt_forward) return LUX_ERROR_NOT_SUPPORTED;

    // Process each polynomial in sequence
    for (size_t i = 0; i < count; i++) {
        LuxBackendError err = gpu->vtbl->op_ntt_forward(gpu->ctx, polys[i], n, modulus);
        if (err != LUX_BACKEND_OK) return static_cast<LuxError>(err);
    }
    return LUX_OK;
}

// =============================================================================
// Polynomial Arithmetic
// =============================================================================

LuxError lux_poly_mul(LuxGPU* gpu, const uint64_t* a, const uint64_t* b, uint64_t* result, size_t n, uint64_t modulus) {
    if (!gpu || !gpu->vtbl || !a || !b || !result) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_poly_mul) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_poly_mul(gpu->ctx, a, b, result, n, modulus));
}

// =============================================================================
// TFHE Operations
// =============================================================================

LuxError lux_tfhe_bootstrap(LuxGPU* gpu,
                            const uint64_t* lwe_in, uint64_t* lwe_out,
                            const uint64_t* bsk, const uint64_t* test_poly,
                            uint32_t n_lwe, uint32_t N, uint32_t k, uint32_t l, uint64_t q) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!lwe_in || !lwe_out || !bsk || !test_poly) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_tfhe_bootstrap) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_tfhe_bootstrap(gpu->ctx, lwe_in, lwe_out, bsk, test_poly, n_lwe, N, k, l, q));
}

LuxError lux_tfhe_keyswitch(LuxGPU* gpu,
                            const uint64_t* lwe_in, uint64_t* lwe_out,
                            const uint64_t* ksk,
                            uint32_t n_in, uint32_t n_out, uint32_t l, uint32_t base_log, uint64_t q) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!lwe_in || !lwe_out || !ksk) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_tfhe_keyswitch) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_tfhe_keyswitch(gpu->ctx, lwe_in, lwe_out, ksk, n_in, n_out, l, base_log, q));
}

LuxError lux_blind_rotate(LuxGPU* gpu,
                          uint64_t* acc, const uint64_t* bsk, const uint64_t* lwe_a,
                          uint32_t n_lwe, uint32_t N, uint32_t k, uint32_t l, uint64_t q) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!acc || !bsk || !lwe_a) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_blind_rotate) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_blind_rotate(gpu->ctx, acc, bsk, lwe_a, n_lwe, N, k, l, q));
}

// =============================================================================
// Crypto: Hash Functions
// =============================================================================

LuxError lux_poseidon2_hash(LuxGPU* gpu, const uint64_t* inputs, uint64_t* outputs, size_t rate, size_t num_hashes) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!inputs || !outputs) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_poseidon2_hash) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_poseidon2_hash(gpu->ctx, inputs, outputs, rate, num_hashes));
}

LuxError lux_blake3_hash(LuxGPU* gpu, const uint8_t* inputs, uint8_t* outputs, const size_t* input_lens, size_t num_hashes) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!inputs || !outputs || !input_lens) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_blake3_hash) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_blake3_hash(gpu->ctx, inputs, outputs, input_lens, num_hashes));
}

// =============================================================================
// Crypto: MSM
// =============================================================================

LuxError lux_msm(LuxGPU* gpu, const void* scalars, const void* points, void* result, size_t count, LuxCurve curve) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!scalars || !points || !result) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_msm) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_msm(gpu->ctx, scalars, points, result, count, static_cast<int>(curve)));
}

// =============================================================================
// Crypto: BLS12-381 Curve
// =============================================================================

LuxError lux_bls12_381_add(LuxGPU* gpu, const void* a, const void* b, void* out, size_t count, bool is_g2) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!a || !b || !out) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_bls12_381_add) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_bls12_381_add(gpu->ctx, a, b, out, count, is_g2));
}

LuxError lux_bls12_381_mul(LuxGPU* gpu, const void* points, const void* scalars, void* out, size_t count, bool is_g2) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!points || !scalars || !out) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_bls12_381_mul) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_bls12_381_mul(gpu->ctx, points, scalars, out, count, is_g2));
}

LuxError lux_bls12_381_pairing(LuxGPU* gpu, const void* g1_points, const void* g2_points, void* out, size_t count) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!g1_points || !g2_points || !out) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_bls12_381_pairing) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_bls12_381_pairing(gpu->ctx, g1_points, g2_points, out, count));
}

// =============================================================================
// Crypto: BN254 Curve
// =============================================================================

LuxError lux_bn254_add(LuxGPU* gpu, const void* a, const void* b, void* out, size_t count, bool is_g2) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!a || !b || !out) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_bn254_add) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_bn254_add(gpu->ctx, a, b, out, count, is_g2));
}

LuxError lux_bn254_mul(LuxGPU* gpu, const void* points, const void* scalars, void* out, size_t count, bool is_g2) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!points || !scalars || !out) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_bn254_mul) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_bn254_mul(gpu->ctx, points, scalars, out, count, is_g2));
}

// =============================================================================
// Crypto: KZG Polynomial Commitments
// =============================================================================

LuxError lux_kzg_commit(LuxGPU* gpu, const void* coeffs, const void* srs, void* commitment, size_t degree, LuxCurve curve) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!coeffs || !srs || !commitment) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_kzg_commit) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_kzg_commit(gpu->ctx, coeffs, srs, commitment, degree, static_cast<int>(curve)));
}

LuxError lux_kzg_open(LuxGPU* gpu, const void* coeffs, const void* srs, const void* point, void* proof, size_t degree, LuxCurve curve) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!coeffs || !srs || !point || !proof) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_kzg_open) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_kzg_open(gpu->ctx, coeffs, srs, point, proof, degree, static_cast<int>(curve)));
}

LuxError lux_kzg_verify(LuxGPU* gpu, const void* commitment, const void* proof, const void* point, const void* value, const void* srs_g2, bool* result, LuxCurve curve) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!commitment || !proof || !point || !value || !srs_g2 || !result) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_kzg_verify) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_kzg_verify(gpu->ctx, commitment, proof, point, value, srs_g2, result, static_cast<int>(curve)));
}

// =============================================================================
// BLS Signature Operations
// =============================================================================
//
// BLS signatures use BLS12-381 curve with:
// - G1: 48-byte compressed points (public keys)
// - G2: 96-byte compressed points (signatures)
// - Verification: e(pubkey, H(msg)) == e(G1_generator, sig)
//
// These functions require the backend to support BLS12-381 pairing operations.
// Without backend support, they return LUX_ERROR_NOT_SUPPORTED.

// BLS12-381 constants
static constexpr size_t BLS_G1_COMPRESSED_SIZE = 48;
static constexpr size_t BLS_G2_COMPRESSED_SIZE = 96;
static constexpr size_t BLS_G1_UNCOMPRESSED_SIZE = 96;
static constexpr size_t BLS_G2_UNCOMPRESSED_SIZE = 192;

LuxError lux_bls_verify(LuxGPU* gpu,
                        const uint8_t* sig, size_t sig_len,
                        const uint8_t* msg, size_t msg_len,
                        const uint8_t* pubkey, size_t pubkey_len,
                        bool* result) {
    // Validate inputs
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!sig || !msg || !pubkey || !result) return LUX_ERROR_INVALID_ARGUMENT;

    // Check backend supports required operations
    if (!gpu->vtbl->op_bls12_381_pairing) {
        gpu->set_error("BLS verify requires BLS12-381 pairing support");
        return LUX_ERROR_NOT_SUPPORTED;
    }

    // Validate point sizes
    // Accept both compressed (48/96) and uncompressed (96/192) formats
    bool pubkey_compressed = (pubkey_len == BLS_G1_COMPRESSED_SIZE);
    bool sig_compressed = (sig_len == BLS_G2_COMPRESSED_SIZE);

    if (!pubkey_compressed && pubkey_len != BLS_G1_UNCOMPRESSED_SIZE) {
        gpu->set_error("Invalid public key size: expected 48 or 96 bytes");
        return LUX_ERROR_INVALID_ARGUMENT;
    }
    if (!sig_compressed && sig_len != BLS_G2_UNCOMPRESSED_SIZE) {
        gpu->set_error("Invalid signature size: expected 96 or 192 bytes");
        return LUX_ERROR_INVALID_ARGUMENT;
    }

    // BLS verification requires hash-to-curve (H(msg) -> G2) and pairing check
    // Full implementation requires:
    // 1. Decompress pubkey to G1 point
    // 2. Hash message to G2 point (hash_to_curve per RFC 9380)
    // 3. Decompress sig to G2 point
    // 4. Verify: e(pubkey, H(msg)) == e(G1_generator, sig)
    //
    // The backend's pairing operation handles the pairing math.
    // Hash-to-curve requires field arithmetic not yet in the vtable.
    //
    // For now, we return NOT_SUPPORTED with clear error message.
    // A full implementation would integrate with a crypto library like blst.

    gpu->set_error("BLS verify requires hash-to-curve; integrate blst or similar");
    return LUX_ERROR_NOT_SUPPORTED;
}

LuxError lux_bls_verify_batch(LuxGPU* gpu,
                              const uint8_t* const* sigs, const size_t* sig_lens,
                              const uint8_t* const* msgs, const size_t* msg_lens,
                              const uint8_t* const* pubkeys, const size_t* pubkey_lens,
                              int count, bool* results) {
    // Validate inputs
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!sigs || !sig_lens || !msgs || !msg_lens) return LUX_ERROR_INVALID_ARGUMENT;
    if (!pubkeys || !pubkey_lens || !results) return LUX_ERROR_INVALID_ARGUMENT;
    if (count <= 0) return LUX_ERROR_INVALID_ARGUMENT;

    // Check backend supports required operations
    if (!gpu->vtbl->op_bls12_381_pairing) {
        gpu->set_error("BLS batch verify requires BLS12-381 pairing support");
        return LUX_ERROR_NOT_SUPPORTED;
    }

    // Batch verification uses randomized linear combination for efficiency:
    // Instead of n independent pairing checks, verify:
    // e(sum(r_i * pubkey_i), H(msg_i)) == e(G1, sum(r_i * sig_i))
    // where r_i are random scalars.
    //
    // This requires the same primitives as single verify plus:
    // - Scalar multiplication on G1 and G2
    // - Point addition on G1 and G2
    //
    // For now, fall back to sequential verification.
    for (int i = 0; i < count; i++) {
        LuxError err = lux_bls_verify(gpu, sigs[i], sig_lens[i],
                                       msgs[i], msg_lens[i],
                                       pubkeys[i], pubkey_lens[i],
                                       &results[i]);
        if (err != LUX_OK && err != LUX_ERROR_NOT_SUPPORTED) {
            return err;
        }
        // If single verify is not supported, all results are indeterminate
        if (err == LUX_ERROR_NOT_SUPPORTED) {
            return err;
        }
    }
    return LUX_OK;
}

LuxError lux_bls_aggregate(LuxGPU* gpu,
                           const uint8_t* const* sigs, const size_t* sig_lens,
                           int count, uint8_t* out, size_t* out_len) {
    // Validate inputs
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!sigs || !sig_lens || !out || !out_len) return LUX_ERROR_INVALID_ARGUMENT;
    if (count <= 0) return LUX_ERROR_INVALID_ARGUMENT;

    // Check backend supports required operations
    if (!gpu->vtbl->op_bls12_381_add) {
        gpu->set_error("BLS aggregate requires BLS12-381 point addition");
        return LUX_ERROR_NOT_SUPPORTED;
    }

    // Validate all signatures have consistent size
    size_t first_len = sig_lens[0];
    bool compressed = (first_len == BLS_G2_COMPRESSED_SIZE);
    if (!compressed && first_len != BLS_G2_UNCOMPRESSED_SIZE) {
        gpu->set_error("Invalid signature size");
        return LUX_ERROR_INVALID_ARGUMENT;
    }

    for (int i = 1; i < count; i++) {
        if (sig_lens[i] != first_len) {
            gpu->set_error("All signatures must have same size for aggregation");
            return LUX_ERROR_INVALID_ARGUMENT;
        }
    }

    // Aggregation: agg_sig = sig_1 + sig_2 + ... + sig_n (G2 point addition)
    // For compressed points, we need to decompress, add, recompress.
    // The backend add operation works on uncompressed points.

    if (compressed) {
        // Compressed format requires decompression - not yet implemented
        gpu->set_error("Compressed signature aggregation requires point decompression");
        return LUX_ERROR_NOT_SUPPORTED;
    }

    // Uncompressed format: direct G2 addition
    // Allocate working buffer for accumulator
    std::vector<uint8_t> acc(BLS_G2_UNCOMPRESSED_SIZE);
    std::memcpy(acc.data(), sigs[0], BLS_G2_UNCOMPRESSED_SIZE);

    for (int i = 1; i < count; i++) {
        std::vector<uint8_t> result(BLS_G2_UNCOMPRESSED_SIZE);
        LuxBackendError err = gpu->vtbl->op_bls12_381_add(
            gpu->ctx,
            acc.data(),      // a
            sigs[i],         // b
            result.data(),   // out
            1,               // count
            true             // is_g2
        );
        if (err != LUX_BACKEND_OK) {
            gpu->set_error("G2 point addition failed during aggregation");
            return static_cast<LuxError>(err);
        }
        acc = std::move(result);
    }

    // Copy result
    if (*out_len < BLS_G2_UNCOMPRESSED_SIZE) {
        gpu->set_error("Output buffer too small");
        return LUX_ERROR_INVALID_ARGUMENT;
    }
    std::memcpy(out, acc.data(), BLS_G2_UNCOMPRESSED_SIZE);
    *out_len = BLS_G2_UNCOMPRESSED_SIZE;

    return LUX_OK;
}

} // extern "C"
