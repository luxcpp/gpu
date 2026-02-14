// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Internal header for LuxGPU struct access
// Used by zk_ops.cpp and other internal files that need backend vtable access

#ifndef LUX_GPU_INTERNAL_H
#define LUX_GPU_INTERNAL_H

#include "lux/gpu.h"
#include "lux/gpu/backend_plugin.h"
#include <string>
#include <mutex>

// LuxGPU struct must match gpu_core.cpp definition exactly
struct LuxGPU {
    std::string backend_name;
    const lux_gpu_backend_vtbl* vtbl = nullptr;
    LuxBackendContext* ctx = nullptr;
    std::string last_error;
    std::mutex mutex;

    ~LuxGPU();

    void set_error(const char* msg);
};

#endif // LUX_GPU_INTERNAL_H
