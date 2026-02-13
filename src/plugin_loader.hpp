// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Plugin Loader - Runtime loading of GPU backend shared libraries

#ifndef LUX_GPU_PLUGIN_LOADER_HPP
#define LUX_GPU_PLUGIN_LOADER_HPP

#include "lux/gpu/backend_plugin.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace lux::gpu {

// =============================================================================
// Platform-specific dynamic library handling
// =============================================================================

#ifdef _WIN32
    #include <windows.h>
    typedef HMODULE DylibHandle;
    #define DYLIB_EXT ".dll"
#else
    #include <dlfcn.h>
    typedef void* DylibHandle;
    #ifdef __APPLE__
        #define DYLIB_EXT ".dylib"
    #else
        #define DYLIB_EXT ".so"
    #endif
#endif

// =============================================================================
// Loaded Backend
// =============================================================================

struct LoadedBackend {
    std::string name;
    std::string path;
    DylibHandle handle = nullptr;
    lux_gpu_backend_desc desc = {};
    bool available = false;

    ~LoadedBackend() {
        if (handle) {
#ifdef _WIN32
            FreeLibrary(handle);
#else
            dlclose(handle);
#endif
        }
    }
};

// =============================================================================
// Plugin Loader
// =============================================================================

class PluginLoader {
public:
    static PluginLoader& instance() {
        static PluginLoader loader;
        return loader;
    }

    // Scan directory for backend plugins
    void scan_directory(const std::string& path);

    // Load a specific backend by name
    bool load_backend(const std::string& name);

    // Load a backend from a specific path
    bool load_backend_from_path(const std::string& path);

    // Get all available backends
    std::vector<std::string> available_backends() const;

    // Get a loaded backend by name
    const LoadedBackend* get_backend(const std::string& name) const;

    // Get the best available backend (priority: metal/cuda > webgpu > cpu)
    const LoadedBackend* get_best_backend() const;

    // Check if a backend is available
    bool is_available(const std::string& name) const;

    // Add search paths
    void add_search_path(const std::string& path);

    // Set from environment (LUX_GPU_BACKEND_PATH)
    void init_from_environment();

private:
    PluginLoader() {
        init_from_environment();
    }

    mutable std::mutex mutex_;  // Protects search_paths_ and backends_
    std::vector<std::string> search_paths_;
    std::unordered_map<std::string, std::unique_ptr<LoadedBackend>> backends_;

    DylibHandle open_library(const std::string& path);
    void* get_symbol(DylibHandle handle, const char* name);
    std::string get_last_error();
};

// =============================================================================
// Implementation
// =============================================================================

inline void PluginLoader::init_from_environment() {
    // Default search paths
#ifdef __APPLE__
    search_paths_.push_back("/usr/local/lib/lux-gpu");
    search_paths_.push_back("/opt/homebrew/lib/lux-gpu");
#elif defined(_WIN32)
    search_paths_.push_back("C:\\Program Files\\lux-gpu\\bin");
#else
    search_paths_.push_back("/usr/lib/lux-gpu");
    search_paths_.push_back("/usr/local/lib/lux-gpu");
#endif

    // Environment override
    const char* env_path = std::getenv("LUX_GPU_BACKEND_PATH");
    if (env_path && *env_path) {
        // Split by : (Unix) or ; (Windows)
#ifdef _WIN32
        char delim = ';';
#else
        char delim = ':';
#endif
        std::string paths(env_path);
        size_t pos = 0;
        while ((pos = paths.find(delim)) != std::string::npos) {
            search_paths_.insert(search_paths_.begin(), paths.substr(0, pos));
            paths.erase(0, pos + 1);
        }
        if (!paths.empty()) {
            search_paths_.insert(search_paths_.begin(), paths);
        }
    }

    // Also check directory of the core library itself
    // (plugins often installed alongside core)
}

inline void PluginLoader::add_search_path(const std::string& path) {
    search_paths_.insert(search_paths_.begin(), path);
}

inline void PluginLoader::scan_directory(const std::string& dir) {
    // Look for libluxgpu_backend_*.dylib / .so / .dll
    std::vector<std::string> candidates = {
        "metal", "cuda", "webgpu"
    };

    for (const auto& name : candidates) {
        std::string filename = "libluxgpu_backend_" + name + DYLIB_EXT;
        std::string full_path = dir + "/" + filename;
        load_backend_from_path(full_path);
    }
}

inline bool PluginLoader::load_backend(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Already loaded?
    if (backends_.count(name)) {
        return backends_[name]->available;
    }

    std::string filename = "libluxgpu_backend_" + name + DYLIB_EXT;

    // Copy search paths under lock, then search
    std::vector<std::string> paths = search_paths_;

    // Search in all paths (mutex released during dlopen for better concurrency)
    for (const auto& dir : paths) {
        std::string full_path = dir + "/" + filename;
        // Note: load_backend_from_path_unlocked would be better but requires refactor
        // For now, we hold the lock through loading
        DylibHandle handle = open_library(full_path);
        if (!handle) continue;

        auto init_fn = reinterpret_cast<lux_gpu_backend_init_fn>(
            get_symbol(handle, LUX_GPU_BACKEND_INIT_SYMBOL)
        );

        if (!init_fn) {
#ifdef _WIN32
            FreeLibrary(handle);
#else
            dlclose(handle);
#endif
            continue;
        }

        auto backend = std::make_unique<LoadedBackend>();
        backend->path = full_path;
        backend->handle = handle;

        if (!init_fn(&backend->desc)) {
            backend->available = false;
        } else {
            if (backend->desc.abi_version != LUX_GPU_BACKEND_ABI_VERSION) {
                fprintf(stderr, "lux-gpu: backend %s has ABI version %u, expected %u\n",
                        full_path.c_str(), backend->desc.abi_version, LUX_GPU_BACKEND_ABI_VERSION);
#ifdef _WIN32
                FreeLibrary(handle);
#else
                dlclose(handle);
#endif
                continue;
            }
            backend->available = true;
        }

        backend->name = backend->desc.backend_name ? backend->desc.backend_name : "unknown";
        bool available = backend->available;
        backends_[backend->name] = std::move(backend);
        return available;
    }

    return false;
}

inline bool PluginLoader::load_backend_from_path(const std::string& path) {
    DylibHandle handle = open_library(path);
    if (!handle) {
        return false;
    }

    // Get init function
    auto init_fn = reinterpret_cast<lux_gpu_backend_init_fn>(
        get_symbol(handle, LUX_GPU_BACKEND_INIT_SYMBOL)
    );

    if (!init_fn) {
#ifdef _WIN32
        FreeLibrary(handle);
#else
        dlclose(handle);
#endif
        return false;
    }

    // Call init
    auto backend = std::make_unique<LoadedBackend>();
    backend->path = path;
    backend->handle = handle;

    if (!init_fn(&backend->desc)) {
        // Backend reports unavailable (e.g., no GPU)
        backend->available = false;
    } else {
        // Verify ABI version
        if (backend->desc.abi_version != LUX_GPU_BACKEND_ABI_VERSION) {
            fprintf(stderr, "lux-gpu: backend %s has ABI version %u, expected %u\n",
                    path.c_str(), backend->desc.abi_version, LUX_GPU_BACKEND_ABI_VERSION);
#ifdef _WIN32
            FreeLibrary(handle);
#else
            dlclose(handle);
#endif
            return false;
        }
        backend->available = true;
    }

    backend->name = backend->desc.backend_name ? backend->desc.backend_name : "unknown";
    std::string name = backend->name;  // Save before move
    bool available = backend->available;
    backends_[name] = std::move(backend);
    return available;
}

inline std::vector<std::string> PluginLoader::available_backends() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;
    result.push_back("cpu");  // Always available
    for (const auto& [name, backend] : backends_) {
        if (backend->available) {
            result.push_back(name);
        }
    }
    return result;
}

inline const LoadedBackend* PluginLoader::get_backend(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = backends_.find(name);
    return (it != backends_.end() && it->second->available) ? it->second.get() : nullptr;
}

inline const LoadedBackend* PluginLoader::get_best_backend() const {
    std::lock_guard<std::mutex> lock(mutex_);
    // Priority order
    static const char* priority[] = {"metal", "cuda", "webgpu"};

    for (const char* name : priority) {
        auto it = backends_.find(name);
        if (it != backends_.end() && it->second->available) {
            return it->second.get();
        }
    }

    return nullptr;  // Fall back to CPU (built-in)
}

inline bool PluginLoader::is_available(const std::string& name) const {
    if (name == "cpu") return true;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = backends_.find(name);
    return it != backends_.end() && it->second->available;
}

inline DylibHandle PluginLoader::open_library(const std::string& path) {
#ifdef _WIN32
    return LoadLibraryA(path.c_str());
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

inline void* PluginLoader::get_symbol(DylibHandle handle, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(handle, name));
#else
    return dlsym(handle, name);
#endif
}

inline std::string PluginLoader::get_last_error() {
#ifdef _WIN32
    DWORD err = GetLastError();
    char buf[256];
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, nullptr, err, 0, buf, sizeof(buf), nullptr);
    return buf;
#else
    const char* err = dlerror();
    return err ? err : "unknown error";
#endif
}

} // namespace lux::gpu

#endif // LUX_GPU_PLUGIN_LOADER_HPP
