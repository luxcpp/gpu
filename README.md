# Lux GPU Core

Lightweight plugin-based GPU acceleration library for blockchain and ML workloads.

## Architecture

This is the **core library only**. It provides:
- **Stable ABI** (`backend_plugin.h`) - Plugin contract
- **Plugin Loader** - Dynamic loading of backend plugins
- **CPU Fallback** - Builtin CPU backend for any platform
- **Tests** - Backend-agnostic test harness

Backend plugins are built and distributed separately:

| Plugin | Repo | Platform | Dependencies |
|--------|------|----------|--------------|
| Metal | `luxcpp/metal` | macOS arm64 | MLX, Metal.framework |
| CUDA | `luxcpp/cuda` | Linux, Windows | CUDA Toolkit, CCCL |
| WebGPU | `luxcpp/webgpu` | All | Dawn/wgpu, gpu.cpp |

## Building

```bash
# Core only (CPU backend)
cmake -B build
cmake --build build

# Run tests
ctest --test-dir build
```

## Usage

```c
#include <lux/gpu.h>

int main() {
    // Initialize (loads best available backend)
    lux_gpu_init();

    // Or specify backend explicitly
    // lux_gpu_set_backend(LUX_BACKEND_CUDA);

    LuxContext* ctx = lux_gpu_create_context(-1);

    // Allocate and compute...
    LuxBuffer* buf = lux_gpu_alloc(ctx, 1024 * sizeof(float));

    lux_gpu_free(ctx, buf);
    lux_gpu_destroy_context(ctx);
    lux_gpu_shutdown();
}
```

## Backend Selection

At runtime, backends are selected in priority order:
1. **CUDA** - If NVIDIA GPU detected
2. **Metal** - If macOS arm64
3. **WebGPU** - Cross-platform fallback
4. **CPU** - Final fallback (always available)

Override via environment or API:
```bash
export LUX_BACKEND=cuda  # or metal, webgpu, cpu
```

## Plugin Loading

Backends are loaded from:
1. `LUX_GPU_BACKEND_PATH` environment variable
2. System library paths (`/usr/lib/lux-gpu`, etc.)
3. Relative to executable

Plugin naming: `libluxgpu_backend_<name>.{so,dylib,dll}`

## ABI Stability

The plugin ABI is versioned. Plugins must match the core ABI version:
```c
// backend_plugin.h
#define LUX_GPU_ABI_VERSION 1
```

## License

BSD-3-Clause-Eco

