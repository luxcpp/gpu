# Lux GPU Core {#mainpage}

Lightweight plugin-based GPU acceleration library for blockchain and ML workloads.

## Overview

Lux GPU Core provides a unified C API for GPU acceleration with runtime-switchable backends. The library is designed for high-performance cryptographic operations (ZK proofs, FHE, BLS signatures) and tensor computations.

### Key Features

- **Plugin Architecture**: Backends are dynamically loaded at runtime
- **Stable ABI**: Plugin interface versioned for compatibility
- **Multiple Backends**: Metal, CUDA, WebGPU, and CPU fallback
- **Cryptographic Operations**: BLS12-381, BN254, Poseidon2, KZG commitments
- **FHE Support**: NTT, TFHE bootstrap, key switching
- **Tensor Operations**: GEMM, reductions, activations, normalization

### Supported Backends

| Backend | Platform | GPU Type | Status |
|---------|----------|----------|--------|
| Metal   | macOS arm64 | Apple Silicon | Production |
| CUDA    | Linux/Windows | NVIDIA | Production |
| WebGPU  | Cross-platform | Any | Experimental |
| CPU     | All | SIMD | Fallback |

## Quick Start

```c
#include <lux/gpu.h>

int main() {
    // Create GPU context (auto-selects best backend)
    LuxGPU* gpu = lux_gpu_create();

    // Create tensors
    int64_t shape[] = {1024, 1024};
    LuxTensor* a = lux_tensor_ones(gpu, shape, 2, LUX_FLOAT32);
    LuxTensor* b = lux_tensor_ones(gpu, shape, 2, LUX_FLOAT32);

    // Matrix multiplication
    LuxTensor* c = lux_tensor_matmul(gpu, a, b);

    // Synchronize and read results
    lux_gpu_sync(gpu);

    // Cleanup
    lux_tensor_destroy(c);
    lux_tensor_destroy(b);
    lux_tensor_destroy(a);
    lux_gpu_destroy(gpu);
    return 0;
}
```

## Architecture

```
+------------------+
|   Application    |
+------------------+
         |
    lux/gpu.h (C API)
         |
+------------------+
|   Core Library   |  Plugin Loader + Dispatch
+------------------+
         |
   backend_plugin.h (ABI)
         |
+--------+--------+--------+--------+
| Metal  | CUDA   | WebGPU | CPU    |
+--------+--------+--------+--------+
```

The core library provides:

- **Unified API** (`lux/gpu.h`): Application-facing C interface
- **Plugin ABI** (`lux/gpu/backend_plugin.h`): Stable interface for backends
- **Kernel Loader** (`lux/gpu/kernel_loader.h`): Generic kernel management

Backends are separate shared libraries loaded at runtime.

## Documentation

- @ref api.md "API Reference" - Complete function reference
- @ref examples.md "Examples" - Code samples and usage patterns
- @ref install.md "Installation" - Build and installation guide

## License

BSD-3-Clause-Eco
