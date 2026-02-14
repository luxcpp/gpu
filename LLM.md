# Lux GPU - C++ Backend Library

Plugin-based GPU acceleration for blockchain and ML workloads.

## Structure

```
luxcpp/gpu/
├── include/lux/gpu.h      # C API (stable)
├── src/
│   ├── gpu_core.cpp       # Core dispatch logic
│   ├── cpu_backend.cpp    # CPU SIMD backend
│   ├── bn254_field.hpp    # BN254 field arithmetic
│   └── zk_ops.cpp         # ZK primitives
├── webgpu/                # Dawn WebGPU backend
│   ├── gpu.hpp            # gpu.cpp header (Dawn wrapper)
│   └── kernels/           # WGSL kernel sources
├── kernels/cpu/           # CPU kernel implementations
├── benchmarks/            # Performance tests
└── test/                  # Unit tests
```

## Backends

| Backend | Location | Notes |
|---------|----------|-------|
| CPU | `src/cpu_backend.cpp` | Always available, SIMD optimized |
| Metal | `luxcpp/metal` (separate) | Apple Silicon, uses MLX |
| CUDA | `luxcpp/cuda` (separate) | NVIDIA, uses CCCL |
| WebGPU | `webgpu/` | Dawn-based, WGSL kernels |

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

## Key C API Functions

### Context
```c
LuxGPU* lux_gpu_create(void);
void lux_gpu_destroy(LuxGPU* gpu);
LuxError lux_gpu_sync(LuxGPU* gpu);
LuxError lux_gpu_set_backend(LuxGPU* gpu, LuxBackend backend);
```

### Tensors
```c
LuxTensor* lux_tensor_zeros(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype);
LuxTensor* lux_tensor_add(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
LuxTensor* lux_tensor_matmul(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
void lux_tensor_destroy(LuxTensor* tensor);
```

### Crypto Operations
```c
// BLS12-381
LuxError lux_bls12_381_add(LuxGPU* gpu, ...);
LuxError lux_bls12_381_mul(LuxGPU* gpu, ...);
LuxError lux_bls12_381_pairing(LuxGPU* gpu, ...);
LuxError lux_bls_verify(LuxGPU* gpu, ...);

// BN254
LuxError lux_bn254_add(LuxGPU* gpu, ...);
LuxError lux_bn254_mul(LuxGPU* gpu, ...);

// KZG Commitments
LuxError lux_kzg_commit(LuxGPU* gpu, ...);
LuxError lux_kzg_open(LuxGPU* gpu, ...);
LuxError lux_kzg_verify(LuxGPU* gpu, ...);

// Poseidon2 Hash
LuxError lux_poseidon2_hash(LuxGPU* gpu, ...);
LuxError lux_gpu_poseidon2(LuxGPU* gpu, ...);
LuxError lux_gpu_merkle_root(LuxGPU* gpu, ...);
```

### FHE Operations
```c
LuxError lux_ntt_forward(LuxGPU* gpu, uint64_t* data, size_t n, uint64_t modulus);
LuxError lux_ntt_inverse(LuxGPU* gpu, uint64_t* data, size_t n, uint64_t modulus);
LuxError lux_tfhe_bootstrap(LuxGPU* gpu, ...);
LuxError lux_tfhe_keyswitch(LuxGPU* gpu, ...);
LuxError lux_blind_rotate(LuxGPU* gpu, ...);
```

## Environment Variables

- `LUX_GPU_BACKEND` - Force backend: `metal`, `cuda`, `webgpu`, `cpu`
- `LUX_GPU_BACKEND_PATH` - Custom plugin search path

## Go Bindings

See `lux/gpu` for Go bindings that wrap this C API via CGO.
