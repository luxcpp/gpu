# API Reference {#api}

Complete reference for the Lux GPU Core C API.

## Header Files

| Header | Description |
|--------|-------------|
| `<lux/gpu.h>` | Main API - GPU context, tensors, operations |
| `<lux/gpu/backend_plugin.h>` | Backend plugin interface (for implementers) |
| `<lux/gpu/kernel_loader.h>` | Kernel loading and caching |

---

## GPU Context

### lux_gpu_create

```c
LuxGPU* lux_gpu_create(void);
```

Create a GPU context with automatic backend selection.

Backend priority: CUDA > Metal > WebGPU > CPU

**Returns**: GPU context handle, or NULL on failure.

### lux_gpu_create_with_backend

```c
LuxGPU* lux_gpu_create_with_backend(LuxBackend backend);
```

Create a GPU context with a specific backend.

**Parameters**:
- `backend`: Backend type (LUX_BACKEND_CUDA, LUX_BACKEND_METAL, etc.)

**Returns**: GPU context handle, or NULL if backend unavailable.

### lux_gpu_create_with_device

```c
LuxGPU* lux_gpu_create_with_device(LuxBackend backend, int device_index);
```

Create a GPU context with a specific backend and device.

**Parameters**:
- `backend`: Backend type
- `device_index`: Device index (0 for first device)

**Returns**: GPU context handle, or NULL on failure.

### lux_gpu_destroy

```c
void lux_gpu_destroy(LuxGPU* gpu);
```

Destroy a GPU context and release all resources.

### lux_gpu_sync

```c
LuxError lux_gpu_sync(LuxGPU* gpu);
```

Synchronize all pending GPU operations.

**Returns**: LUX_OK on success, error code otherwise.

### lux_gpu_backend

```c
LuxBackend lux_gpu_backend(LuxGPU* gpu);
```

Get the current backend type.

### lux_gpu_backend_name

```c
const char* lux_gpu_backend_name(LuxGPU* gpu);
```

Get the current backend name as a string.

### lux_gpu_set_backend

```c
LuxError lux_gpu_set_backend(LuxGPU* gpu, LuxBackend backend);
```

Switch to a different backend at runtime.

**Note**: This invalidates all existing tensors and buffers.

### lux_gpu_device_info

```c
LuxError lux_gpu_device_info(LuxGPU* gpu, LuxDeviceInfo* info);
```

Get information about the current device.

### lux_gpu_error

```c
const char* lux_gpu_error(LuxGPU* gpu);
```

Get the last error message.

---

## Backend Query

### lux_backend_count

```c
int lux_backend_count(void);
```

Get the number of available backends.

### lux_backend_available

```c
bool lux_backend_available(LuxBackend backend);
```

Check if a backend is available on this system.

### lux_backend_name

```c
const char* lux_backend_name(LuxBackend backend);
```

Get the name of a backend as a string.

### lux_device_count

```c
int lux_device_count(LuxBackend backend);
```

Get the number of devices for a backend.

### lux_device_info

```c
LuxError lux_device_info(LuxBackend backend, int index, LuxDeviceInfo* info);
```

Get device info for a specific backend and device index.

---

## Tensor Creation

### lux_tensor_zeros

```c
LuxTensor* lux_tensor_zeros(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype);
```

Create a tensor filled with zeros.

**Parameters**:
- `gpu`: GPU context
- `shape`: Array of dimension sizes
- `ndim`: Number of dimensions
- `dtype`: Data type (LUX_FLOAT32, LUX_FLOAT16, etc.)

**Returns**: Tensor handle, or NULL on failure.

### lux_tensor_ones

```c
LuxTensor* lux_tensor_ones(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype);
```

Create a tensor filled with ones.

### lux_tensor_full

```c
LuxTensor* lux_tensor_full(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype, double value);
```

Create a tensor filled with a specific value.

### lux_tensor_from_data

```c
LuxTensor* lux_tensor_from_data(LuxGPU* gpu, const void* data, const int64_t* shape, int ndim, LuxDtype dtype);
```

Create a tensor from host data.

**Parameters**:
- `data`: Pointer to host data (copied to device)
- `shape`: Array of dimension sizes
- `ndim`: Number of dimensions
- `dtype`: Data type

### lux_tensor_destroy

```c
void lux_tensor_destroy(LuxTensor* tensor);
```

Destroy a tensor and free device memory.

---

## Tensor Properties

### lux_tensor_ndim

```c
int lux_tensor_ndim(LuxTensor* tensor);
```

Get the number of dimensions.

### lux_tensor_shape

```c
int64_t lux_tensor_shape(LuxTensor* tensor, int dim);
```

Get the size of a specific dimension.

### lux_tensor_size

```c
int64_t lux_tensor_size(LuxTensor* tensor);
```

Get the total number of elements.

### lux_tensor_dtype

```c
LuxDtype lux_tensor_dtype(LuxTensor* tensor);
```

Get the data type.

### lux_tensor_to_host

```c
LuxError lux_tensor_to_host(LuxTensor* tensor, void* data, size_t size);
```

Copy tensor data to host memory.

**Parameters**:
- `tensor`: Source tensor
- `data`: Destination host buffer
- `size`: Size of destination buffer in bytes

---

## Arithmetic Operations

All arithmetic operations return a new tensor with the result.

### lux_tensor_add

```c
LuxTensor* lux_tensor_add(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
```

Element-wise addition: `result = a + b`

### lux_tensor_sub

```c
LuxTensor* lux_tensor_sub(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
```

Element-wise subtraction: `result = a - b`

### lux_tensor_mul

```c
LuxTensor* lux_tensor_mul(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
```

Element-wise multiplication: `result = a * b`

### lux_tensor_div

```c
LuxTensor* lux_tensor_div(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
```

Element-wise division: `result = a / b`

### lux_tensor_matmul

```c
LuxTensor* lux_tensor_matmul(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
```

Matrix multiplication: `result = a @ b`

**Parameters**:
- `a`: Left matrix [M, K]
- `b`: Right matrix [K, N]

**Returns**: Result matrix [M, N]

---

## Unary Operations

### lux_tensor_neg

```c
LuxTensor* lux_tensor_neg(LuxGPU* gpu, LuxTensor* t);
```

Element-wise negation: `result = -t`

### lux_tensor_exp

```c
LuxTensor* lux_tensor_exp(LuxGPU* gpu, LuxTensor* t);
```

Element-wise exponential: `result = exp(t)`

### lux_tensor_log

```c
LuxTensor* lux_tensor_log(LuxGPU* gpu, LuxTensor* t);
```

Element-wise natural logarithm: `result = log(t)`

### lux_tensor_sqrt

```c
LuxTensor* lux_tensor_sqrt(LuxGPU* gpu, LuxTensor* t);
```

Element-wise square root: `result = sqrt(t)`

### lux_tensor_abs

```c
LuxTensor* lux_tensor_abs(LuxGPU* gpu, LuxTensor* t);
```

Element-wise absolute value: `result = |t|`

### lux_tensor_tanh

```c
LuxTensor* lux_tensor_tanh(LuxGPU* gpu, LuxTensor* t);
```

Element-wise hyperbolic tangent.

### lux_tensor_sigmoid

```c
LuxTensor* lux_tensor_sigmoid(LuxGPU* gpu, LuxTensor* t);
```

Element-wise sigmoid: `result = 1 / (1 + exp(-t))`

### lux_tensor_relu

```c
LuxTensor* lux_tensor_relu(LuxGPU* gpu, LuxTensor* t);
```

Element-wise ReLU: `result = max(0, t)`

### lux_tensor_gelu

```c
LuxTensor* lux_tensor_gelu(LuxGPU* gpu, LuxTensor* t);
```

Element-wise GELU activation.

---

## Reduction Operations

### Full Reductions (to scalar)

```c
float lux_tensor_reduce_sum(LuxGPU* gpu, LuxTensor* t);
float lux_tensor_reduce_max(LuxGPU* gpu, LuxTensor* t);
float lux_tensor_reduce_min(LuxGPU* gpu, LuxTensor* t);
float lux_tensor_reduce_mean(LuxGPU* gpu, LuxTensor* t);
```

### Axis Reductions

```c
LuxTensor* lux_tensor_sum(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes);
LuxTensor* lux_tensor_mean(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes);
LuxTensor* lux_tensor_max(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes);
LuxTensor* lux_tensor_min(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes);
```

**Parameters**:
- `axes`: Array of axes to reduce
- `naxes`: Number of axes

---

## Normalization Operations

### lux_tensor_softmax

```c
LuxTensor* lux_tensor_softmax(LuxGPU* gpu, LuxTensor* t, int axis);
```

Softmax along specified axis.

### lux_tensor_log_softmax

```c
LuxTensor* lux_tensor_log_softmax(LuxGPU* gpu, LuxTensor* t, int axis);
```

Log-softmax along specified axis (numerically stable).

### lux_tensor_layer_norm

```c
LuxTensor* lux_tensor_layer_norm(LuxGPU* gpu, LuxTensor* t, LuxTensor* gamma, LuxTensor* beta, float eps);
```

Layer normalization with learnable parameters.

### lux_tensor_rms_norm

```c
LuxTensor* lux_tensor_rms_norm(LuxGPU* gpu, LuxTensor* t, LuxTensor* weight, float eps);
```

RMS normalization (used in LLaMA, etc.).

---

## Cryptographic Operations

### Hash Functions

#### lux_poseidon2_hash

```c
LuxError lux_poseidon2_hash(LuxGPU* gpu,
                            const uint64_t* inputs,
                            uint64_t* outputs,
                            size_t rate,
                            size_t num_hashes);
```

Poseidon2 algebraic hash for ZK circuits.

#### lux_blake3_hash

```c
LuxError lux_blake3_hash(LuxGPU* gpu,
                         const uint8_t* inputs,
                         uint8_t* outputs,
                         const size_t* input_lens,
                         size_t num_hashes);
```

BLAKE3 cryptographic hash.

### Multi-Scalar Multiplication

#### lux_msm

```c
LuxError lux_msm(LuxGPU* gpu,
                 const void* scalars,
                 const void* points,
                 void* result,
                 size_t count,
                 LuxCurve curve);
```

Multi-scalar multiplication for elliptic curves.

**Supported Curves**:
- `LUX_CURVE_BLS12_381`
- `LUX_CURVE_BN254`
- `LUX_CURVE_SECP256K1`
- `LUX_CURVE_ED25519`

### BLS12-381 Operations

```c
LuxError lux_bls12_381_add(LuxGPU* gpu, const void* a, const void* b, void* out, size_t count, bool is_g2);
LuxError lux_bls12_381_mul(LuxGPU* gpu, const void* points, const void* scalars, void* out, size_t count, bool is_g2);
LuxError lux_bls12_381_pairing(LuxGPU* gpu, const void* g1_points, const void* g2_points, void* out, size_t count);
```

### BLS Signature Operations

```c
LuxError lux_bls_verify(LuxGPU* gpu, const uint8_t* sig, size_t sig_len,
                        const uint8_t* msg, size_t msg_len,
                        const uint8_t* pubkey, size_t pubkey_len,
                        bool* result);

LuxError lux_bls_verify_batch(LuxGPU* gpu, ...);  // Batch verification
LuxError lux_bls_aggregate(LuxGPU* gpu, ...);     // Signature aggregation
```

### BN254 Operations

```c
LuxError lux_bn254_add(LuxGPU* gpu, const void* a, const void* b, void* out, size_t count, bool is_g2);
LuxError lux_bn254_mul(LuxGPU* gpu, const void* points, const void* scalars, void* out, size_t count, bool is_g2);
```

### KZG Polynomial Commitments

```c
LuxError lux_kzg_commit(LuxGPU* gpu, const void* coeffs, const void* srs, void* commitment, size_t degree, LuxCurve curve);
LuxError lux_kzg_open(LuxGPU* gpu, const void* coeffs, const void* srs, const void* point, void* proof, size_t degree, LuxCurve curve);
LuxError lux_kzg_verify(LuxGPU* gpu, const void* commitment, const void* proof, const void* point, const void* value, const void* srs_g2, bool* result, LuxCurve curve);
```

---

## FHE Operations

### NTT (Number Theoretic Transform)

```c
LuxError lux_ntt_forward(LuxGPU* gpu, uint64_t* data, size_t n, uint64_t modulus);
LuxError lux_ntt_inverse(LuxGPU* gpu, uint64_t* data, size_t n, uint64_t modulus);
LuxError lux_ntt_batch(LuxGPU* gpu, uint64_t** polys, size_t count, size_t n, uint64_t modulus);
```

### Polynomial Multiplication

```c
LuxError lux_poly_mul(LuxGPU* gpu, const uint64_t* a, const uint64_t* b, uint64_t* result, size_t n, uint64_t modulus);
```

Polynomial multiplication modulo (X^n + 1).

### TFHE Operations

```c
LuxError lux_tfhe_bootstrap(LuxGPU* gpu, const uint64_t* lwe_in, uint64_t* lwe_out,
                            const uint64_t* bsk, const uint64_t* test_poly,
                            uint32_t n_lwe, uint32_t N, uint32_t k, uint32_t l, uint64_t q);

LuxError lux_tfhe_keyswitch(LuxGPU* gpu, const uint64_t* lwe_in, uint64_t* lwe_out,
                            const uint64_t* ksk, uint32_t n_in, uint32_t n_out,
                            uint32_t l, uint32_t base_log, uint64_t q);

LuxError lux_blind_rotate(LuxGPU* gpu, uint64_t* acc, const uint64_t* bsk,
                          const uint64_t* lwe_a, uint32_t n_lwe, uint32_t N,
                          uint32_t k, uint32_t l, uint64_t q);
```

---

## ZK Primitives

### LuxFr256 Type

```c
typedef struct {
    uint64_t limbs[4];
} LuxFr256;
```

BN254 scalar field element (256-bit).

### lux_gpu_poseidon2

```c
LuxError lux_gpu_poseidon2(LuxGPU* gpu, LuxFr256* out, const LuxFr256* left, const LuxFr256* right, size_t n);
```

Poseidon2 2-to-1 compression.

### lux_gpu_merkle_root

```c
LuxError lux_gpu_merkle_root(LuxGPU* gpu, LuxFr256* out, const LuxFr256* leaves, size_t n);
```

Compute Merkle tree root from leaves.

### lux_gpu_commitment

```c
LuxError lux_gpu_commitment(LuxGPU* gpu, LuxFr256* out, const LuxFr256* values,
                            const LuxFr256* blindings, const LuxFr256* salts, size_t n);
```

Pedersen-style commitments.

### lux_gpu_nullifier

```c
LuxError lux_gpu_nullifier(LuxGPU* gpu, LuxFr256* out, const LuxFr256* keys,
                           const LuxFr256* commitments, const LuxFr256* indices, size_t n);
```

Derive nullifiers for double-spend prevention.

---

## Stream and Event Management

### Streams

```c
LuxStream* lux_stream_create(LuxGPU* gpu);
void lux_stream_destroy(LuxStream* stream);
LuxError lux_stream_sync(LuxStream* stream);
```

### Events

```c
LuxEvent* lux_event_create(LuxGPU* gpu);
void lux_event_destroy(LuxEvent* event);
LuxError lux_event_record(LuxEvent* event, LuxStream* stream);
LuxError lux_event_wait(LuxEvent* event, LuxStream* stream);
float lux_event_elapsed(LuxEvent* start, LuxEvent* end);
```

---

## Enumerations

### LuxBackend

```c
typedef enum {
    LUX_BACKEND_AUTO  = 0,  // Auto-detect best backend
    LUX_BACKEND_CPU   = 1,  // CPU with SIMD
    LUX_BACKEND_METAL = 2,  // Apple Metal
    LUX_BACKEND_CUDA  = 3,  // NVIDIA CUDA
    LUX_BACKEND_DAWN  = 4,  // WebGPU via Dawn
} LuxBackend;
```

### LuxDtype

```c
typedef enum {
    LUX_FLOAT32  = 0,
    LUX_FLOAT16  = 1,
    LUX_BFLOAT16 = 2,
    LUX_INT32    = 3,
    LUX_INT64    = 4,
    LUX_UINT32   = 5,
    LUX_UINT64   = 6,
    LUX_BOOL     = 7,
} LuxDtype;
```

### LuxError

```c
typedef enum {
    LUX_OK                         = 0,
    LUX_ERROR_INVALID_ARGUMENT     = 1,
    LUX_ERROR_OUT_OF_MEMORY        = 2,
    LUX_ERROR_BACKEND_NOT_AVAILABLE = 3,
    LUX_ERROR_DEVICE_NOT_FOUND     = 4,
    LUX_ERROR_KERNEL_FAILED        = 5,
    LUX_ERROR_NOT_SUPPORTED        = 6,
} LuxError;
```

### LuxCurve

```c
typedef enum {
    LUX_CURVE_BLS12_381 = 0,
    LUX_CURVE_BN254     = 1,
    LUX_CURVE_SECP256K1 = 2,
    LUX_CURVE_ED25519   = 3,
} LuxCurve;
```

---

## Structures

### LuxDeviceInfo

```c
typedef struct {
    LuxBackend backend;
    int index;
    const char* name;
    const char* vendor;
    uint64_t memory_total;
    uint64_t memory_available;
    bool is_discrete;
    bool is_unified_memory;
    int compute_units;
    int max_workgroup_size;
} LuxDeviceInfo;
```
