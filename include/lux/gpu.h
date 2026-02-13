// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Lux GPU - Unified GPU acceleration with switchable backends
//
// Backends:
//   - Metal: Apple Silicon (macOS/iOS)
//   - CUDA: NVIDIA GPUs
//   - Dawn: WebGPU via Dawn (cross-platform)
//   - CPU: SIMD-optimized fallback
//
// Usage:
//   #include <lux/gpu.h>
//
//   LuxGPU* gpu = lux_gpu_create();
//   lux_gpu_set_backend(gpu, LUX_BACKEND_METAL);
//
//   LuxTensor* a = lux_tensor_zeros(gpu, shape, 2, LUX_FLOAT32);
//   LuxTensor* b = lux_tensor_ones(gpu, shape, 2, LUX_FLOAT32);
//   LuxTensor* c = lux_tensor_add(gpu, a, b);
//
//   lux_gpu_sync(gpu);
//   lux_gpu_destroy(gpu);

#ifndef LUX_GPU_H
#define LUX_GPU_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Version
// =============================================================================

#define LUX_GPU_VERSION_MAJOR 0
#define LUX_GPU_VERSION_MINOR 2
#define LUX_GPU_VERSION_PATCH 0

// =============================================================================
// Backend Types
// =============================================================================

typedef enum {
    LUX_BACKEND_AUTO = 0,  // Auto-detect best backend
    LUX_BACKEND_CPU  = 1,  // CPU with SIMD
    LUX_BACKEND_METAL = 2, // Apple Metal
    LUX_BACKEND_CUDA = 3,  // NVIDIA CUDA
    LUX_BACKEND_DAWN = 4,  // WebGPU via Dawn
} LuxBackend;

typedef enum {
    LUX_FLOAT32 = 0,
    LUX_FLOAT16 = 1,
    LUX_BFLOAT16 = 2,
    LUX_INT32 = 3,
    LUX_INT64 = 4,
    LUX_UINT32 = 5,
    LUX_UINT64 = 6,
    LUX_BOOL = 7,
} LuxDtype;

typedef enum {
    LUX_OK = 0,
    LUX_ERROR_INVALID_ARGUMENT = 1,
    LUX_ERROR_OUT_OF_MEMORY = 2,
    LUX_ERROR_BACKEND_NOT_AVAILABLE = 3,
    LUX_ERROR_DEVICE_NOT_FOUND = 4,
    LUX_ERROR_KERNEL_FAILED = 5,
    LUX_ERROR_NOT_SUPPORTED = 6,
} LuxError;

// =============================================================================
// Curve Types (for crypto operations)
// =============================================================================

typedef enum {
    LUX_CURVE_BLS12_381 = 0,
    LUX_CURVE_BN254 = 1,
    LUX_CURVE_SECP256K1 = 2,
    LUX_CURVE_ED25519 = 3,
} LuxCurve;

// =============================================================================
// Opaque Types
// =============================================================================

typedef struct LuxGPU LuxGPU;
typedef struct LuxTensor LuxTensor;
typedef struct LuxStream LuxStream;
typedef struct LuxEvent LuxEvent;

// =============================================================================
// Device Info
// =============================================================================

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

// =============================================================================
// GPU Context
// =============================================================================

// Create GPU context (auto-detects best backend)
LuxGPU* lux_gpu_create(void);

// Create GPU context with specific backend
LuxGPU* lux_gpu_create_with_backend(LuxBackend backend);

// Create GPU context with specific device
LuxGPU* lux_gpu_create_with_device(LuxBackend backend, int device_index);

// Destroy GPU context
void lux_gpu_destroy(LuxGPU* gpu);

// Get current backend
LuxBackend lux_gpu_backend(LuxGPU* gpu);

// Get backend name string
const char* lux_gpu_backend_name(LuxGPU* gpu);

// Switch backend at runtime
LuxError lux_gpu_set_backend(LuxGPU* gpu, LuxBackend backend);

// Get device info
LuxError lux_gpu_device_info(LuxGPU* gpu, LuxDeviceInfo* info);

// Synchronize all operations
LuxError lux_gpu_sync(LuxGPU* gpu);

// Get last error message
const char* lux_gpu_error(LuxGPU* gpu);

// =============================================================================
// Backend Query
// =============================================================================

// Get number of available backends
int lux_backend_count(void);

// Check if backend is available
bool lux_backend_available(LuxBackend backend);

// Get backend name
const char* lux_backend_name(LuxBackend backend);

// Get number of devices for backend
int lux_device_count(LuxBackend backend);

// Get device info for backend/index
LuxError lux_device_info(LuxBackend backend, int index, LuxDeviceInfo* info);

// =============================================================================
// Tensor Operations
// =============================================================================

// Create tensor filled with zeros
LuxTensor* lux_tensor_zeros(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype);

// Create tensor filled with ones
LuxTensor* lux_tensor_ones(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype);

// Create tensor filled with value
LuxTensor* lux_tensor_full(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype, double value);

// Create tensor from data
LuxTensor* lux_tensor_from_data(LuxGPU* gpu, const void* data, const int64_t* shape, int ndim, LuxDtype dtype);

// Destroy tensor
void lux_tensor_destroy(LuxTensor* tensor);

// Get tensor shape
int lux_tensor_ndim(LuxTensor* tensor);
int64_t lux_tensor_shape(LuxTensor* tensor, int dim);
int64_t lux_tensor_size(LuxTensor* tensor);
LuxDtype lux_tensor_dtype(LuxTensor* tensor);

// Copy tensor data to host
LuxError lux_tensor_to_host(LuxTensor* tensor, void* data, size_t size);

// Arithmetic operations
LuxTensor* lux_tensor_add(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
LuxTensor* lux_tensor_sub(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
LuxTensor* lux_tensor_mul(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
LuxTensor* lux_tensor_div(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);
LuxTensor* lux_tensor_matmul(LuxGPU* gpu, LuxTensor* a, LuxTensor* b);

// Unary operations
LuxTensor* lux_tensor_neg(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_exp(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_log(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_sqrt(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_abs(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_tanh(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_sigmoid(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_relu(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_gelu(LuxGPU* gpu, LuxTensor* t);

// Reductions (full tensor -> scalar)
float lux_tensor_reduce_sum(LuxGPU* gpu, LuxTensor* t);
float lux_tensor_reduce_max(LuxGPU* gpu, LuxTensor* t);
float lux_tensor_reduce_min(LuxGPU* gpu, LuxTensor* t);
float lux_tensor_reduce_mean(LuxGPU* gpu, LuxTensor* t);

// Reductions along axes
LuxTensor* lux_tensor_sum(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes);
LuxTensor* lux_tensor_mean(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes);
LuxTensor* lux_tensor_max(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes);
LuxTensor* lux_tensor_min(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes);

// Softmax and normalization
LuxTensor* lux_tensor_softmax(LuxGPU* gpu, LuxTensor* t, int axis);
LuxTensor* lux_tensor_log_softmax(LuxGPU* gpu, LuxTensor* t, int axis);
LuxTensor* lux_tensor_layer_norm(LuxGPU* gpu, LuxTensor* t, LuxTensor* gamma, LuxTensor* beta, float eps);
LuxTensor* lux_tensor_rms_norm(LuxGPU* gpu, LuxTensor* t, LuxTensor* weight, float eps);

// Transpose and copy
LuxTensor* lux_tensor_transpose(LuxGPU* gpu, LuxTensor* t);
LuxTensor* lux_tensor_copy(LuxGPU* gpu, LuxTensor* t);

// =============================================================================
// Crypto Operations: Hash Functions
// =============================================================================

// Poseidon2 hash (algebraic hash for ZK circuits)
LuxError lux_poseidon2_hash(LuxGPU* gpu,
                            const uint64_t* inputs,      // [num_hashes * rate]
                            uint64_t* outputs,           // [num_hashes]
                            size_t rate,                 // Poseidon rate parameter
                            size_t num_hashes);

// BLAKE3 hash (high-performance cryptographic hash)
LuxError lux_blake3_hash(LuxGPU* gpu,
                         const uint8_t* inputs,         // Concatenated inputs
                         uint8_t* outputs,              // [num_hashes * 32]
                         const size_t* input_lens,      // Length of each input
                         size_t num_hashes);

// =============================================================================
// Crypto Operations: MSM (Multi-Scalar Multiplication)
// =============================================================================

LuxError lux_msm(LuxGPU* gpu,
                 const void* scalars,           // Scalar field elements
                 const void* points,            // Curve points (affine)
                 void* result,                  // Single output point
                 size_t count,                  // Number of scalar-point pairs
                 LuxCurve curve);               // Which curve to use

// =============================================================================
// Crypto Operations: BLS12-381 Curve
// =============================================================================

// Point addition (G1 or G2)
LuxError lux_bls12_381_add(LuxGPU* gpu,
                           const void* a, const void* b, void* out,
                           size_t count, bool is_g2);

// Scalar multiplication (G1 or G2)
LuxError lux_bls12_381_mul(LuxGPU* gpu,
                           const void* points, const void* scalars, void* out,
                           size_t count, bool is_g2);

// Pairing computation (multi-pairing for efficiency)
LuxError lux_bls12_381_pairing(LuxGPU* gpu,
                               const void* g1_points, const void* g2_points,
                               void* out, size_t count);

// High-level BLS signature verification
LuxError lux_bls_verify(LuxGPU* gpu,
                        const uint8_t* sig, size_t sig_len,
                        const uint8_t* msg, size_t msg_len,
                        const uint8_t* pubkey, size_t pubkey_len,
                        bool* result);

LuxError lux_bls_verify_batch(LuxGPU* gpu,
                              const uint8_t* const* sigs, const size_t* sig_lens,
                              const uint8_t* const* msgs, const size_t* msg_lens,
                              const uint8_t* const* pubkeys, const size_t* pubkey_lens,
                              int count, bool* results);

LuxError lux_bls_aggregate(LuxGPU* gpu,
                           const uint8_t* const* sigs, const size_t* sig_lens,
                           int count, uint8_t* out, size_t* out_len);

// =============================================================================
// Crypto Operations: BN254 Curve
// =============================================================================

// Point addition (G1 or G2)
LuxError lux_bn254_add(LuxGPU* gpu,
                       const void* a, const void* b, void* out,
                       size_t count, bool is_g2);

// Scalar multiplication (G1 or G2)
LuxError lux_bn254_mul(LuxGPU* gpu,
                       const void* points, const void* scalars, void* out,
                       size_t count, bool is_g2);

// =============================================================================
// Crypto Operations: KZG Polynomial Commitments
// =============================================================================

// Commit to polynomial using SRS
LuxError lux_kzg_commit(LuxGPU* gpu,
                        const void* coeffs,        // Polynomial coefficients
                        const void* srs,           // SRS G1 points
                        void* commitment,          // Output commitment
                        size_t degree,             // Polynomial degree
                        LuxCurve curve);

// Open commitment at evaluation point
LuxError lux_kzg_open(LuxGPU* gpu,
                      const void* coeffs,          // Polynomial coefficients
                      const void* srs,             // SRS G1 points
                      const void* point,           // Evaluation point
                      void* proof,                 // Output proof
                      size_t degree,               // Polynomial degree
                      LuxCurve curve);

// Verify KZG opening proof
LuxError lux_kzg_verify(LuxGPU* gpu,
                        const void* commitment,    // Commitment point
                        const void* proof,         // Proof point
                        const void* point,         // Evaluation point
                        const void* value,         // Claimed evaluation
                        const void* srs_g2,        // G2 element from SRS
                        bool* result,              // Verification result
                        LuxCurve curve);

// =============================================================================
// FHE Operations: NTT (Number Theoretic Transform)
// =============================================================================

LuxError lux_ntt_forward(LuxGPU* gpu, uint64_t* data, size_t n, uint64_t modulus);
LuxError lux_ntt_inverse(LuxGPU* gpu, uint64_t* data, size_t n, uint64_t modulus);
LuxError lux_ntt_batch(LuxGPU* gpu, uint64_t** polys, size_t count, size_t n, uint64_t modulus);

// =============================================================================
// FHE Operations: Polynomial Arithmetic
// =============================================================================

// Polynomial multiplication: result = a * b mod (X^n + 1) mod modulus
LuxError lux_poly_mul(LuxGPU* gpu,
                      const uint64_t* a, const uint64_t* b,
                      uint64_t* result, size_t n, uint64_t modulus);

// =============================================================================
// FHE Operations: TFHE
// =============================================================================

// TFHE programmable bootstrap: evaluates LUT on encrypted input
LuxError lux_tfhe_bootstrap(LuxGPU* gpu,
                            const uint64_t* lwe_in,       // Input LWE [n_lwe + 1]
                            uint64_t* lwe_out,            // Output LWE [N + 1]
                            const uint64_t* bsk,          // Bootstrapping key
                            const uint64_t* test_poly,    // Test polynomial (LUT)
                            uint32_t n_lwe,               // Input LWE dimension
                            uint32_t N,                   // GLWE polynomial degree
                            uint32_t k,                   // GLWE dimension
                            uint32_t l,                   // Decomposition levels
                            uint64_t q);                  // Modulus

// TFHE key switching: changes LWE key
LuxError lux_tfhe_keyswitch(LuxGPU* gpu,
                            const uint64_t* lwe_in,       // Input LWE [n_in + 1]
                            uint64_t* lwe_out,            // Output LWE [n_out + 1]
                            const uint64_t* ksk,          // Key switching key
                            uint32_t n_in,                // Input dimension
                            uint32_t n_out,               // Output dimension
                            uint32_t l,                   // Decomposition levels
                            uint32_t base_log,            // Base log
                            uint64_t q);                  // Modulus

// Blind rotation: rotates polynomial accumulator by encrypted amount
LuxError lux_blind_rotate(LuxGPU* gpu,
                          uint64_t* acc,                  // Accumulator GLWE [(k+1) * N]
                          const uint64_t* bsk,            // Bootstrapping key
                          const uint64_t* lwe_a,          // LWE 'a' coefficients [n_lwe]
                          uint32_t n_lwe,                 // LWE dimension
                          uint32_t N,                     // GLWE polynomial degree
                          uint32_t k,                     // GLWE dimension
                          uint32_t l,                     // Decomposition levels
                          uint64_t q);                    // Modulus

// =============================================================================
// ZK Primitives: Field Elements and High-Level Operations
// =============================================================================

// BN254 scalar field element (Fr) - 256-bit integer in 4 x 64-bit limbs
// Represents elements of the scalar field of BN254 curve
typedef struct {
    uint64_t limbs[4];
} LuxFr256;

// Poseidon2 compression: out[i] = Poseidon2(left[i], right[i])
// Poseidon2 is an algebraic hash function optimized for ZK circuits.
LuxError lux_gpu_poseidon2(LuxGPU* gpu,
                           LuxFr256* out,
                           const LuxFr256* left,
                           const LuxFr256* right,
                           size_t n);

// Merkle tree root computation using Poseidon2 hash
// Computes root from n leaves (pads to next power of 2 internally)
LuxError lux_gpu_merkle_root(LuxGPU* gpu,
                             LuxFr256* out,
                             const LuxFr256* leaves,
                             size_t n);

// Pedersen-style commitment: out[i] = Poseidon2(Poseidon2(value, blinding), salt)
// Suitable for hiding commitments in ZK protocols
LuxError lux_gpu_commitment(LuxGPU* gpu,
                            LuxFr256* out,
                            const LuxFr256* values,
                            const LuxFr256* blindings,
                            const LuxFr256* salts,
                            size_t n);

// Nullifier derivation: out[i] = Poseidon2(Poseidon2(key, commitment), index)
// Used to prevent double-spending in ZK protocols
LuxError lux_gpu_nullifier(LuxGPU* gpu,
                           LuxFr256* out,
                           const LuxFr256* keys,
                           const LuxFr256* commitments,
                           const LuxFr256* indices,
                           size_t n);

// =============================================================================
// Stream/Event Management
// =============================================================================

LuxStream* lux_stream_create(LuxGPU* gpu);
void lux_stream_destroy(LuxStream* stream);
LuxError lux_stream_sync(LuxStream* stream);

LuxEvent* lux_event_create(LuxGPU* gpu);
void lux_event_destroy(LuxEvent* event);
LuxError lux_event_record(LuxEvent* event, LuxStream* stream);
LuxError lux_event_wait(LuxEvent* event, LuxStream* stream);
float lux_event_elapsed(LuxEvent* start, LuxEvent* end);

#ifdef __cplusplus
}
#endif

#endif // LUX_GPU_H
