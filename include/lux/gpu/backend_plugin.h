// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Backend Plugin ABI - Stable C interface for runtime-loaded GPU backends
//
// Each backend shared library exports one symbol: lux_gpu_backend_init
// The core library dlopen()s backends and calls this to get the vtable.

#ifndef LUX_GPU_BACKEND_PLUGIN_H
#define LUX_GPU_BACKEND_PLUGIN_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// ABI Version - bump on breaking changes
// =============================================================================

#define LUX_GPU_BACKEND_ABI_VERSION 2

// =============================================================================
// Forward declarations (opaque handles)
// =============================================================================

typedef struct LuxBackendContext LuxBackendContext;
typedef struct LuxBackendBuffer LuxBackendBuffer;
typedef struct LuxBackendKernel LuxBackendKernel;

// =============================================================================
// Data types (must match lux/gpu.h)
// =============================================================================

typedef enum {
    LUX_DTYPE_FLOAT32 = 0,
    LUX_DTYPE_FLOAT16 = 1,
    LUX_DTYPE_BFLOAT16 = 2,
    LUX_DTYPE_INT32 = 3,
    LUX_DTYPE_INT64 = 4,
    LUX_DTYPE_UINT32 = 5,
    LUX_DTYPE_UINT64 = 6,
    LUX_DTYPE_BOOL = 7,
} LuxBackendDtype;

typedef enum {
    LUX_BACKEND_OK = 0,
    LUX_BACKEND_ERROR_INVALID_ARGUMENT = 1,
    LUX_BACKEND_ERROR_OUT_OF_MEMORY = 2,
    LUX_BACKEND_ERROR_NOT_SUPPORTED = 3,
    LUX_BACKEND_ERROR_DEVICE_LOST = 4,
    LUX_BACKEND_ERROR_INTERNAL = 5,
} LuxBackendError;

typedef struct {
    const char* name;
    const char* vendor;
    uint64_t memory_total;
    uint64_t memory_available;
    int compute_units;
    int max_workgroup_size;
    bool is_discrete;
    bool is_unified_memory;
} LuxBackendDeviceInfo;

// =============================================================================
// Curve type identifiers for elliptic curve operations
// Use LuxCurve from lux/gpu.h - values must match:
//   LUX_CURVE_BLS12_381 = 0
//   LUX_CURVE_BN254     = 1
//   LUX_CURVE_SECP256K1 = 2
//   LUX_CURVE_ED25519   = 3
// =============================================================================

// =============================================================================
// Backend Virtual Table
// =============================================================================

typedef struct lux_gpu_backend_vtbl {
    // Lifecycle
    LuxBackendContext* (*create_context)(int device_index);
    void (*destroy_context)(LuxBackendContext* ctx);

    // Device info
    LuxBackendError (*get_device_count)(int* count);
    LuxBackendError (*get_device_info)(LuxBackendContext* ctx, LuxBackendDeviceInfo* info);

    // Synchronization
    LuxBackendError (*sync)(LuxBackendContext* ctx);

    // Buffer management
    LuxBackendBuffer* (*buffer_alloc)(LuxBackendContext* ctx, size_t bytes);
    LuxBackendBuffer* (*buffer_alloc_with_data)(LuxBackendContext* ctx, const void* data, size_t bytes);
    void (*buffer_free)(LuxBackendContext* ctx, LuxBackendBuffer* buf);
    LuxBackendError (*buffer_copy_to_host)(LuxBackendContext* ctx, LuxBackendBuffer* buf, void* dst, size_t bytes);
    LuxBackendError (*buffer_copy_from_host)(LuxBackendContext* ctx, LuxBackendBuffer* buf, const void* src, size_t bytes);
    void* (*buffer_get_host_ptr)(LuxBackendContext* ctx, LuxBackendBuffer* buf);  // For unified memory

    // Kernel management (for custom kernels)
    LuxBackendKernel* (*kernel_load)(LuxBackendContext* ctx, const char* source, const char* entry_point);
    LuxBackendKernel* (*kernel_load_binary)(LuxBackendContext* ctx, const void* binary, size_t size, const char* entry_point);
    void (*kernel_destroy)(LuxBackendContext* ctx, LuxBackendKernel* kernel);

    // Kernel dispatch
    LuxBackendError (*kernel_dispatch)(
        LuxBackendContext* ctx,
        LuxBackendKernel* kernel,
        uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
        uint32_t block_x, uint32_t block_y, uint32_t block_z,
        LuxBackendBuffer** buffers, int num_buffers
    );

    // ==========================================================================
    // Elementwise Operations
    // ==========================================================================

    LuxBackendError (*op_add_f32)(LuxBackendContext* ctx, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n);
    LuxBackendError (*op_sub_f32)(LuxBackendContext* ctx, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n);
    LuxBackendError (*op_mul_f32)(LuxBackendContext* ctx, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n);
    LuxBackendError (*op_div_f32)(LuxBackendContext* ctx, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n);

    // ==========================================================================
    // Matrix Operations
    // ==========================================================================

    LuxBackendError (*op_matmul_f32)(LuxBackendContext* ctx, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, int M, int K, int N);
    LuxBackendError (*op_transpose_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, int rows, int cols);

    // ==========================================================================
    // Reduction Operations
    // ==========================================================================

    // Full array reductions (n elements -> 1 scalar)
    LuxBackendError (*op_reduce_sum_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n);
    LuxBackendError (*op_reduce_max_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n);
    LuxBackendError (*op_reduce_min_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n);
    LuxBackendError (*op_reduce_mean_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n);

    // Axis reductions (outer_size x inner_size -> outer_size)
    LuxBackendError (*op_reduce_sum_axis_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t outer_size, size_t inner_size);
    LuxBackendError (*op_reduce_max_axis_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t outer_size, size_t inner_size);

    // ==========================================================================
    // Softmax Operations
    // ==========================================================================

    LuxBackendError (*op_softmax_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t batch_size, size_t dim);
    LuxBackendError (*op_log_softmax_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t batch_size, size_t dim);

    // ==========================================================================
    // Unary Operations
    // ==========================================================================

    LuxBackendError (*op_exp_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n);
    LuxBackendError (*op_log_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n);
    LuxBackendError (*op_sqrt_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n);
    LuxBackendError (*op_neg_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n);
    LuxBackendError (*op_abs_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n);
    LuxBackendError (*op_tanh_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n);
    LuxBackendError (*op_sigmoid_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n);
    LuxBackendError (*op_relu_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n);
    LuxBackendError (*op_gelu_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n);

    // ==========================================================================
    // Copy Operations
    // ==========================================================================

    LuxBackendError (*op_copy_f32)(LuxBackendContext* ctx, LuxBackendBuffer* src, LuxBackendBuffer* dst, size_t n);

    // ==========================================================================
    // Normalization Operations
    // ==========================================================================

    LuxBackendError (*op_layer_norm_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, LuxBackendBuffer* gamma, LuxBackendBuffer* beta, size_t batch_size, size_t dim, float eps);
    LuxBackendError (*op_rms_norm_f32)(LuxBackendContext* ctx, LuxBackendBuffer* in, LuxBackendBuffer* out, LuxBackendBuffer* weight, size_t batch_size, size_t dim, float eps);

    // ==========================================================================
    // NTT Operations (for FHE/ZK)
    // ==========================================================================
    LuxBackendError (*op_ntt_forward)(LuxBackendContext* ctx, uint64_t* data, size_t n, uint64_t modulus);
    LuxBackendError (*op_ntt_inverse)(LuxBackendContext* ctx, uint64_t* data, size_t n, uint64_t modulus);

    // MSM operations (for ZK)
    LuxBackendError (*op_msm)(LuxBackendContext* ctx, const void* scalars, const void* points, void* result, size_t n, int curve_type);

    // ==========================================================================
    // FHE Operations (Fully Homomorphic Encryption)
    // ==========================================================================

    // Polynomial multiplication via NTT: result = a * b mod (X^n + 1) mod q
    LuxBackendError (*op_poly_mul)(
        LuxBackendContext* ctx,
        const uint64_t* a,
        const uint64_t* b,
        uint64_t* result,
        size_t n,
        uint64_t modulus
    );

    // TFHE programmable bootstrap: evaluates LUT on encrypted input
    LuxBackendError (*op_tfhe_bootstrap)(
        LuxBackendContext* ctx,
        const uint64_t* lwe_in,       // Input LWE [n_lwe + 1]
        uint64_t* lwe_out,            // Output LWE [N + 1]
        const uint64_t* bsk,          // Bootstrapping key
        const uint64_t* test_poly,    // Test polynomial (LUT)
        uint32_t n_lwe,               // Input LWE dimension
        uint32_t N,                   // GLWE polynomial degree
        uint32_t k,                   // GLWE dimension
        uint32_t l,                   // Decomposition levels
        uint64_t q                    // Modulus
    );

    // TFHE key switching: changes LWE key
    LuxBackendError (*op_tfhe_keyswitch)(
        LuxBackendContext* ctx,
        const uint64_t* lwe_in,       // Input LWE [n_in + 1]
        uint64_t* lwe_out,            // Output LWE [n_out + 1]
        const uint64_t* ksk,          // Key switching key
        uint32_t n_in,                // Input dimension
        uint32_t n_out,               // Output dimension
        uint32_t l,                   // Decomposition levels
        uint32_t base_log,            // Base log
        uint64_t q                    // Modulus
    );

    // Blind rotation: rotates polynomial accumulator by encrypted amount
    LuxBackendError (*op_blind_rotate)(
        LuxBackendContext* ctx,
        uint64_t* acc,                // Accumulator GLWE [(k+1) * N]
        const uint64_t* bsk,          // Bootstrapping key
        const uint64_t* lwe_a,        // LWE 'a' coefficients [n_lwe]
        uint32_t n_lwe,               // LWE dimension
        uint32_t N,                   // GLWE polynomial degree
        uint32_t k,                   // GLWE dimension
        uint32_t l,                   // Decomposition levels
        uint64_t q                    // Modulus
    );

    // Sample extraction: extracts LWE from GLWE at position 0
    LuxBackendError (*op_sample_extract)(
        LuxBackendContext* ctx,
        const uint64_t* glwe,         // Input GLWE [(k+1) * N]
        uint64_t* lwe,                // Output LWE [N + 1]
        uint32_t N,                   // Polynomial degree
        uint32_t k,                   // GLWE dimension
        uint64_t q                    // Modulus
    );

    // Sample polynomial in NTT domain with discrete Gaussian noise
    LuxBackendError (*op_sample_ntt)(
        LuxBackendContext* ctx,
        uint64_t* output,             // Output polynomial in NTT domain
        size_t n,                     // Polynomial degree
        uint64_t modulus,             // Prime modulus
        double sigma,                 // Standard deviation for Gaussian
        uint64_t seed                 // RNG seed
    );

    // ==========================================================================
    // Crypto: Hash Operations
    // ==========================================================================

    // Poseidon2 hash (algebraic hash for ZK)
    LuxBackendError (*op_poseidon2_hash)(
        LuxBackendContext* ctx,
        const uint64_t* inputs,       // Input field elements [num_hashes * rate]
        uint64_t* outputs,            // Output hashes [num_hashes]
        size_t rate,                  // Poseidon rate parameter
        size_t num_hashes             // Number of parallel hashes
    );

    // BLAKE3 hash
    LuxBackendError (*op_blake3_hash)(
        LuxBackendContext* ctx,
        const uint8_t* inputs,        // Input data (concatenated)
        uint8_t* outputs,             // Output 32-byte hashes [num_hashes * 32]
        const size_t* input_lens,     // Length of each input
        size_t num_hashes             // Number of parallel hashes
    );

    // Keccak-256 hash (Ethereum variant)
    LuxBackendError (*op_keccak256_hash)(
        LuxBackendContext* ctx,
        const uint8_t* inputs,        // Input data (concatenated)
        uint8_t* outputs,             // Output 32-byte hashes [num_inputs * 32]
        const size_t* input_lens,     // Length of each input
        size_t num_inputs             // Number of parallel hashes
    );

    // ==========================================================================
    // Crypto: BLS12-381 Curve Operations
    // ==========================================================================

    // Point addition (G1 or G2)
    LuxBackendError (*op_bls12_381_add)(
        LuxBackendContext* ctx,
        const void* a,                // Points (G1 or G2)
        const void* b,                // Points (G1 or G2)
        void* out,                    // Sum points
        size_t n,                     // Number of point pairs
        bool is_g2                    // true for G2, false for G1
    );

    // Scalar multiplication (G1 or G2)
    LuxBackendError (*op_bls12_381_mul)(
        LuxBackendContext* ctx,
        const void* points,           // Points (G1 or G2)
        const void* scalars,          // Scalar field elements
        void* out,                    // Product points
        size_t n,                     // Number of operations
        bool is_g2                    // true for G2, false for G1
    );

    // Pairing computation (multi-pairing supported)
    LuxBackendError (*op_bls12_381_pairing)(
        LuxBackendContext* ctx,
        const void* g1_points,        // G1 points
        const void* g2_points,        // G2 points
        void* out,                    // Pairing result (Gt element)
        size_t n                      // Number of pairings (multi-pairing)
    );

    // ==========================================================================
    // Crypto: BN254 Curve Operations
    // ==========================================================================

    // Point addition (G1 or G2)
    LuxBackendError (*op_bn254_add)(
        LuxBackendContext* ctx,
        const void* a,                // Points
        const void* b,                // Points
        void* out,                    // Sum points
        size_t n,                     // Number of point pairs
        bool is_g2                    // true for G2, false for G1
    );

    // Scalar multiplication (G1 or G2)
    LuxBackendError (*op_bn254_mul)(
        LuxBackendContext* ctx,
        const void* points,           // Points
        const void* scalars,          // Scalar field elements
        void* out,                    // Product points
        size_t n,                     // Number of operations
        bool is_g2                    // true for G2, false for G1
    );

    // ==========================================================================
    // Crypto: KZG Polynomial Commitments
    // ==========================================================================

    // Commit to polynomial using SRS
    LuxBackendError (*op_kzg_commit)(
        LuxBackendContext* ctx,
        const void* coeffs,           // Polynomial coefficients (field elements)
        const void* srs,              // SRS G1 points (powers of tau)
        void* commitment,             // Output commitment point
        size_t degree,                // Polynomial degree
        int curve_type                // LuxCurveType
    );

    // Open commitment at evaluation point
    LuxBackendError (*op_kzg_open)(
        LuxBackendContext* ctx,
        const void* coeffs,           // Polynomial coefficients
        const void* srs,              // SRS G1 points
        const void* point,            // Evaluation point (field element)
        void* proof,                  // Output proof (point)
        size_t degree,                // Polynomial degree
        int curve_type                // LuxCurveType
    );

    // Verify KZG opening proof
    LuxBackendError (*op_kzg_verify)(
        LuxBackendContext* ctx,
        const void* commitment,       // Commitment point
        const void* proof,            // Proof point
        const void* point,            // Evaluation point
        const void* value,            // Claimed evaluation
        const void* srs_g2,           // G2 element from SRS
        bool* result,                 // Verification result
        int curve_type                // LuxCurveType
    );

    // Reserved for future expansion (don't break ABI)
    void* _reserved[4];

} lux_gpu_backend_vtbl;

// =============================================================================
// Backend Descriptor (returned by plugin init)
// =============================================================================

typedef struct {
    uint32_t abi_version;           // Must be LUX_GPU_BACKEND_ABI_VERSION
    const char* backend_name;       // "cpu" | "metal" | "cuda" | "webgpu"
    const char* backend_version;    // e.g., "0.1.0"
    uint32_t capabilities;          // Bitmask of supported features
    const lux_gpu_backend_vtbl* vtbl;
} lux_gpu_backend_desc;

// Capability flags
#define LUX_CAP_TENSOR_OPS      (1 << 0)   // Basic tensor ops (add, sub, mul, div)
#define LUX_CAP_MATMUL          (1 << 1)   // Matrix multiplication
#define LUX_CAP_NTT             (1 << 2)   // NTT operations
#define LUX_CAP_MSM             (1 << 3)   // Multi-scalar multiplication
#define LUX_CAP_CUSTOM_KERNELS  (1 << 4)   // Custom kernel loading
#define LUX_CAP_UNIFIED_MEMORY  (1 << 5)   // Unified memory support
#define LUX_CAP_FHE             (1 << 6)   // Fully homomorphic encryption
#define LUX_CAP_TFHE            (1 << 7)   // TFHE bootstrap/keyswitch
#define LUX_CAP_REDUCE          (1 << 8)   // Reduction ops (sum, max, min, mean)
#define LUX_CAP_SOFTMAX         (1 << 9)   // Softmax and log-softmax
#define LUX_CAP_UNARY           (1 << 10)  // Unary ops (exp, log, sqrt, tanh, etc.)
#define LUX_CAP_NORMALIZATION   (1 << 11)  // Layer norm, RMS norm
#define LUX_CAP_BLS12_381       (1 << 12)  // BLS12-381 curve operations
#define LUX_CAP_BN254           (1 << 13)  // BN254 curve operations
#define LUX_CAP_KZG             (1 << 14)  // KZG polynomial commitments
#define LUX_CAP_POSEIDON2       (1 << 15)  // Poseidon2 hash
#define LUX_CAP_BLAKE3          (1 << 16)  // BLAKE3 hash
#define LUX_CAP_BLIND_ROTATE    (1 << 17)  // Blind rotation
#define LUX_CAP_POLY_MUL        (1 << 18)  // Polynomial multiplication
#define LUX_CAP_KECCAK256       (1 << 19)  // Keccak-256 hash (Ethereum)

// =============================================================================
// Plugin Entry Point
// =============================================================================

// Every backend shared library must export this symbol
// Returns true on success, false if backend unavailable on this system
typedef bool (*lux_gpu_backend_init_fn)(lux_gpu_backend_desc* out);

// Symbol name to dlopen
#define LUX_GPU_BACKEND_INIT_SYMBOL "lux_gpu_backend_init"

// Macro to declare the entry point
#ifdef _WIN32
#define LUX_GPU_BACKEND_EXPORT __declspec(dllexport)
#else
#define LUX_GPU_BACKEND_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#define LUX_GPU_DECLARE_BACKEND(init_func) \
    extern "C" LUX_GPU_BACKEND_EXPORT bool lux_gpu_backend_init(lux_gpu_backend_desc* out) { \
        return init_func(out); \
    }
#else
#define LUX_GPU_DECLARE_BACKEND(init_func) \
    LUX_GPU_BACKEND_EXPORT bool lux_gpu_backend_init(lux_gpu_backend_desc* out) { \
        return init_func(out); \
    }
#endif

#ifdef __cplusplus
}
#endif

#endif // LUX_GPU_BACKEND_PLUGIN_H
