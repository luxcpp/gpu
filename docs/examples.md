# Examples {#examples}

Code samples and usage patterns for Lux GPU Core.

## Basic Usage

### Hello GPU

Minimal example showing GPU context creation and tensor operations.

```c
#include <lux/gpu.h>
#include <stdio.h>

int main() {
    // Create GPU context (auto-selects best backend)
    LuxGPU* gpu = lux_gpu_create();
    if (!gpu) {
        fprintf(stderr, "Failed to create GPU context\n");
        return 1;
    }

    printf("Using backend: %s\n", lux_gpu_backend_name(gpu));

    // Create vectors
    int64_t shape[] = {1024};
    LuxTensor* a = lux_tensor_ones(gpu, shape, 1, LUX_FLOAT32);
    LuxTensor* b = lux_tensor_full(gpu, shape, 1, LUX_FLOAT32, 2.0);

    // Compute c = a + b
    LuxTensor* c = lux_tensor_add(gpu, a, b);

    // Verify result
    float sum = lux_tensor_reduce_sum(gpu, c);
    printf("Sum: %.1f (expected: 3072.0)\n", sum);

    // Cleanup
    lux_tensor_destroy(c);
    lux_tensor_destroy(b);
    lux_tensor_destroy(a);
    lux_gpu_destroy(gpu);

    return 0;
}
```

### Device Enumeration

Query available backends and devices.

```c
#include <lux/gpu.h>
#include <stdio.h>

int main() {
    printf("Available backends:\n");

    LuxBackend backends[] = {LUX_BACKEND_CUDA, LUX_BACKEND_METAL,
                             LUX_BACKEND_DAWN, LUX_BACKEND_CPU};
    const char* names[] = {"CUDA", "Metal", "Dawn", "CPU"};

    for (int i = 0; i < 4; i++) {
        if (lux_backend_available(backends[i])) {
            int count = lux_device_count(backends[i]);
            printf("  %s: %d device(s)\n", names[i], count);

            for (int d = 0; d < count; d++) {
                LuxDeviceInfo info;
                if (lux_device_info(backends[i], d, &info) == LUX_OK) {
                    printf("    [%d] %s (%s)\n", d, info.name, info.vendor);
                    printf("        Memory: %lu MB\n",
                           (unsigned long)(info.memory_total / (1024 * 1024)));
                    printf("        Compute units: %d\n", info.compute_units);
                }
            }
        }
    }

    return 0;
}
```

---

## Matrix Operations

### Matrix Multiplication

```c
#include <lux/gpu.h>
#include <stdlib.h>
#include <stdio.h>

int main() {
    LuxGPU* gpu = lux_gpu_create();

    // Create matrices A[1024, 512] and B[512, 2048]
    int64_t shape_a[] = {1024, 512};
    int64_t shape_b[] = {512, 2048};

    LuxTensor* A = lux_tensor_ones(gpu, shape_a, 2, LUX_FLOAT32);
    LuxTensor* B = lux_tensor_ones(gpu, shape_b, 2, LUX_FLOAT32);

    // C = A @ B -> [1024, 2048]
    LuxTensor* C = lux_tensor_matmul(gpu, A, B);
    lux_gpu_sync(gpu);

    printf("Result shape: [%ld, %ld]\n",
           lux_tensor_shape(C, 0), lux_tensor_shape(C, 1));

    // Verify: each element should be 512 (sum of 512 ones)
    float mean = lux_tensor_reduce_mean(gpu, C);
    printf("Mean value: %.1f (expected: 512.0)\n", mean);

    lux_tensor_destroy(C);
    lux_tensor_destroy(B);
    lux_tensor_destroy(A);
    lux_gpu_destroy(gpu);

    return 0;
}
```

### Batch Matrix Multiply with Reshape

```c
#include <lux/gpu.h>

void batch_matmul_example(LuxGPU* gpu) {
    // Simulate batch matmul by flattening batch dimension
    // Input: [batch, M, K] @ [batch, K, N] -> [batch, M, N]

    int batch = 32;
    int M = 64, K = 128, N = 64;

    // For now, process each batch element sequentially
    int64_t shape_a[] = {M, K};
    int64_t shape_b[] = {K, N};

    LuxTensor* A = lux_tensor_ones(gpu, shape_a, 2, LUX_FLOAT32);
    LuxTensor* B = lux_tensor_ones(gpu, shape_b, 2, LUX_FLOAT32);

    for (int b = 0; b < batch; b++) {
        LuxTensor* C = lux_tensor_matmul(gpu, A, B);
        // Process result...
        lux_tensor_destroy(C);
    }

    lux_tensor_destroy(B);
    lux_tensor_destroy(A);
}
```

---

## Neural Network Layers

### Softmax

```c
#include <lux/gpu.h>
#include <stdio.h>

void softmax_example(LuxGPU* gpu) {
    // Logits: [batch=32, classes=1000]
    int64_t shape[] = {32, 1000};
    LuxTensor* logits = lux_tensor_ones(gpu, shape, 2, LUX_FLOAT32);

    // Softmax along last axis
    LuxTensor* probs = lux_tensor_softmax(gpu, logits, 1);

    // Each row should sum to 1.0
    float total = lux_tensor_reduce_sum(gpu, probs);
    printf("Total probability mass: %.1f (expected: 32.0)\n", total);

    lux_tensor_destroy(probs);
    lux_tensor_destroy(logits);
}
```

### Layer Normalization

```c
#include <lux/gpu.h>

void layer_norm_example(LuxGPU* gpu) {
    // Input: [batch=16, seq=512, hidden=768]
    int64_t input_shape[] = {16 * 512, 768};  // Flattened for 2D
    int64_t param_shape[] = {768};

    LuxTensor* x = lux_tensor_ones(gpu, input_shape, 2, LUX_FLOAT32);
    LuxTensor* gamma = lux_tensor_ones(gpu, param_shape, 1, LUX_FLOAT32);
    LuxTensor* beta = lux_tensor_zeros(gpu, param_shape, 1, LUX_FLOAT32);

    LuxTensor* normed = lux_tensor_layer_norm(gpu, x, gamma, beta, 1e-5f);

    // Mean should be ~0, std should be ~1
    float mean = lux_tensor_reduce_mean(gpu, normed);
    printf("Post-norm mean: %f\n", mean);

    lux_tensor_destroy(normed);
    lux_tensor_destroy(beta);
    lux_tensor_destroy(gamma);
    lux_tensor_destroy(x);
}
```

### RMS Normalization (LLaMA-style)

```c
#include <lux/gpu.h>

void rms_norm_example(LuxGPU* gpu) {
    int64_t input_shape[] = {8192, 4096};  // [tokens, hidden]
    int64_t weight_shape[] = {4096};

    LuxTensor* x = lux_tensor_ones(gpu, input_shape, 2, LUX_FLOAT32);
    LuxTensor* weight = lux_tensor_ones(gpu, weight_shape, 1, LUX_FLOAT32);

    LuxTensor* normed = lux_tensor_rms_norm(gpu, x, weight, 1e-6f);

    lux_tensor_destroy(normed);
    lux_tensor_destroy(weight);
    lux_tensor_destroy(x);
}
```

### GELU Activation

```c
#include <lux/gpu.h>

void mlp_example(LuxGPU* gpu) {
    // MLP: Linear -> GELU -> Linear
    int64_t shape[] = {1024, 2048};

    LuxTensor* x = lux_tensor_ones(gpu, shape, 2, LUX_FLOAT32);
    LuxTensor* activated = lux_tensor_gelu(gpu, x);

    // GELU(1) approx 0.841
    float mean = lux_tensor_reduce_mean(gpu, activated);
    printf("GELU(1) mean: %f\n", mean);

    lux_tensor_destroy(activated);
    lux_tensor_destroy(x);
}
```

---

## Cryptographic Operations

### Poseidon2 Hash

```c
#include <lux/gpu.h>
#include <stdio.h>
#include <string.h>

void poseidon2_example(LuxGPU* gpu) {
    // Hash 1000 pairs of field elements
    size_t n = 1000;
    LuxFr256* left = malloc(n * sizeof(LuxFr256));
    LuxFr256* right = malloc(n * sizeof(LuxFr256));
    LuxFr256* out = malloc(n * sizeof(LuxFr256));

    // Initialize with test data
    for (size_t i = 0; i < n; i++) {
        memset(&left[i], 0, sizeof(LuxFr256));
        memset(&right[i], 0, sizeof(LuxFr256));
        left[i].limbs[0] = i;
        right[i].limbs[0] = i + 1;
    }

    LuxError err = lux_gpu_poseidon2(gpu, out, left, right, n);
    if (err == LUX_OK) {
        printf("Poseidon2: computed %zu hashes\n", n);
        printf("First hash limb[0]: 0x%016lx\n", (unsigned long)out[0].limbs[0]);
    }

    free(out);
    free(right);
    free(left);
}
```

### Merkle Tree Root

```c
#include <lux/gpu.h>
#include <stdio.h>
#include <string.h>

void merkle_example(LuxGPU* gpu) {
    // Compute Merkle root from 1024 leaves
    size_t n = 1024;
    LuxFr256* leaves = malloc(n * sizeof(LuxFr256));
    LuxFr256 root;

    // Initialize leaves
    for (size_t i = 0; i < n; i++) {
        memset(&leaves[i], 0, sizeof(LuxFr256));
        leaves[i].limbs[0] = i;
    }

    LuxError err = lux_gpu_merkle_root(gpu, &root, leaves, n);
    if (err == LUX_OK) {
        printf("Merkle root computed from %zu leaves\n", n);
        printf("Root: 0x%016lx...\n", (unsigned long)root.limbs[0]);
    }

    free(leaves);
}
```

### BLS Signature Verification

```c
#include <lux/gpu.h>
#include <stdio.h>

void bls_verify_example(LuxGPU* gpu) {
    // Example BLS signature verification
    // (Using placeholder data - real signatures would come from signing)

    uint8_t signature[96];  // G2 point
    uint8_t message[] = "Hello, Lux!";
    uint8_t pubkey[48];     // G1 point

    // Initialize with zeros (placeholder)
    memset(signature, 0, sizeof(signature));
    memset(pubkey, 0, sizeof(pubkey));

    bool valid = false;
    LuxError err = lux_bls_verify(gpu,
                                   signature, sizeof(signature),
                                   message, sizeof(message) - 1,
                                   pubkey, sizeof(pubkey),
                                   &valid);

    if (err == LUX_OK) {
        printf("Signature valid: %s\n", valid ? "true" : "false");
    } else {
        printf("Verification error: %d\n", err);
    }
}
```

### Batch BLS Verification

```c
#include <lux/gpu.h>
#include <stdio.h>

void batch_bls_verify_example(LuxGPU* gpu) {
    // Verify 100 signatures in parallel
    int count = 100;

    // Allocate arrays (simplified - real code would populate these)
    uint8_t** sigs = malloc(count * sizeof(uint8_t*));
    size_t* sig_lens = malloc(count * sizeof(size_t));
    uint8_t** msgs = malloc(count * sizeof(uint8_t*));
    size_t* msg_lens = malloc(count * sizeof(size_t));
    uint8_t** pubkeys = malloc(count * sizeof(uint8_t*));
    size_t* pubkey_lens = malloc(count * sizeof(size_t));
    bool* results = malloc(count * sizeof(bool));

    // Initialize (placeholder)
    for (int i = 0; i < count; i++) {
        sigs[i] = calloc(96, 1);
        sig_lens[i] = 96;
        msgs[i] = (uint8_t*)"test message";
        msg_lens[i] = 12;
        pubkeys[i] = calloc(48, 1);
        pubkey_lens[i] = 48;
    }

    LuxError err = lux_bls_verify_batch(gpu,
                                         (const uint8_t* const*)sigs, sig_lens,
                                         (const uint8_t* const*)msgs, msg_lens,
                                         (const uint8_t* const*)pubkeys, pubkey_lens,
                                         count, results);

    if (err == LUX_OK) {
        int valid_count = 0;
        for (int i = 0; i < count; i++) {
            if (results[i]) valid_count++;
        }
        printf("Valid signatures: %d / %d\n", valid_count, count);
    }

    // Cleanup
    for (int i = 0; i < count; i++) {
        free(sigs[i]);
        free(pubkeys[i]);
    }
    free(sigs);
    free(sig_lens);
    free(msgs);
    free(msg_lens);
    free(pubkeys);
    free(pubkey_lens);
    free(results);
}
```

---

## FHE Operations

### NTT (Number Theoretic Transform)

```c
#include <lux/gpu.h>
#include <stdio.h>
#include <stdlib.h>

void ntt_example(LuxGPU* gpu) {
    // NTT for polynomial multiplication in FHE
    size_t n = 4096;  // Polynomial degree (power of 2)
    uint64_t modulus = 0xFFFFFFFF00000001ULL;  // Goldilocks prime

    uint64_t* poly = malloc(n * sizeof(uint64_t));

    // Initialize polynomial coefficients
    for (size_t i = 0; i < n; i++) {
        poly[i] = i % modulus;
    }

    // Forward NTT
    LuxError err = lux_ntt_forward(gpu, poly, n, modulus);
    if (err != LUX_OK) {
        printf("NTT forward failed\n");
        free(poly);
        return;
    }

    printf("Forward NTT complete\n");

    // Inverse NTT
    err = lux_ntt_inverse(gpu, poly, n, modulus);
    if (err != LUX_OK) {
        printf("NTT inverse failed\n");
        free(poly);
        return;
    }

    printf("Inverse NTT complete\n");

    // Verify roundtrip
    int errors = 0;
    for (size_t i = 0; i < n; i++) {
        if (poly[i] != i % modulus) errors++;
    }
    printf("Roundtrip errors: %d\n", errors);

    free(poly);
}
```

### Polynomial Multiplication

```c
#include <lux/gpu.h>
#include <stdlib.h>
#include <string.h>

void poly_mul_example(LuxGPU* gpu) {
    size_t n = 2048;
    uint64_t modulus = 0xFFFFFFFF00000001ULL;

    uint64_t* a = malloc(n * sizeof(uint64_t));
    uint64_t* b = malloc(n * sizeof(uint64_t));
    uint64_t* result = malloc(n * sizeof(uint64_t));

    // Initialize polynomials
    memset(a, 0, n * sizeof(uint64_t));
    memset(b, 0, n * sizeof(uint64_t));
    a[0] = 1;  // Constant polynomial
    b[0] = 1;
    b[1] = 1;  // (1 + x)

    // result = a * b mod (X^n + 1)
    LuxError err = lux_poly_mul(gpu, a, b, result, n, modulus);

    if (err == LUX_OK) {
        printf("Polynomial multiplication complete\n");
        printf("result[0] = %lu, result[1] = %lu\n",
               (unsigned long)result[0], (unsigned long)result[1]);
    }

    free(result);
    free(b);
    free(a);
}
```

---

## Performance Timing

### Using Events

```c
#include <lux/gpu.h>
#include <stdio.h>

void timing_example(LuxGPU* gpu) {
    LuxEvent* start = lux_event_create(gpu);
    LuxEvent* end = lux_event_create(gpu);
    LuxStream* stream = lux_stream_create(gpu);

    int64_t shape[] = {4096, 4096};
    LuxTensor* A = lux_tensor_ones(gpu, shape, 2, LUX_FLOAT32);
    LuxTensor* B = lux_tensor_ones(gpu, shape, 2, LUX_FLOAT32);

    // Record start
    lux_event_record(start, stream);

    // Matrix multiplication
    LuxTensor* C = lux_tensor_matmul(gpu, A, B);

    // Record end
    lux_event_record(end, stream);

    // Wait and get elapsed time
    lux_stream_sync(stream);
    float elapsed_ms = lux_event_elapsed(start, end);

    printf("GEMM [4096x4096] x [4096x4096]: %.2f ms\n", elapsed_ms);

    // Calculate TFLOPS
    double flops = 2.0 * 4096 * 4096 * 4096;
    double tflops = flops / (elapsed_ms * 1e9);
    printf("Performance: %.2f TFLOPS\n", tflops);

    lux_tensor_destroy(C);
    lux_tensor_destroy(B);
    lux_tensor_destroy(A);
    lux_stream_destroy(stream);
    lux_event_destroy(end);
    lux_event_destroy(start);
}
```

---

## Error Handling

### Proper Error Checking

```c
#include <lux/gpu.h>
#include <stdio.h>

int robust_example(void) {
    LuxGPU* gpu = lux_gpu_create_with_backend(LUX_BACKEND_CUDA);
    if (!gpu) {
        // Fall back to CPU
        printf("CUDA not available, using CPU\n");
        gpu = lux_gpu_create_with_backend(LUX_BACKEND_CPU);
        if (!gpu) {
            fprintf(stderr, "No GPU backend available\n");
            return 1;
        }
    }

    int64_t shape[] = {1000000};  // 1M elements
    LuxTensor* tensor = lux_tensor_zeros(gpu, shape, 1, LUX_FLOAT32);
    if (!tensor) {
        fprintf(stderr, "Allocation failed: %s\n", lux_gpu_error(gpu));
        lux_gpu_destroy(gpu);
        return 1;
    }

    LuxError err = lux_gpu_sync(gpu);
    if (err != LUX_OK) {
        fprintf(stderr, "Sync failed: %s\n", lux_gpu_error(gpu));
        lux_tensor_destroy(tensor);
        lux_gpu_destroy(gpu);
        return 1;
    }

    lux_tensor_destroy(tensor);
    lux_gpu_destroy(gpu);
    return 0;
}
```

---

## Build Examples

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(lux_gpu_examples)

find_package(luxgpu REQUIRED)

add_executable(hello_gpu hello_gpu.c)
target_link_libraries(hello_gpu lux::luxgpu_core)

add_executable(matmul_bench matmul_bench.c)
target_link_libraries(matmul_bench lux::luxgpu_core)
```

### Compile with pkg-config

```bash
# Compile single file
gcc -o hello_gpu hello_gpu.c $(pkg-config --cflags --libs luxgpu)

# Or with clang
clang -o hello_gpu hello_gpu.c $(pkg-config --cflags --libs luxgpu)
```
