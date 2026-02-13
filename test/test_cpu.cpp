// Simple test for lux-gpu CPU backend
#include "lux/gpu.h"
#include <cstdio>
#include <cmath>

int main() {
    printf("=== Lux GPU Test (CPU Backend) ===\n\n");

    // Create GPU context (will auto-detect Metal on macOS)
    LuxGPU* gpu = lux_gpu_create();
    if (!gpu) {
        printf("FAIL: Could not create GPU context\n");
        return 1;
    }

    printf("Backend: %s\n", lux_gpu_backend_name(gpu));

    // Get device info
    LuxDeviceInfo info;
    if (lux_gpu_device_info(gpu, &info) == LUX_OK) {
        printf("Device: %s (%s)\n", info.name, info.vendor);
        printf("Compute units: %d\n", info.compute_units);
    }

    // Test tensor operations
    printf("\n--- Tensor Operations ---\n");

    int64_t shape[] = {4};
    LuxTensor* a = lux_tensor_ones(gpu, shape, 1, LUX_FLOAT32);
    LuxTensor* b = lux_tensor_full(gpu, shape, 1, LUX_FLOAT32, 2.0);

    // Add: 1 + 2 = 3
    LuxTensor* c = lux_tensor_add(gpu, a, b);

    float result[4];
    lux_tensor_to_host(c, result, sizeof(result));
    printf("Add (1+2): [%.1f, %.1f, %.1f, %.1f]\n", result[0], result[1], result[2], result[3]);

    bool pass = (result[0] == 3.0f && result[1] == 3.0f && result[2] == 3.0f && result[3] == 3.0f);
    printf("Add test: %s\n", pass ? "PASS" : "FAIL");

    // Multiply
    LuxTensor* d = lux_tensor_mul(gpu, a, b);
    lux_tensor_to_host(d, result, sizeof(result));
    printf("Mul (1*2): [%.1f, %.1f, %.1f, %.1f]\n", result[0], result[1], result[2], result[3]);

    pass = (result[0] == 2.0f);
    printf("Mul test: %s\n", pass ? "PASS" : "FAIL");

    // Matmul
    printf("\n--- Matrix Multiplication ---\n");
    int64_t mat_shape[] = {2, 2};
    float mat_a[] = {1, 2, 3, 4};
    float mat_b[] = {5, 6, 7, 8};
    LuxTensor* A = lux_tensor_from_data(gpu, mat_a, mat_shape, 2, LUX_FLOAT32);
    LuxTensor* B = lux_tensor_from_data(gpu, mat_b, mat_shape, 2, LUX_FLOAT32);
    LuxTensor* C = lux_tensor_matmul(gpu, A, B);

    float mat_result[4];
    lux_tensor_to_host(C, mat_result, sizeof(mat_result));
    // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    printf("Matmul result: [[%.0f, %.0f], [%.0f, %.0f]]\n",
           mat_result[0], mat_result[1], mat_result[2], mat_result[3]);
    pass = (mat_result[0] == 19.0f && mat_result[1] == 22.0f && mat_result[2] == 43.0f && mat_result[3] == 50.0f);
    printf("Matmul test: %s\n", pass ? "PASS" : "FAIL");

    // NTT test
    printf("\n--- NTT Operations ---\n");
    uint64_t modulus = 0xFFFFFFFF00000001ULL;  // Goldilocks prime
    uint64_t ntt_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    LuxError err = lux_ntt_forward(gpu, ntt_data, 8, modulus);
    printf("NTT forward: %s\n", err == LUX_OK ? "OK" : "FAIL");

    err = lux_ntt_inverse(gpu, ntt_data, 8, modulus);
    printf("NTT inverse: %s\n", err == LUX_OK ? "OK" : "FAIL");

    // Check roundtrip
    printf("NTT result: [%llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu]\n",
           ntt_data[0], ntt_data[1], ntt_data[2], ntt_data[3],
           ntt_data[4], ntt_data[5], ntt_data[6], ntt_data[7]);
    pass = (ntt_data[0] == 1 && ntt_data[7] == 8);
    printf("NTT roundtrip: %s\n", pass ? "PASS" : "FAIL");

    // Cleanup
    lux_tensor_destroy(a);
    lux_tensor_destroy(b);
    lux_tensor_destroy(c);
    lux_tensor_destroy(d);
    lux_tensor_destroy(A);
    lux_tensor_destroy(B);
    lux_tensor_destroy(C);
    lux_gpu_destroy(gpu);

    printf("\n=== Tests Complete ===\n");

    // Fixed: Return failure if any test failed
    int failures = 0;
    if (result[0] != 3.0f || result[1] != 3.0f || result[2] != 3.0f || result[3] != 3.0f) failures++;
    if (mat_result[0] != 19.0f || mat_result[1] != 22.0f || mat_result[2] != 43.0f || mat_result[3] != 50.0f) failures++;
    // NTT roundtrip: check ALL elements, not just first and last
    bool ntt_ok = true;
    uint64_t expected_ntt[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (int i = 0; i < 8; i++) {
        if (ntt_data[i] != expected_ntt[i]) ntt_ok = false;
    }
    if (!ntt_ok) failures++;

    return failures > 0 ? 1 : 0;
}
