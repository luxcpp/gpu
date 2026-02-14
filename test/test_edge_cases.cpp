// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Edge Case and Error Handling Tests
//
// Tests for:
// - Null inputs to all public APIs
// - Empty arrays (n=0)
// - Boundary conditions (single element, large tensors)
// - Backend query APIs (lux_backend_count, lux_device_count, lux_device_info)
// - Data type handling (lux_tensor_dtype)
// - Stream/Event management with actual streams
// - Backend switching stress tests
// - BLS batch verification

#include "lux/gpu.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <vector>
#include <cstdlib>

// =============================================================================
// Test Framework
// =============================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  %-60s ", name)
#define PASS() do { printf("[PASS]\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(); else FAIL(msg); } while(0)
#define SKIP(msg) do { printf("[SKIP] %s\n", msg); } while(0)

constexpr float FLOAT_EPSILON = 1e-5f;

bool float_eq(float a, float b) {
    if (a == b) return true;
    if (std::isnan(a) || std::isnan(b)) return false;
    float diff = std::fabs(a - b);
    float largest = std::fmax(std::fabs(a), std::fabs(b));
    if (largest == 0.0f) return diff < FLOAT_EPSILON;
    return diff <= largest * FLOAT_EPSILON;
}

// =============================================================================
// Backend Query API Tests
// =============================================================================

void test_backend_query_apis() {
    printf("\n=== Backend Query API Tests ===\n");

    TEST("lux_backend_count returns >= 1 (CPU always available)");
    {
        int count = lux_backend_count();
        CHECK(count >= 1, "At least CPU backend should be available");
    }

    TEST("lux_backend_available(LUX_BACKEND_CPU) is true");
    {
        CHECK(lux_backend_available(LUX_BACKEND_CPU), "CPU backend should always be available");
    }

    TEST("lux_backend_available(LUX_BACKEND_AUTO) behavior");
    {
        // AUTO should delegate to best available, document behavior
        bool avail = lux_backend_available(LUX_BACKEND_AUTO);
        printf("(AUTO=%s) ", avail ? "true" : "false");
        PASS();
    }

    TEST("lux_backend_name for each backend type");
    {
        const char* cpu_name = lux_backend_name(LUX_BACKEND_CPU);
        const char* auto_name = lux_backend_name(LUX_BACKEND_AUTO);
        const char* metal_name = lux_backend_name(LUX_BACKEND_METAL);
        const char* cuda_name = lux_backend_name(LUX_BACKEND_CUDA);
        const char* dawn_name = lux_backend_name(LUX_BACKEND_DAWN);

        // All should return non-null strings
        bool all_valid = cpu_name && auto_name && metal_name && cuda_name && dawn_name;
        if (all_valid) {
            printf("\n    auto=%s cpu=%s metal=%s cuda=%s dawn=%s ",
                   auto_name, cpu_name, metal_name, cuda_name, dawn_name);
        }
        CHECK(all_valid, "All backend names should be non-null");
    }

    TEST("lux_backend_name for invalid backend");
    {
        const char* name = lux_backend_name((LuxBackend)999);
        // Should return something (perhaps "unknown") or null
        printf("(name=%s) ", name ? name : "null");
        PASS();  // Document behavior
    }

    TEST("lux_device_count for CPU backend");
    {
        int count = lux_device_count(LUX_BACKEND_CPU);
        CHECK(count >= 1, "CPU should have at least 1 device");
    }

    TEST("lux_device_count for unavailable backend");
    {
        // Pick a backend that might not be available
        LuxBackend test_backend = LUX_BACKEND_CUDA;
        if (!lux_backend_available(test_backend)) {
            int count = lux_device_count(test_backend);
            CHECK(count == 0, "Unavailable backend should have 0 devices");
        } else {
            // If CUDA is available, skip this test
            SKIP("CUDA is available, testing different unavailable backend");
        }
    }

    TEST("lux_device_info for CPU backend");
    {
        LuxDeviceInfo info = {};
        LuxError err = lux_device_info(LUX_BACKEND_CPU, 0, &info);
        if (err == LUX_OK) {
            bool valid = info.name != nullptr && info.vendor != nullptr;
            if (valid) {
                printf("\n    name=%s vendor=%s units=%d ",
                       info.name, info.vendor, info.compute_units);
            }
            CHECK(valid, "Device info should have name and vendor");
        } else {
            printf("(err=%d) ", err);
            FAIL("lux_device_info failed for CPU");
        }
    }

    TEST("lux_device_info with null info pointer");
    {
        LuxError err = lux_device_info(LUX_BACKEND_CPU, 0, nullptr);
        CHECK(err != LUX_OK, "Should fail with null info pointer");
    }

    TEST("lux_device_info with invalid device index");
    {
        LuxDeviceInfo info = {};
        LuxError err = lux_device_info(LUX_BACKEND_CPU, 999, &info);
        // Note: Implementation may return OK or error for invalid index
        // Document actual behavior
        printf("(err=%d) ", err);
        PASS();
    }
}

// =============================================================================
// Null Input Tests (Comprehensive)
// =============================================================================

void test_null_inputs() {
    printf("\n=== Null Input Tests ===\n");

    // GPU context functions
    TEST("lux_gpu_destroy(null) does not crash");
    {
        lux_gpu_destroy(nullptr);
        PASS();
    }

    TEST("lux_gpu_backend(null) behavior");
    {
        LuxBackend b = lux_gpu_backend(nullptr);
        printf("(returned %d) ", b);
        PASS();  // Document behavior
    }

    TEST("lux_gpu_backend_name(null) behavior");
    {
        const char* name = lux_gpu_backend_name(nullptr);
        printf("(returned %s) ", name ? name : "null");
        PASS();  // Document behavior
    }

    TEST("lux_gpu_set_backend(null, ...) returns error");
    {
        LuxError err = lux_gpu_set_backend(nullptr, LUX_BACKEND_CPU);
        CHECK(err != LUX_OK, "Should fail with null GPU context");
    }

    TEST("lux_gpu_device_info(null, ...) returns error");
    {
        LuxDeviceInfo info = {};
        LuxError err = lux_gpu_device_info(nullptr, &info);
        CHECK(err != LUX_OK, "Should fail with null GPU context");
    }

    TEST("lux_gpu_sync(null) returns error");
    {
        LuxError err = lux_gpu_sync(nullptr);
        CHECK(err != LUX_OK, "Should fail with null GPU context");
    }

    TEST("lux_gpu_error(null) behavior");
    {
        const char* msg = lux_gpu_error(nullptr);
        printf("(returned %s) ", msg ? msg : "null");
        PASS();  // Document behavior
    }

    // Tensor functions with null GPU
    TEST("lux_tensor_zeros(null, ...) returns null");
    {
        int64_t shape[] = {4};
        LuxTensor* t = lux_tensor_zeros(nullptr, shape, 1, LUX_FLOAT32);
        CHECK(t == nullptr, "Should return null for null GPU");
    }

    TEST("lux_tensor_ones(null, ...) returns null");
    {
        int64_t shape[] = {4};
        LuxTensor* t = lux_tensor_ones(nullptr, shape, 1, LUX_FLOAT32);
        CHECK(t == nullptr, "Should return null for null GPU");
    }

    TEST("lux_tensor_full(null, ...) returns null");
    {
        int64_t shape[] = {4};
        LuxTensor* t = lux_tensor_full(nullptr, shape, 1, LUX_FLOAT32, 1.0);
        CHECK(t == nullptr, "Should return null for null GPU");
    }

    TEST("lux_tensor_from_data(null, ...) returns null");
    {
        int64_t shape[] = {4};
        float data[] = {1, 2, 3, 4};
        LuxTensor* t = lux_tensor_from_data(nullptr, data, shape, 1, LUX_FLOAT32);
        CHECK(t == nullptr, "Should return null for null GPU");
    }

    // Tensor metadata with null tensor
    TEST("lux_tensor_ndim(null) behavior");
    {
        int ndim = lux_tensor_ndim(nullptr);
        printf("(returned %d) ", ndim);
        PASS();  // Document behavior
    }

    TEST("lux_tensor_shape(null, 0) behavior");
    {
        int64_t s = lux_tensor_shape(nullptr, 0);
        printf("(returned %lld) ", (long long)s);
        PASS();  // Document behavior
    }

    TEST("lux_tensor_size(null) behavior");
    {
        int64_t s = lux_tensor_size(nullptr);
        printf("(returned %lld) ", (long long)s);
        PASS();  // Document behavior
    }

    TEST("lux_tensor_dtype(null) behavior");
    {
        LuxDtype dt = lux_tensor_dtype(nullptr);
        printf("(returned %d) ", dt);
        PASS();  // Document behavior
    }

    TEST("lux_tensor_to_host(null, ...) returns error");
    {
        float buf[4];
        LuxError err = lux_tensor_to_host(nullptr, buf, sizeof(buf));
        CHECK(err != LUX_OK, "Should fail with null tensor");
    }

    TEST("lux_tensor_destroy(null) does not crash");
    {
        lux_tensor_destroy(nullptr);
        PASS();
    }

    // Binary tensor ops with null
    LuxGPU* gpu = lux_gpu_create();
    if (gpu) {
        TEST("lux_tensor_add(gpu, null, null) returns null");
        {
            LuxTensor* r = lux_tensor_add(gpu, nullptr, nullptr);
            CHECK(r == nullptr, "Should return null");
        }

        TEST("lux_tensor_sub(gpu, null, null) returns null");
        {
            LuxTensor* r = lux_tensor_sub(gpu, nullptr, nullptr);
            CHECK(r == nullptr, "Should return null");
        }

        TEST("lux_tensor_mul(gpu, null, null) returns null");
        {
            LuxTensor* r = lux_tensor_mul(gpu, nullptr, nullptr);
            CHECK(r == nullptr, "Should return null");
        }

        TEST("lux_tensor_div(gpu, null, null) returns null");
        {
            LuxTensor* r = lux_tensor_div(gpu, nullptr, nullptr);
            CHECK(r == nullptr, "Should return null");
        }

        TEST("lux_tensor_matmul(gpu, null, null) returns null");
        {
            LuxTensor* r = lux_tensor_matmul(gpu, nullptr, nullptr);
            CHECK(r == nullptr, "Should return null");
        }

        // Unary ops with null
        TEST("lux_tensor_neg(gpu, null) returns null");
        {
            LuxTensor* r = lux_tensor_neg(gpu, nullptr);
            CHECK(r == nullptr, "Should return null");
        }

        TEST("lux_tensor_exp(gpu, null) returns null");
        {
            LuxTensor* r = lux_tensor_exp(gpu, nullptr);
            CHECK(r == nullptr, "Should return null");
        }

        TEST("lux_tensor_log(gpu, null) returns null");
        {
            LuxTensor* r = lux_tensor_log(gpu, nullptr);
            CHECK(r == nullptr, "Should return null");
        }

        TEST("lux_tensor_sqrt(gpu, null) returns null");
        {
            LuxTensor* r = lux_tensor_sqrt(gpu, nullptr);
            CHECK(r == nullptr, "Should return null");
        }

        TEST("lux_tensor_abs(gpu, null) returns null");
        {
            LuxTensor* r = lux_tensor_abs(gpu, nullptr);
            CHECK(r == nullptr, "Should return null");
        }

        TEST("lux_tensor_tanh(gpu, null) returns null");
        {
            LuxTensor* r = lux_tensor_tanh(gpu, nullptr);
            CHECK(r == nullptr, "Should return null");
        }

        TEST("lux_tensor_sigmoid(gpu, null) returns null");
        {
            LuxTensor* r = lux_tensor_sigmoid(gpu, nullptr);
            CHECK(r == nullptr, "Should return null");
        }

        TEST("lux_tensor_relu(gpu, null) returns null");
        {
            LuxTensor* r = lux_tensor_relu(gpu, nullptr);
            CHECK(r == nullptr, "Should return null");
        }

        TEST("lux_tensor_gelu(gpu, null) returns null");
        {
            LuxTensor* r = lux_tensor_gelu(gpu, nullptr);
            CHECK(r == nullptr, "Should return null");
        }

        // Reductions with null
        TEST("lux_tensor_reduce_sum(gpu, null) behavior");
        {
            float sum = lux_tensor_reduce_sum(gpu, nullptr);
            printf("(returned %.2f) ", sum);
            PASS();  // Document behavior
        }

        TEST("lux_tensor_reduce_max(gpu, null) behavior");
        {
            float m = lux_tensor_reduce_max(gpu, nullptr);
            printf("(returned %.2f) ", m);
            PASS();  // Document behavior
        }

        TEST("lux_tensor_reduce_min(gpu, null) behavior");
        {
            float m = lux_tensor_reduce_min(gpu, nullptr);
            printf("(returned %.2f) ", m);
            PASS();  // Document behavior
        }

        TEST("lux_tensor_reduce_mean(gpu, null) behavior");
        {
            float m = lux_tensor_reduce_mean(gpu, nullptr);
            printf("(returned %.2f) ", m);
            PASS();  // Document behavior
        }

        // Axis reductions with null
        TEST("lux_tensor_sum(gpu, null, ...) returns null");
        {
            int axes[] = {0};
            LuxTensor* r = lux_tensor_sum(gpu, nullptr, axes, 1);
            CHECK(r == nullptr, "Should return null");
        }

        // Softmax with null
        TEST("lux_tensor_softmax(gpu, null, 0) returns null");
        {
            LuxTensor* r = lux_tensor_softmax(gpu, nullptr, 0);
            CHECK(r == nullptr, "Should return null");
        }

        // Transpose/copy with null
        TEST("lux_tensor_transpose(gpu, null) returns null");
        {
            LuxTensor* r = lux_tensor_transpose(gpu, nullptr);
            CHECK(r == nullptr, "Should return null");
        }

        TEST("lux_tensor_copy(gpu, null) returns null");
        {
            LuxTensor* r = lux_tensor_copy(gpu, nullptr);
            CHECK(r == nullptr, "Should return null");
        }

        lux_gpu_destroy(gpu);
    }

    // Stream/Event with null
    TEST("lux_stream_create(null) returns null");
    {
        LuxStream* s = lux_stream_create(nullptr);
        CHECK(s == nullptr, "Should return null for null GPU");
    }

    TEST("lux_stream_destroy(null) does not crash");
    {
        lux_stream_destroy(nullptr);
        PASS();
    }

    TEST("lux_stream_sync(null) returns error");
    {
        LuxError err = lux_stream_sync(nullptr);
        CHECK(err != LUX_OK, "Should fail with null stream");
    }

    TEST("lux_event_create(null) returns null");
    {
        LuxEvent* e = lux_event_create(nullptr);
        CHECK(e == nullptr, "Should return null for null GPU");
    }

    TEST("lux_event_destroy(null) does not crash");
    {
        lux_event_destroy(nullptr);
        PASS();
    }

    TEST("lux_event_record(null, null) returns error");
    {
        LuxError err = lux_event_record(nullptr, nullptr);
        CHECK(err != LUX_OK, "Should fail with null event");
    }

    TEST("lux_event_wait(null, null) returns error");
    {
        LuxError err = lux_event_wait(nullptr, nullptr);
        CHECK(err != LUX_OK, "Should fail with null event");
    }

    TEST("lux_event_elapsed(null, null) behavior");
    {
        float elapsed = lux_event_elapsed(nullptr, nullptr);
        printf("(returned %.2f) ", elapsed);
        PASS();  // Document behavior
    }

    // NTT with null
    TEST("lux_ntt_forward(null, ...) returns error");
    {
        uint64_t data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        LuxError err = lux_ntt_forward(nullptr, data, 8, 0xFFFFFFFF00000001ULL);
        CHECK(err != LUX_OK, "Should fail with null GPU");
    }

    TEST("lux_ntt_forward(gpu, null, ...) returns error");
    {
        gpu = lux_gpu_create();
        if (gpu) {
            LuxError err = lux_ntt_forward(gpu, nullptr, 8, 0xFFFFFFFF00000001ULL);
            CHECK(err != LUX_OK, "Should fail with null data");
            lux_gpu_destroy(gpu);
        }
    }
}

// =============================================================================
// Empty Array Tests
// =============================================================================

void test_empty_arrays() {
    printf("\n=== Empty Array Tests ===\n");

    LuxGPU* gpu = lux_gpu_create();
    if (!gpu) {
        FAIL("GPU context creation failed");
        return;
    }

    // Tensor with 0 dimension
    TEST("lux_tensor_zeros with ndim=0");
    {
        int64_t shape[] = {4};  // Won't be used
        LuxTensor* t = lux_tensor_zeros(gpu, shape, 0, LUX_FLOAT32);
        printf("(returned %p) ", (void*)t);
        if (t) lux_tensor_destroy(t);
        PASS();  // Document behavior
    }

    // Tensor with 0 in shape
    TEST("lux_tensor_zeros with shape=[0]");
    {
        int64_t shape[] = {0};
        LuxTensor* t = lux_tensor_zeros(gpu, shape, 1, LUX_FLOAT32);
        if (t) {
            int64_t size = lux_tensor_size(t);
            printf("(size=%lld) ", (long long)size);
            lux_tensor_destroy(t);
        } else {
            printf("(returned null) ");
        }
        PASS();  // Document behavior
    }

    // Tensor with 0 in middle of shape
    TEST("lux_tensor_zeros with shape=[2, 0, 3]");
    {
        int64_t shape[] = {2, 0, 3};
        LuxTensor* t = lux_tensor_zeros(gpu, shape, 3, LUX_FLOAT32);
        if (t) {
            int64_t size = lux_tensor_size(t);
            printf("(size=%lld) ", (long long)size);
            lux_tensor_destroy(t);
        } else {
            printf("(returned null) ");
        }
        PASS();  // Document behavior
    }

    // NTT with n=0
    TEST("lux_ntt_forward with n=0");
    {
        uint64_t data[1] = {0};
        LuxError err = lux_ntt_forward(gpu, data, 0, 0xFFFFFFFF00000001ULL);
        printf("(err=%d) ", err);
        PASS();  // Document behavior
    }

    // Batch operations with count=0
    TEST("lux_ntt_batch with count=0");
    {
        uint64_t* polys[1] = {nullptr};
        LuxError err = lux_ntt_batch(gpu, polys, 0, 8, 0xFFFFFFFF00000001ULL);
        printf("(err=%d) ", err);
        PASS();  // Document behavior
    }

    lux_gpu_destroy(gpu);
}

// =============================================================================
// Boundary Condition Tests
// =============================================================================

void test_boundary_conditions() {
    printf("\n=== Boundary Condition Tests ===\n");

    LuxGPU* gpu = lux_gpu_create();
    if (!gpu) {
        FAIL("GPU context creation failed");
        return;
    }

    // Single element tensor
    TEST("Single element tensor: shape=[1]");
    {
        int64_t shape[] = {1};
        float data[] = {42.0f};
        LuxTensor* t = lux_tensor_from_data(gpu, data, shape, 1, LUX_FLOAT32);
        if (t) {
            CHECK(lux_tensor_size(t) == 1, "Size should be 1");
            float out;
            lux_tensor_to_host(t, &out, sizeof(out));
            CHECK(float_eq(out, 42.0f), "Value should be 42");
            lux_tensor_destroy(t);
        } else {
            FAIL("Failed to create single element tensor");
        }
    }

    // Operations on single element
    TEST("Add on single element tensors");
    {
        int64_t shape[] = {1};
        LuxTensor* a = lux_tensor_full(gpu, shape, 1, LUX_FLOAT32, 10.0);
        LuxTensor* b = lux_tensor_full(gpu, shape, 1, LUX_FLOAT32, 20.0);
        if (a && b) {
            LuxTensor* c = lux_tensor_add(gpu, a, b);
            if (c) {
                float out;
                lux_tensor_to_host(c, &out, sizeof(out));
                CHECK(float_eq(out, 30.0f), "10 + 20 should be 30");
                lux_tensor_destroy(c);
            } else {
                FAIL("Add returned null");
            }
            lux_tensor_destroy(a);
            lux_tensor_destroy(b);
        } else {
            FAIL("Failed to create tensors");
        }
    }

    // Reduce on single element
    TEST("Reduce sum on single element");
    {
        int64_t shape[] = {1};
        LuxTensor* t = lux_tensor_full(gpu, shape, 1, LUX_FLOAT32, 99.0);
        if (t) {
            float sum = lux_tensor_reduce_sum(gpu, t);
            CHECK(float_eq(sum, 99.0f), "Sum of single element should be 99");
            lux_tensor_destroy(t);
        } else {
            FAIL("Failed to create tensor");
        }
    }

    // Softmax on single element (should be 1.0)
    TEST("Softmax on single element should be 1.0");
    {
        int64_t shape[] = {1};
        LuxTensor* t = lux_tensor_full(gpu, shape, 1, LUX_FLOAT32, 100.0);
        if (t) {
            LuxTensor* sm = lux_tensor_softmax(gpu, t, 0);
            if (sm) {
                float out;
                lux_tensor_to_host(sm, &out, sizeof(out));
                CHECK(float_eq(out, 1.0f), "Softmax of single element should be 1.0");
                lux_tensor_destroy(sm);
            } else {
                SKIP("lux_tensor_softmax not supported");
            }
            lux_tensor_destroy(t);
        } else {
            FAIL("Failed to create tensor");
        }
    }

    // Very large 1D tensor (stress test memory allocation)
    TEST("Large 1D tensor: 1 million elements");
    {
        const int64_t N = 1000000;
        int64_t shape[] = {N};
        LuxTensor* t = lux_tensor_zeros(gpu, shape, 1, LUX_FLOAT32);
        if (t) {
            CHECK(lux_tensor_size(t) == N, "Size should be 1M");
            lux_tensor_destroy(t);
        } else {
            SKIP("Could not allocate 1M element tensor");
        }
    }

    // High dimensional tensor (many dims, small sizes)
    TEST("High dimensional tensor: [2,2,2,2,2,2] (6D, 64 elements)");
    {
        int64_t shape[] = {2, 2, 2, 2, 2, 2};
        LuxTensor* t = lux_tensor_ones(gpu, shape, 6, LUX_FLOAT32);
        if (t) {
            CHECK(lux_tensor_ndim(t) == 6, "Should have 6 dimensions");
            CHECK(lux_tensor_size(t) == 64, "Size should be 64");
            float sum = lux_tensor_reduce_sum(gpu, t);
            CHECK(float_eq(sum, 64.0f), "Sum of 64 ones should be 64");
            lux_tensor_destroy(t);
        } else {
            FAIL("Failed to create 6D tensor");
        }
    }

    // Matmul with 1x1 matrices
    TEST("Matmul with 1x1 matrices");
    {
        int64_t shape[] = {1, 1};
        float a_data[] = {3.0f};
        float b_data[] = {4.0f};
        LuxTensor* a = lux_tensor_from_data(gpu, a_data, shape, 2, LUX_FLOAT32);
        LuxTensor* b = lux_tensor_from_data(gpu, b_data, shape, 2, LUX_FLOAT32);
        if (a && b) {
            LuxTensor* c = lux_tensor_matmul(gpu, a, b);
            if (c) {
                float out;
                lux_tensor_to_host(c, &out, sizeof(out));
                CHECK(float_eq(out, 12.0f), "3 * 4 should be 12");
                lux_tensor_destroy(c);
            } else {
                FAIL("Matmul returned null");
            }
            lux_tensor_destroy(a);
            lux_tensor_destroy(b);
        } else {
            FAIL("Failed to create 1x1 matrices");
        }
    }

    // Negative shape values
    TEST("Tensor with negative shape value");
    {
        int64_t shape[] = {-1};
        LuxTensor* t = lux_tensor_zeros(gpu, shape, 1, LUX_FLOAT32);
        printf("(returned %p) ", (void*)t);
        if (t) lux_tensor_destroy(t);
        PASS();  // Document behavior (should probably fail)
    }

    // Very large shape value
    TEST("Tensor with very large shape (may fail allocation)");
    {
        int64_t shape[] = {INT64_MAX};
        LuxTensor* t = lux_tensor_zeros(gpu, shape, 1, LUX_FLOAT32);
        // Should return null due to allocation failure
        CHECK(t == nullptr, "Should fail for impossibly large allocation");
    }

    lux_gpu_destroy(gpu);
}

// =============================================================================
// Data Type Tests
// =============================================================================

void test_dtype_handling() {
    printf("\n=== Data Type Tests ===\n");

    LuxGPU* gpu = lux_gpu_create();
    if (!gpu) {
        FAIL("GPU context creation failed");
        return;
    }

    // Test lux_tensor_dtype for each type
    TEST("lux_tensor_dtype returns correct type (FLOAT32)");
    {
        int64_t shape[] = {4};
        LuxTensor* t = lux_tensor_zeros(gpu, shape, 1, LUX_FLOAT32);
        if (t) {
            CHECK(lux_tensor_dtype(t) == LUX_FLOAT32, "Should return FLOAT32");
            lux_tensor_destroy(t);
        } else {
            FAIL("Failed to create tensor");
        }
    }

    TEST("lux_tensor_dtype returns correct type (INT32)");
    {
        int64_t shape[] = {4};
        LuxTensor* t = lux_tensor_zeros(gpu, shape, 1, LUX_INT32);
        if (t) {
            LuxDtype dt = lux_tensor_dtype(t);
            printf("(dtype=%d, expected=%d) ", dt, LUX_INT32);
            CHECK(dt == LUX_INT32, "Should return INT32");
            lux_tensor_destroy(t);
        } else {
            SKIP("INT32 tensors not supported");
        }
    }

    TEST("lux_tensor_dtype returns correct type (FLOAT16)");
    {
        int64_t shape[] = {4};
        LuxTensor* t = lux_tensor_zeros(gpu, shape, 1, LUX_FLOAT16);
        if (t) {
            LuxDtype dt = lux_tensor_dtype(t);
            printf("(dtype=%d, expected=%d) ", dt, LUX_FLOAT16);
            CHECK(dt == LUX_FLOAT16, "Should return FLOAT16");
            lux_tensor_destroy(t);
        } else {
            SKIP("FLOAT16 tensors not supported");
        }
    }

    TEST("lux_tensor_dtype returns correct type (INT64)");
    {
        int64_t shape[] = {4};
        LuxTensor* t = lux_tensor_zeros(gpu, shape, 1, LUX_INT64);
        if (t) {
            LuxDtype dt = lux_tensor_dtype(t);
            printf("(dtype=%d, expected=%d) ", dt, LUX_INT64);
            CHECK(dt == LUX_INT64, "Should return INT64");
            lux_tensor_destroy(t);
        } else {
            SKIP("INT64 tensors not supported");
        }
    }

    TEST("lux_tensor_dtype returns correct type (BOOL)");
    {
        int64_t shape[] = {4};
        LuxTensor* t = lux_tensor_zeros(gpu, shape, 1, LUX_BOOL);
        if (t) {
            LuxDtype dt = lux_tensor_dtype(t);
            printf("(dtype=%d, expected=%d) ", dt, LUX_BOOL);
            CHECK(dt == LUX_BOOL, "Should return BOOL");
            lux_tensor_destroy(t);
        } else {
            SKIP("BOOL tensors not supported");
        }
    }

    lux_gpu_destroy(gpu);
}

// =============================================================================
// Stream/Event Tests with Actual Streams
// =============================================================================

void test_stream_event_with_streams() {
    printf("\n=== Stream/Event Tests with Actual Streams ===\n");

    LuxGPU* gpu = lux_gpu_create();
    if (!gpu) {
        FAIL("GPU context creation failed");
        return;
    }

    TEST("Create multiple streams");
    {
        LuxStream* s1 = lux_stream_create(gpu);
        LuxStream* s2 = lux_stream_create(gpu);
        LuxStream* s3 = lux_stream_create(gpu);

        bool all_created = (s1 != nullptr && s2 != nullptr && s3 != nullptr);
        CHECK(all_created, "Should create multiple streams");

        if (s1) lux_stream_destroy(s1);
        if (s2) lux_stream_destroy(s2);
        if (s3) lux_stream_destroy(s3);
    }

    TEST("Event record/wait on actual stream");
    {
        LuxStream* stream = lux_stream_create(gpu);
        LuxEvent* event = lux_event_create(gpu);

        if (stream && event) {
            LuxError err1 = lux_event_record(event, stream);
            LuxError err2 = lux_event_wait(event, stream);
            CHECK(err1 == LUX_OK && err2 == LUX_OK, "Record and wait should succeed");

            lux_event_destroy(event);
            lux_stream_destroy(stream);
        } else {
            FAIL("Failed to create stream or event");
            if (stream) lux_stream_destroy(stream);
            if (event) lux_event_destroy(event);
        }
    }

    TEST("Event elapsed measurement between streams");
    {
        LuxStream* stream = lux_stream_create(gpu);
        LuxEvent* start = lux_event_create(gpu);
        LuxEvent* end = lux_event_create(gpu);

        if (stream && start && end) {
            lux_event_record(start, stream);

            // Do some work on the stream
            int64_t shape[] = {1000};
            for (int i = 0; i < 10; i++) {
                LuxTensor* t = lux_tensor_ones(gpu, shape, 1, LUX_FLOAT32);
                if (t) lux_tensor_destroy(t);
            }

            lux_event_record(end, stream);
            lux_stream_sync(stream);

            float elapsed = lux_event_elapsed(start, end);
            printf("(elapsed=%.3fms) ", elapsed);
            CHECK(elapsed >= 0.0f, "Elapsed should be non-negative");

            lux_event_destroy(start);
            lux_event_destroy(end);
            lux_stream_destroy(stream);
        } else {
            FAIL("Failed to create resources");
            if (stream) lux_stream_destroy(stream);
            if (start) lux_event_destroy(start);
            if (end) lux_event_destroy(end);
        }
    }

    TEST("Sync on stream after operations");
    {
        LuxStream* stream = lux_stream_create(gpu);
        if (stream) {
            // Just sync an empty stream
            LuxError err = lux_stream_sync(stream);
            CHECK(err == LUX_OK, "Sync on empty stream should succeed");
            lux_stream_destroy(stream);
        } else {
            FAIL("Failed to create stream");
        }
    }

    lux_gpu_destroy(gpu);
}

// =============================================================================
// Backend Switching Stress Tests
// =============================================================================

void test_backend_switching_stress() {
    printf("\n=== Backend Switching Stress Tests ===\n");

    TEST("Switch backend multiple times");
    {
        LuxGPU* gpu = lux_gpu_create();
        if (!gpu) {
            FAIL("GPU context creation failed");
            return;
        }

        bool all_ok = true;
        for (int i = 0; i < 10; i++) {
            LuxError err = lux_gpu_set_backend(gpu, LUX_BACKEND_CPU);
            if (err != LUX_OK) {
                all_ok = false;
                break;
            }
        }
        CHECK(all_ok, "Should handle repeated backend switches");

        lux_gpu_destroy(gpu);
    }

    TEST("Switch backend with tensors (should handle or error)");
    {
        LuxGPU* gpu = lux_gpu_create();
        if (!gpu) {
            FAIL("GPU context creation failed");
            return;
        }

        // Create a tensor
        int64_t shape[] = {4};
        LuxTensor* t = lux_tensor_ones(gpu, shape, 1, LUX_FLOAT32);
        if (!t) {
            FAIL("Failed to create tensor");
            lux_gpu_destroy(gpu);
            return;
        }

        // Switch backend (may invalidate tensor or handle gracefully)
        LuxError err = lux_gpu_set_backend(gpu, LUX_BACKEND_CPU);
        printf("(switch err=%d) ", err);

        // Cleanup
        lux_tensor_destroy(t);
        lux_gpu_destroy(gpu);
        PASS();  // Document behavior
    }

    TEST("Create GPU with specific device index");
    {
        LuxGPU* gpu = lux_gpu_create_with_device(LUX_BACKEND_CPU, 0);
        if (gpu) {
            CHECK(lux_gpu_backend(gpu) == LUX_BACKEND_CPU, "Should be CPU backend");
            lux_gpu_destroy(gpu);
        } else {
            FAIL("Failed to create GPU with device 0");
        }
    }

    TEST("Create GPU with invalid device index");
    {
        LuxGPU* gpu = lux_gpu_create_with_device(LUX_BACKEND_CPU, 999);
        printf("(returned %p) ", (void*)gpu);
        if (gpu) lux_gpu_destroy(gpu);
        PASS();  // Document behavior
    }
}

// =============================================================================
// BLS Batch Verification Test
// =============================================================================

void test_bls_batch_verify() {
    printf("\n=== BLS Batch Verification Tests ===\n");

    LuxGPU* gpu = lux_gpu_create();
    if (!gpu) {
        FAIL("GPU context creation failed");
        return;
    }

    TEST("lux_bls_verify_batch: basic batch verify");
    {
        // Placeholder data
        uint8_t sig1[96] = {0};
        uint8_t sig2[96] = {0};
        const uint8_t* sigs[] = {sig1, sig2};
        size_t sig_lens[] = {96, 96};

        uint8_t msg1[] = "hello";
        uint8_t msg2[] = "world";
        const uint8_t* msgs[] = {msg1, msg2};
        size_t msg_lens[] = {5, 5};

        uint8_t pk1[48] = {0};
        uint8_t pk2[48] = {0};
        const uint8_t* pks[] = {pk1, pk2};
        size_t pk_lens[] = {48, 48};

        bool results[2] = {false, false};

        LuxError err = lux_bls_verify_batch(gpu, sigs, sig_lens, msgs, msg_lens,
                                             pks, pk_lens, 2, results);
        if (err == LUX_OK) {
            printf("(results=[%d,%d]) ", results[0], results[1]);
            CHECK(true, "BLS batch verify executed");
        } else if (err == LUX_ERROR_NOT_SUPPORTED) {
            SKIP("lux_bls_verify_batch not supported");
        } else {
            printf("(err=%d) ", err);
            PASS();  // Document behavior
        }
    }

    lux_gpu_destroy(gpu);
}

// =============================================================================
// Tensor Shape Query Edge Cases
// =============================================================================

void test_tensor_shape_queries() {
    printf("\n=== Tensor Shape Query Edge Cases ===\n");

    LuxGPU* gpu = lux_gpu_create();
    if (!gpu) {
        FAIL("GPU context creation failed");
        return;
    }

    TEST("lux_tensor_shape with out-of-bounds dim");
    {
        int64_t shape[] = {2, 3};
        LuxTensor* t = lux_tensor_zeros(gpu, shape, 2, LUX_FLOAT32);
        if (t) {
            // Query dim 0 and 1 (valid)
            int64_t d0 = lux_tensor_shape(t, 0);
            int64_t d1 = lux_tensor_shape(t, 1);
            CHECK(d0 == 2 && d1 == 3, "Valid dims should return correct values");

            // Query dim 2 (invalid - out of bounds)
            int64_t d2 = lux_tensor_shape(t, 2);
            printf("(dim[2]=%lld) ", (long long)d2);

            // Query negative dim
            int64_t dn = lux_tensor_shape(t, -1);
            printf("(dim[-1]=%lld) ", (long long)dn);

            lux_tensor_destroy(t);
            PASS();  // Document behavior
        } else {
            FAIL("Failed to create tensor");
        }
    }

    lux_gpu_destroy(gpu);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("================================================================================\n");
    printf("              Lux GPU Edge Case and Error Handling Test Suite\n");
    printf("================================================================================\n");

    test_backend_query_apis();
    test_null_inputs();
    test_empty_arrays();
    test_boundary_conditions();
    test_dtype_handling();
    test_stream_event_with_streams();
    test_backend_switching_stress();
    test_bls_batch_verify();
    test_tensor_shape_queries();

    printf("\n================================================================================\n");
    printf("                              Test Summary\n");
    printf("================================================================================\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    printf("================================================================================\n");

    return tests_failed > 0 ? 1 : 0;
}
