// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Backend Integration Test
//
// Detects all available backends, runs keccak256 batch hash on each,
// verifies all produce identical output, and prints throughput.

#include "lux/gpu.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

// =============================================================================
// Test Framework
// =============================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  %-55s ", name)
#define PASS() do { printf("[PASS]\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(); else FAIL(msg); } while(0)

// =============================================================================
// Reference: CPU Keccak-256
// =============================================================================

static constexpr uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

static inline uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

static void keccak_f(uint64_t st[25]) {
    static constexpr int PI[24] = {
        10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1
    };
    static constexpr int RHO[24] = {
        1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44
    };
    for (int r = 0; r < 24; ++r) {
        uint64_t C[5];
        for (int x = 0; x < 5; ++x)
            C[x] = st[x]^st[x+5]^st[x+10]^st[x+15]^st[x+20];
        for (int x = 0; x < 5; ++x) {
            uint64_t d = C[(x+4)%5]^rotl64(C[(x+1)%5],1);
            for (int y = 0; y < 5; ++y) st[x+5*y]^=d;
        }
        uint64_t t = st[1];
        for (int i = 0; i < 24; ++i) {
            uint64_t tmp = st[PI[i]];
            st[PI[i]] = rotl64(t, RHO[i]);
            t = tmp;
        }
        for (int y = 0; y < 5; ++y) {
            uint64_t row[5];
            for (int x = 0; x < 5; ++x) row[x] = st[x+5*y];
            for (int x = 0; x < 5; ++x)
                st[x+5*y] = row[x]^((~row[(x+1)%5])&row[(x+2)%5]);
        }
        st[0] ^= RC[r];
    }
}

static void ref_keccak256(const uint8_t* data, size_t len, uint8_t out[32]) {
    uint64_t state[25] = {};
    constexpr size_t rate = 136;

    size_t absorbed = 0;
    while (absorbed + rate <= len) {
        for (size_t w = 0; w < rate/8; ++w) {
            uint64_t lane = 0;
            for (size_t b = 0; b < 8; ++b)
                lane |= uint64_t(data[absorbed + w*8 + b]) << (b*8);
            state[w] ^= lane;
        }
        keccak_f(state);
        absorbed += rate;
    }

    uint8_t padded[136] = {};
    size_t rem = len - absorbed;
    std::memcpy(padded, data + absorbed, rem);
    padded[rem] = 0x01;
    padded[rate - 1] |= 0x80;

    for (size_t w = 0; w < rate/8; ++w) {
        uint64_t lane = 0;
        for (size_t b = 0; b < 8; ++b)
            lane |= uint64_t(padded[w*8 + b]) << (b*8);
        state[w] ^= lane;
    }
    keccak_f(state);

    for (size_t w = 0; w < 4; ++w) {
        uint64_t lane = state[w];
        for (size_t b = 0; b < 8; ++b)
            out[w*8 + b] = static_cast<uint8_t>(lane >> (b*8));
    }
}

// =============================================================================
// Hex Printing
// =============================================================================

static void print_hex(const uint8_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) printf("%02x", data[i]);
}

// =============================================================================
// Backend Tester
// =============================================================================

struct BackendResult {
    const char* name;
    bool available;
    bool keccak_correct;
    double keccak_throughput_mhps;  // million hashes per second
};

static BackendResult test_backend(LuxBackend backend, const char* name) {
    BackendResult result = {};
    result.name = name;
    result.available = lux_backend_available(backend);

    if (!result.available) {
        return result;
    }

    LuxGPU* gpu = lux_gpu_create_with_backend(backend);
    if (!gpu) {
        result.available = false;
        return result;
    }

    // -- Correctness: known test vectors --

    // Empty input: keccak256("")
    const uint8_t empty_expected[32] = {
        0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c,
        0x92, 0x7e, 0x7d, 0xb2, 0xdc, 0xc7, 0x03, 0xc0,
        0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b,
        0x7b, 0xfa, 0xd8, 0x04, 0x5d, 0x85, 0xa4, 0x70,
    };

    // "abc"
    const uint8_t abc[] = {'a', 'b', 'c'};
    uint8_t abc_expected[32];
    ref_keccak256(abc, 3, abc_expected);

    // "hello world"
    const uint8_t hello[] = {'h','e','l','l','o',' ','w','o','r','l','d'};
    uint8_t hello_expected[32];
    ref_keccak256(hello, 11, hello_expected);

    // Batch: empty + "abc" + "hello world"
    uint8_t concat_data[14];  // 0 + 3 + 11
    std::memcpy(concat_data, abc, 3);
    std::memcpy(concat_data + 3, hello, 11);

    size_t lengths[3] = {0, 3, 11};
    uint8_t gpu_outputs[3 * 32];

    // The API takes concatenated data; empty input contributes 0 bytes
    LuxError err = lux_gpu_keccak256_batch(gpu, concat_data, gpu_outputs, lengths, 3);

    result.keccak_correct = true;
    if (err != LUX_OK) {
        result.keccak_correct = false;
    } else {
        if (std::memcmp(gpu_outputs, empty_expected, 32) != 0) {
            printf("\n    empty hash mismatch: got ");
            print_hex(gpu_outputs, 32);
            printf("\n    expected ");
            print_hex(empty_expected, 32);
            printf("\n");
            result.keccak_correct = false;
        }
        if (std::memcmp(gpu_outputs + 32, abc_expected, 32) != 0) {
            printf("\n    abc hash mismatch\n");
            result.keccak_correct = false;
        }
        if (std::memcmp(gpu_outputs + 64, hello_expected, 32) != 0) {
            printf("\n    hello hash mismatch\n");
            result.keccak_correct = false;
        }
    }

    // -- Throughput: batch of 10,000 hashes of 32-byte inputs --

    constexpr size_t BENCH_COUNT = 10000;
    constexpr size_t INPUT_SIZE = 32;

    std::vector<uint8_t> bench_data(BENCH_COUNT * INPUT_SIZE);
    std::vector<size_t> bench_lens(BENCH_COUNT, INPUT_SIZE);
    std::vector<uint8_t> bench_out(BENCH_COUNT * 32);

    // Fill with deterministic pattern
    for (size_t i = 0; i < bench_data.size(); i++) {
        bench_data[i] = static_cast<uint8_t>((i * 7 + 13) & 0xFF);
    }

    // Warm up
    lux_gpu_keccak256_batch(gpu, bench_data.data(), bench_out.data(),
                            bench_lens.data(), BENCH_COUNT);

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERATIONS = 5;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        lux_gpu_keccak256_batch(gpu, bench_data.data(), bench_out.data(),
                                bench_lens.data(), BENCH_COUNT);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double total_hashes = static_cast<double>(BENCH_COUNT) * ITERATIONS;
    result.keccak_throughput_mhps = total_hashes / (elapsed_ms * 1000.0);

    lux_gpu_destroy(gpu);
    return result;
}

// =============================================================================
// Ecrecover Test
// =============================================================================

static void test_ecrecover_basic() {
    printf("\n=== Ecrecover Tests ===\n");

    LuxGPU* gpu = lux_gpu_create();
    if (!gpu) {
        FAIL("GPU create failed");
        return;
    }

    // Test with a single dummy signature (won't produce valid recovery,
    // but tests the dispatch path and error handling)
    LuxEcrecoverInput sig = {};
    sig.r[0] = 0x01;
    sig.s[0] = 0x01;
    sig.v = 0;
    sig.msg_hash[0] = 0xAA;

    LuxEcrecoverOutput addr = {};

    TEST("ecrecover batch dispatch");
    LuxError err = lux_gpu_ecrecover_batch(gpu, &sig, &addr, 1);
    CHECK(err == LUX_OK, "ecrecover batch should return LUX_OK");

    TEST("ecrecover empty batch");
    err = lux_gpu_ecrecover_batch(gpu, &sig, &addr, 0);
    CHECK(err == LUX_OK, "empty batch should return LUX_OK");

    lux_gpu_destroy(gpu);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("=== Lux GPU Backend Integration Test ===\n\n");

    // Detect backends
    printf("--- Available Backends ---\n");

    struct { LuxBackend id; const char* name; } backends[] = {
        {LUX_BACKEND_CPU,   "cpu"},
        {LUX_BACKEND_METAL, "metal"},
        {LUX_BACKEND_CUDA,  "cuda"},
        {LUX_BACKEND_DAWN,  "webgpu"},
    };

    for (const auto& b : backends) {
        bool avail = lux_backend_available(b.id);
        printf("  %-10s %s\n", b.name, avail ? "YES" : "no");
    }

    // Test each backend
    printf("\n--- Keccak-256 Correctness & Throughput ---\n");

    std::vector<BackendResult> results;
    uint8_t reference_output[10000 * 32] = {};  // For cross-backend comparison
    bool have_reference = false;

    for (const auto& b : backends) {
        BackendResult r = test_backend(b.id, b.name);

        char test_name[128];
        snprintf(test_name, sizeof(test_name), "%s: keccak256 available", r.name);
        TEST(test_name);
        if (r.available) {
            PASS();
        } else {
            printf("[SKIP] not available\n");
            tests_passed++;
            continue;
        }

        snprintf(test_name, sizeof(test_name), "%s: keccak256 correctness", r.name);
        TEST(test_name);
        CHECK(r.keccak_correct, "hash mismatch vs reference");

        printf("  %-10s throughput: %.2f M hashes/sec\n", r.name, r.keccak_throughput_mhps);

        // Cross-backend comparison: first backend sets the reference
        if (r.keccak_correct) {
            constexpr size_t CHECK_COUNT = 100;
            constexpr size_t CHECK_SIZE = 32;
            std::vector<uint8_t> check_data(CHECK_COUNT * CHECK_SIZE);
            std::vector<size_t> check_lens(CHECK_COUNT, CHECK_SIZE);
            std::vector<uint8_t> check_out(CHECK_COUNT * 32);

            for (size_t i = 0; i < check_data.size(); i++)
                check_data[i] = static_cast<uint8_t>((i * 31 + 17) & 0xFF);

            LuxGPU* gpu = lux_gpu_create_with_backend(b.id);
            if (gpu) {
                lux_gpu_keccak256_batch(gpu, check_data.data(), check_out.data(),
                                        check_lens.data(), CHECK_COUNT);

                if (!have_reference) {
                    std::memcpy(reference_output, check_out.data(), CHECK_COUNT * 32);
                    have_reference = true;
                } else {
                    snprintf(test_name, sizeof(test_name),
                             "%s: keccak256 matches other backends", r.name);
                    TEST(test_name);
                    CHECK(std::memcmp(reference_output, check_out.data(),
                                      CHECK_COUNT * 32) == 0,
                          "output differs from reference backend");
                }

                lux_gpu_destroy(gpu);
            }
        }

        results.push_back(r);
    }

    // Ecrecover tests
    test_ecrecover_basic();

    // Summary
    printf("\n=== Results ===\n");
    printf("  Passed: %d\n", tests_passed);
    printf("  Failed: %d\n", tests_failed);

    if (!results.empty()) {
        printf("\n--- Throughput Summary ---\n");
        for (const auto& r : results) {
            if (r.available) {
                printf("  %-10s %8.2f M hashes/sec  %s\n",
                       r.name, r.keccak_throughput_mhps,
                       r.keccak_correct ? "OK" : "FAIL");
            }
        }
    }

    return tests_failed > 0 ? 1 : 0;
}
