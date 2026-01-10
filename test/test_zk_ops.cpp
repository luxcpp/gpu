// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Test ZK Operations - Poseidon2, Merkle, Commitment

#include "lux/gpu.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cassert>

// LuxFr256 from zk_ops.cpp
struct LuxFr256 {
    uint64_t limbs[4];
};

// External ZK functions
extern "C" {
    LuxError lux_gpu_poseidon2(LuxGPU* gpu, LuxFr256* out, const LuxFr256* left, const LuxFr256* right, size_t n);
    LuxError lux_gpu_merkle_root(LuxGPU* gpu, LuxFr256* out, const LuxFr256* leaves, size_t n);
    LuxError lux_gpu_commitment(LuxGPU* gpu, LuxFr256* out, const LuxFr256* values, const LuxFr256* blindings, const LuxFr256* salts, size_t n);
}

// Helper to print Fr256
void print_fr256(const char* name, const LuxFr256& v) {
    printf("%s: [0x%016llx, 0x%016llx, 0x%016llx, 0x%016llx]\n",
           name,
           (unsigned long long)v.limbs[0],
           (unsigned long long)v.limbs[1],
           (unsigned long long)v.limbs[2],
           (unsigned long long)v.limbs[3]);
}

// Check Fr256 is not all zeros (basic sanity)
bool is_nonzero(const LuxFr256& v) {
    return v.limbs[0] || v.limbs[1] || v.limbs[2] || v.limbs[3];
}

// Check two Fr256 are equal
bool fr_equal(const LuxFr256& a, const LuxFr256& b) {
    return memcmp(&a, &b, sizeof(LuxFr256)) == 0;
}

int main() {
    printf("=== ZK Operations Test ===\n\n");

    LuxGPU* gpu = lux_gpu_create();
    if (!gpu) {
        printf("FAIL: Could not create GPU context\n");
        return 1;
    }
    printf("Backend: %s\n\n", lux_gpu_backend_name(gpu));

    int passed = 0;
    int failed = 0;

    // Test 1: Poseidon2 single hash
    {
        printf("Test 1: Poseidon2 single hash\n");
        LuxFr256 left = {{1, 0, 0, 0}};
        LuxFr256 right = {{2, 0, 0, 0}};
        LuxFr256 out = {{0, 0, 0, 0}};

        LuxError err = lux_gpu_poseidon2(gpu, &out, &left, &right, 1);
        if (err == LUX_OK && is_nonzero(out)) {
            print_fr256("  H(1, 2)", out);
            printf("  PASS\n\n");
            passed++;
        } else {
            printf("  FAIL: err=%d, nonzero=%d\n\n", err, is_nonzero(out));
            failed++;
        }
    }

    // Test 2: Poseidon2 determinism
    {
        printf("Test 2: Poseidon2 determinism\n");
        LuxFr256 left = {{0x123456789abcdef0ULL, 0xfedcba9876543210ULL, 0, 0}};
        LuxFr256 right = {{0x0fedcba987654321ULL, 0x123456789abcdef0ULL, 0, 0}};
        LuxFr256 out1, out2;

        lux_gpu_poseidon2(gpu, &out1, &left, &right, 1);
        lux_gpu_poseidon2(gpu, &out2, &left, &right, 1);

        if (fr_equal(out1, out2)) {
            printf("  Hash is deterministic\n");
            printf("  PASS\n\n");
            passed++;
        } else {
            printf("  FAIL: hash not deterministic\n");
            print_fr256("  out1", out1);
            print_fr256("  out2", out2);
            printf("\n");
            failed++;
        }
    }

    // Test 3: Poseidon2 batch
    {
        printf("Test 3: Poseidon2 batch (4 hashes)\n");
        std::vector<LuxFr256> left(4), right(4), out(4);
        for (int i = 0; i < 4; i++) {
            left[i] = {{(uint64_t)i, 0, 0, 0}};
            right[i] = {{(uint64_t)(i + 10), 0, 0, 0}};
        }

        LuxError err = lux_gpu_poseidon2(gpu, out.data(), left.data(), right.data(), 4);
        bool all_nonzero = true;
        bool all_different = true;
        for (int i = 0; i < 4; i++) {
            if (!is_nonzero(out[i])) all_nonzero = false;
            for (int j = i + 1; j < 4; j++) {
                if (fr_equal(out[i], out[j])) all_different = false;
            }
        }

        if (err == LUX_OK && all_nonzero && all_different) {
            printf("  All 4 hashes computed, all different\n");
            printf("  PASS\n\n");
            passed++;
        } else {
            printf("  FAIL: err=%d, all_nonzero=%d, all_different=%d\n\n",
                   err, all_nonzero, all_different);
            failed++;
        }
    }

    // Test 4: Merkle root single leaf
    {
        printf("Test 4: Merkle root single leaf\n");
        LuxFr256 leaf = {{42, 0, 0, 0}};
        LuxFr256 root;

        LuxError err = lux_gpu_merkle_root(gpu, &root, &leaf, 1);
        if (err == LUX_OK && fr_equal(root, leaf)) {
            printf("  Root of single leaf == leaf\n");
            printf("  PASS\n\n");
            passed++;
        } else {
            printf("  FAIL: single leaf root mismatch\n\n");
            failed++;
        }
    }

    // Test 5: Merkle root two leaves
    {
        printf("Test 5: Merkle root two leaves\n");
        LuxFr256 leaves[2] = {{{1, 0, 0, 0}}, {{2, 0, 0, 0}}};
        LuxFr256 root;
        LuxFr256 expected_root;

        // Compute expected: H(leaf[0], leaf[1])
        lux_gpu_poseidon2(gpu, &expected_root, &leaves[0], &leaves[1], 1);

        LuxError err = lux_gpu_merkle_root(gpu, &root, leaves, 2);
        if (err == LUX_OK && fr_equal(root, expected_root)) {
            printf("  Root matches H(leaf0, leaf1)\n");
            printf("  PASS\n\n");
            passed++;
        } else {
            printf("  FAIL: root mismatch\n");
            print_fr256("  expected", expected_root);
            print_fr256("  got", root);
            printf("\n");
            failed++;
        }
    }

    // Test 6: Merkle root four leaves
    {
        printf("Test 6: Merkle root four leaves\n");
        LuxFr256 leaves[4] = {{{1, 0, 0, 0}}, {{2, 0, 0, 0}}, {{3, 0, 0, 0}}, {{4, 0, 0, 0}}};
        LuxFr256 root;

        // Manual tree: H(H(1,2), H(3,4))
        LuxFr256 h01, h23, expected;
        lux_gpu_poseidon2(gpu, &h01, &leaves[0], &leaves[1], 1);
        lux_gpu_poseidon2(gpu, &h23, &leaves[2], &leaves[3], 1);
        lux_gpu_poseidon2(gpu, &expected, &h01, &h23, 1);

        LuxError err = lux_gpu_merkle_root(gpu, &root, leaves, 4);
        if (err == LUX_OK && fr_equal(root, expected)) {
            printf("  Root matches manual tree computation\n");
            printf("  PASS\n\n");
            passed++;
        } else {
            printf("  FAIL: root mismatch\n");
            print_fr256("  expected", expected);
            print_fr256("  got", root);
            printf("\n");
            failed++;
        }
    }

    // Test 7: Commitment
    {
        printf("Test 7: Commitment\n");
        LuxFr256 value = {{100, 0, 0, 0}};
        LuxFr256 blinding = {{200, 0, 0, 0}};
        LuxFr256 salt = {{300, 0, 0, 0}};
        LuxFr256 commitment;

        // Expected: H(H(value, blinding), salt)
        LuxFr256 inner, expected;
        lux_gpu_poseidon2(gpu, &inner, &value, &blinding, 1);
        lux_gpu_poseidon2(gpu, &expected, &inner, &salt, 1);

        LuxError err = lux_gpu_commitment(gpu, &commitment, &value, &blinding, &salt, 1);
        if (err == LUX_OK && fr_equal(commitment, expected)) {
            printf("  Commitment matches H(H(v,b),s)\n");
            printf("  PASS\n\n");
            passed++;
        } else {
            printf("  FAIL: commitment mismatch\n");
            print_fr256("  expected", expected);
            print_fr256("  got", commitment);
            printf("\n");
            failed++;
        }
    }

    // Test 8: Batch commitment
    {
        printf("Test 8: Batch commitment (10 items)\n");
        const int N = 10;
        std::vector<LuxFr256> values(N), blindings(N), salts(N), commitments(N);
        for (int i = 0; i < N; i++) {
            values[i] = {{(uint64_t)i, 0, 0, 0}};
            blindings[i] = {{(uint64_t)(i + 100), 0, 0, 0}};
            salts[i] = {{(uint64_t)(i + 200), 0, 0, 0}};
        }

        LuxError err = lux_gpu_commitment(gpu, commitments.data(), values.data(),
                                          blindings.data(), salts.data(), N);
        bool all_nonzero = true;
        for (int i = 0; i < N; i++) {
            if (!is_nonzero(commitments[i])) all_nonzero = false;
        }

        if (err == LUX_OK && all_nonzero) {
            printf("  All 10 commitments computed\n");
            printf("  PASS\n\n");
            passed++;
        } else {
            printf("  FAIL: err=%d, all_nonzero=%d\n\n", err, all_nonzero);
            failed++;
        }
    }

    // Cleanup
    lux_gpu_destroy(gpu);

    // Summary
    printf("=== Summary ===\n");
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", failed);

    return failed > 0 ? 1 : 0;
}
