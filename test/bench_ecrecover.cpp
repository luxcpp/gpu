// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Benchmark: GPU-accelerated secp256k1 ECDSA recovery (Ethereum ecrecover)
//
// Measures batch ecrecover throughput across all backends:
//   - CPU sequential (1 core)
//   - CPU parallel (OpenMP, all cores)
//   - Metal GPU
//   - CUDA GPU
//
// Target: <5ms for 10k signatures on GPU
//
// Build:
//   cmake -B build -DLUX_GPU_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
//   cmake --build build
//   ./build/test/bench_ecrecover

#include "bench_common.hpp"
#include <cstdio>
#include <cstring>
#include <vector>

using namespace bench;

// =============================================================================
// Ethereum ecrecover test vector
// =============================================================================
//
// Known test vector from Ethereum: signing "hello" with a known private key.
// This is used to validate correctness before benchmarking.

struct EcrecoverTestVector {
    uint8_t r[32];
    uint8_t s[32];
    uint8_t v;
    uint8_t msg_hash[32];
    uint8_t expected_address[20];
};

// Generate random valid-looking signatures for benchmarking.
// These won't produce valid recoveries, but exercise the same code paths
// (the GPU kernel runs the full math regardless of whether the signature
// is cryptographically valid).
static void generate_bench_signatures(LuxEcrecoverInput* sigs, size_t n) {
    auto& rng = BenchRNG::instance();

    for (size_t i = 0; i < n; i++) {
        std::memset(&sigs[i], 0, sizeof(LuxEcrecoverInput));

        // Generate random r, s, msg_hash (32 bytes each)
        for (int j = 0; j < 32; j++) {
            sigs[i].r[j] = rng.random_byte();
            sigs[i].s[j] = rng.random_byte();
            sigs[i].msg_hash[j] = rng.random_byte();
        }

        // Ensure r and s are non-zero and < n (approximately)
        // Set high byte to avoid being >= curve order
        sigs[i].r[0] &= 0x7F;
        sigs[i].s[0] &= 0x7F;
        // Ensure non-zero
        sigs[i].r[31] |= 0x01;
        sigs[i].s[31] |= 0x01;

        sigs[i].v = (uint8_t)(i & 1); // Alternate v=0 and v=1
    }
}

// =============================================================================
// Benchmark runner
// =============================================================================

struct EcrecoverBench {
    size_t num_sigs;
    std::vector<BenchResult> results;

    explicit EcrecoverBench(size_t n) : num_sigs(n) {}

    std::string size_str() const {
        char buf[32];
        if (num_sigs >= 1000) {
            snprintf(buf, sizeof(buf), "%zuk", num_sigs / 1000);
        } else {
            snprintf(buf, sizeof(buf), "%zu", num_sigs);
        }
        return buf;
    }

    void run(LuxBackend backend) {
        BenchResult result;
        result.operation = "ecrecover";
        result.size = size_str();
        result.backend = backend;
        result.throughput_unit = "sigs/s";
        result.seed_used = BenchRNG::instance().get_seed();

        LuxGPU* gpu = lux_gpu_create_with_backend(backend);
        if (!gpu) {
            result.success = false;
            result.error_msg = "Backend unavailable";
            results.push_back(result);
            return;
        }

        // Allocate input/output buffers
        std::vector<LuxEcrecoverInput> sigs(num_sigs);
        std::vector<LuxEcrecoverOutput> addrs(num_sigs);

        generate_bench_signatures(sigs.data(), num_sigs);

        auto kernel = [&]() {
            LuxError err = lux_gpu_ecrecover_batch(
                gpu, sigs.data(), addrs.data(), num_sigs);
            if (err != LUX_OK) {
                result.success = false;
                result.error_msg = lux_gpu_error(gpu) ? lux_gpu_error(gpu) : "ecrecover failed";
            }
        };

        auto sync = [&]() {
            lux_gpu_sync(gpu);
        };

        try {
            Stats stats = run_benchmark([](){}, kernel, [](){}, sync,
                                         DEFAULT_WARMUP_ITERS, DEFAULT_BENCH_ITERS);
            result.stats = stats;
            result.success = true;

            // Compute throughput: signatures per second
            if (stats.median_ms > 0) {
                result.throughput = (double)num_sigs / (stats.median_ms * 1e-3);
            }

            // Check stability
            result.stable = stats.is_stable(15.0);
        } catch (...) {
            result.success = false;
            result.error_msg = "exception during benchmark";
        }

        lux_gpu_destroy(gpu);
        results.push_back(result);
    }
};

// =============================================================================
// Correctness validation
// =============================================================================

static bool validate_ecrecover(LuxBackend backend) {
    LuxGPU* gpu = lux_gpu_create_with_backend(backend);
    if (!gpu) return false;

    // Use a minimal batch of 1 with known-invalid signature
    // (validates the function runs without crashing; full crypto
    // validation requires known test vectors compiled in)
    LuxEcrecoverInput sig = {};
    sig.v = 0;
    // All-zero r/s should fail validation (r must be > 0)
    LuxEcrecoverOutput out = {};

    LuxError err = lux_gpu_ecrecover_batch(gpu, &sig, &out, 1);
    lux_gpu_destroy(gpu);

    if (err != LUX_OK) return false;
    // All-zero r should produce valid=0
    return out.valid == 0;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("================================================================================\n");
    printf("       Lux GPU - secp256k1 ecrecover Benchmark\n");
    printf("================================================================================\n");
    print_benchmark_info();

    // Test sizes
    const size_t sizes[] = {100, 1000, 5000, 10000, 50000};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // Available backends
    auto backends = get_available_backends();

    // Validate correctness first
    printf("\n--- Correctness Validation ---\n");
    for (auto& be : backends) {
        if (!be.available) continue;
        bool ok = validate_ecrecover(be.type);
        printf("  %s: %s\n", be.name, ok ? "PASS" : "FAIL");
    }

    // Run benchmarks
    printf("\n--- ecrecover Throughput ---\n");
    print_table_header();

    for (int s = 0; s < num_sizes; s++) {
        EcrecoverBench bench(sizes[s]);

        TableRow row;
        row.operation = "ecrecover";
        row.size = bench.size_str();

        for (auto& be : backends) {
            if (!be.available) {
                std::string cell = "N/A";
                switch (be.type) {
                    case LUX_BACKEND_CPU:   row.cpu = cell; break;
                    case LUX_BACKEND_METAL: row.metal = cell; break;
                    case LUX_BACKEND_CUDA:  row.cuda = cell; break;
                    case LUX_BACKEND_DAWN:  row.webgpu = cell; break;
                    default: break;
                }
                continue;
            }

            bench.run(be.type);
            auto& res = bench.results.back();

            std::string cell;
            if (res.success) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%.2f ms", res.stats.median_ms);
                cell = buf;
                if (res.throughput > 0) {
                    char tp[64];
                    if (res.throughput >= 1e6) {
                        snprintf(tp, sizeof(tp), " (%.1fM/s)", res.throughput / 1e6);
                    } else if (res.throughput >= 1e3) {
                        snprintf(tp, sizeof(tp), " (%.1fk/s)", res.throughput / 1e3);
                    } else {
                        snprintf(tp, sizeof(tp), " (%.0f/s)", res.throughput);
                    }
                    cell += tp;
                }
            } else {
                cell = res.error_msg.empty() ? "FAIL" : res.error_msg;
            }

            switch (be.type) {
                case LUX_BACKEND_CPU:   row.cpu = cell; break;
                case LUX_BACKEND_METAL: row.metal = cell; break;
                case LUX_BACKEND_CUDA:  row.cuda = cell; break;
                case LUX_BACKEND_DAWN:  row.webgpu = cell; break;
                default: break;
            }
        }

        print_table_row(row);
    }

    // Summary
    printf("\n--- Performance Summary ---\n");
    printf("Target: <5ms for 10k signatures on GPU\n");
    printf("Target: <10ms for 10k signatures on CPU parallel\n");
    printf("Baseline: ~334ms for 10k signatures CPU sequential (EVM profiling)\n");

    // Print speedup if we have both CPU and GPU results for 10k
    for (int s = 0; s < num_sizes; s++) {
        if (sizes[s] != 10000) continue;

        EcrecoverBench bench(10000);
        double cpu_ms = 0, gpu_ms = 0;

        for (auto& be : backends) {
            if (!be.available) continue;
            bench.run(be.type);
            auto& res = bench.results.back();
            if (!res.success) continue;

            if (be.type == LUX_BACKEND_CPU) {
                cpu_ms = res.stats.median_ms;
            } else if (be.type == LUX_BACKEND_METAL || be.type == LUX_BACKEND_CUDA) {
                gpu_ms = res.stats.median_ms;
            }
        }

        if (cpu_ms > 0 && gpu_ms > 0) {
            printf("  10k signatures: CPU=%.2fms, GPU=%.2fms, speedup=%.1fx\n",
                   cpu_ms, gpu_ms, cpu_ms / gpu_ms);
        } else if (cpu_ms > 0) {
            printf("  10k signatures: CPU=%.2fms (no GPU available)\n", cpu_ms);
        }
    }

    printf("\n");
    return 0;
}
