// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
/// @file bench_all_crypto.cpp
/// Benchmark ALL crypto operations: CPU vs Metal GPU.
/// For each operation, runs 10k invocations on CPU then on GPU and prints
/// a comparison table.
///
/// Compile (macOS):
///   clang++ -std=c++20 -O2 bench_all_crypto.cpp \
///       -I../include -L../lib -llux-gpu \
///       -framework Metal -framework Foundation \
///       -o bench_all_crypto

#include "lux/gpu/backend_plugin.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// Forward declaration of backend init functions.
extern "C" bool lux_gpu_backend_init(lux_gpu_backend_desc* out);
extern "C" bool cpu_backend_init(lux_gpu_backend_desc* out);

// =============================================================================
// Timing helpers
// =============================================================================

static double now_ms()
{
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               Clock::now().time_since_epoch())
        .count();
}

struct BenchResult
{
    const char* name;
    size_t count;
    double cpu_ms;
    double gpu_ms;
    bool cpu_ok;
    bool gpu_ok;
};

static void print_table(const std::vector<BenchResult>& results)
{
    printf("\n");
    printf("%-30s  %8s  %10s  %10s  %10s  %s\n",
           "Operation", "Count", "CPU (ms)", "GPU (ms)", "Speedup", "Status");
    printf("%-30s  %8s  %10s  %10s  %10s  %s\n",
           "------------------------------", "--------", "----------",
           "----------", "----------", "------");

    for (const auto& r : results)
    {
        if (!r.gpu_ok)
        {
            printf("%-30s  %8zu  %10.2f  %10s  %10s  %s\n",
                   r.name, r.count, r.cpu_ms, "N/A", "N/A",
                   r.cpu_ok ? "GPU not wired" : "BOTH FAIL");
            continue;
        }

        double speedup = (r.gpu_ms > 0.001) ? (r.cpu_ms / r.gpu_ms) : 0.0;
        printf("%-30s  %8zu  %10.2f  %10.2f  %9.2fx  %s\n",
               r.name, r.count, r.cpu_ms, r.gpu_ms, speedup,
               (r.cpu_ok && r.gpu_ok) ? "OK" : "FAIL");
    }
    printf("\n");
}

// =============================================================================
// Random data generation
// =============================================================================

static std::mt19937 rng(42);

static void fill_random(void* buf, size_t len)
{
    auto* p = static_cast<uint8_t*>(buf);
    for (size_t i = 0; i < len; i++)
        p[i] = static_cast<uint8_t>(rng() & 0xFF);
}

// =============================================================================
// Benchmark runner
// =============================================================================

/// Benchmark a hash operation (keccak256 or blake3).
static BenchResult bench_hash(
    const char* name,
    LuxBackendContext* cpu_ctx, const lux_gpu_backend_vtbl* cpu_vtbl,
    LuxBackendContext* gpu_ctx, const lux_gpu_backend_vtbl* gpu_vtbl,
    LuxBackendError (*cpu_fn)(LuxBackendContext*, const uint8_t*, uint8_t*, const size_t*, size_t),
    LuxBackendError (*gpu_fn)(LuxBackendContext*, const uint8_t*, uint8_t*, const size_t*, size_t),
    size_t count)
{
    BenchResult r = {name, count, 0, 0, false, false};

    // Generate random inputs (32 bytes each for simplicity)
    const size_t input_len = 32;
    std::vector<uint8_t> inputs(count * input_len);
    fill_random(inputs.data(), inputs.size());

    std::vector<size_t> lens(count, input_len);
    std::vector<uint8_t> cpu_out(count * 32, 0);
    std::vector<uint8_t> gpu_out(count * 32, 0);

    // CPU
    double t0 = now_ms();
    LuxBackendError err = cpu_fn(cpu_ctx, inputs.data(), cpu_out.data(), lens.data(), count);
    r.cpu_ms = now_ms() - t0;
    r.cpu_ok = (err == LUX_BACKEND_OK);

    // GPU
    if (!gpu_fn || !gpu_ctx)
    {
        r.gpu_ok = false;
        return r;
    }

    t0 = now_ms();
    err = gpu_fn(gpu_ctx, inputs.data(), gpu_out.data(), lens.data(), count);
    r.gpu_ms = now_ms() - t0;
    r.gpu_ok = (err == LUX_BACKEND_OK);

    return r;
}

/// Benchmark a verify-style operation (ed25519, sr25519, mldsa, slhdsa, frost).
/// These take (pubkeys, messages, signatures, results, count).
static BenchResult bench_verify(
    const char* name,
    LuxBackendContext* cpu_ctx,
    LuxBackendContext* gpu_ctx,
    LuxBackendError (*cpu_fn)(LuxBackendContext*, const void*, const void*, const void*, uint32_t*, size_t),
    LuxBackendError (*gpu_fn)(LuxBackendContext*, const void*, const void*, const void*, uint32_t*, size_t),
    size_t pk_size, size_t msg_size, size_t sig_size,
    size_t count)
{
    BenchResult r = {name, count, 0, 0, false, false};

    std::vector<uint8_t> pubkeys(count * pk_size);
    std::vector<uint8_t> messages(count * msg_size);
    std::vector<uint8_t> signatures(count * sig_size);
    fill_random(pubkeys.data(), pubkeys.size());
    fill_random(messages.data(), messages.size());
    fill_random(signatures.data(), signatures.size());

    std::vector<uint32_t> cpu_results(count, 0);
    std::vector<uint32_t> gpu_results(count, 0);

    // CPU
    double t0 = now_ms();
    LuxBackendError err = cpu_fn(cpu_ctx, pubkeys.data(), messages.data(),
                                 signatures.data(), cpu_results.data(), count);
    r.cpu_ms = now_ms() - t0;
    r.cpu_ok = (err == LUX_BACKEND_OK);

    // GPU
    if (!gpu_fn || !gpu_ctx)
    {
        r.gpu_ok = false;
        return r;
    }

    t0 = now_ms();
    err = gpu_fn(gpu_ctx, pubkeys.data(), messages.data(),
                 signatures.data(), gpu_results.data(), count);
    r.gpu_ms = now_ms() - t0;
    r.gpu_ok = (err == LUX_BACKEND_OK);

    return r;
}

/// Benchmark ecrecover (128 bytes in, 32 bytes out per signature).
static BenchResult bench_ecrecover(
    LuxBackendContext* cpu_ctx, const lux_gpu_backend_vtbl* cpu_vtbl,
    LuxBackendContext* gpu_ctx, const lux_gpu_backend_vtbl* gpu_vtbl,
    size_t count)
{
    BenchResult r = {"ecrecover", count, 0, 0, false, false};

    std::vector<uint8_t> sigs(count * 128);
    fill_random(sigs.data(), sigs.size());

    std::vector<uint8_t> cpu_out(count * 32, 0);
    std::vector<uint8_t> gpu_out(count * 32, 0);

    // CPU
    double t0 = now_ms();
    LuxBackendError err = cpu_vtbl->op_ecrecover_batch(
        cpu_ctx, sigs.data(), cpu_out.data(), count);
    r.cpu_ms = now_ms() - t0;
    r.cpu_ok = (err == LUX_BACKEND_OK);

    // GPU
    if (!gpu_vtbl->op_ecrecover_batch || !gpu_ctx)
    {
        r.gpu_ok = false;
        return r;
    }

    t0 = now_ms();
    err = gpu_vtbl->op_ecrecover_batch(gpu_ctx, sigs.data(), gpu_out.data(), count);
    r.gpu_ms = now_ms() - t0;
    r.gpu_ok = (err == LUX_BACKEND_OK);

    return r;
}

/// Benchmark mlkem decapsulate (sign-style: secret_keys, ciphertexts, shared_secrets, count).
static BenchResult bench_mlkem(
    LuxBackendContext* cpu_ctx, const lux_gpu_backend_vtbl* cpu_vtbl,
    LuxBackendContext* gpu_ctx, const lux_gpu_backend_vtbl* gpu_vtbl,
    size_t count)
{
    BenchResult r = {"mlkem_decapsulate", count, 0, 0, false, false};

    // MLKEMSecretKey = 2400, MLKEMCiphertext = 1088, shared_secret = 32
    std::vector<uint8_t> sks(count * 2400);
    std::vector<uint8_t> cts(count * 1088);
    fill_random(sks.data(), sks.size());
    fill_random(cts.data(), cts.size());

    std::vector<uint8_t> cpu_ss(count * 32, 0);
    std::vector<uint8_t> gpu_ss(count * 32, 0);

    // CPU
    double t0 = now_ms();
    LuxBackendError err = cpu_vtbl->op_mlkem_decapsulate_batch(
        cpu_ctx, sks.data(), cts.data(), cpu_ss.data(), count);
    r.cpu_ms = now_ms() - t0;
    r.cpu_ok = (err == LUX_BACKEND_OK);

    // GPU
    if (!gpu_vtbl->op_mlkem_decapsulate_batch || !gpu_ctx)
    {
        r.gpu_ok = false;
        return r;
    }

    t0 = now_ms();
    err = gpu_vtbl->op_mlkem_decapsulate_batch(
        gpu_ctx, sks.data(), cts.data(), gpu_ss.data(), count);
    r.gpu_ms = now_ms() - t0;
    r.gpu_ok = (err == LUX_BACKEND_OK);

    return r;
}

/// Benchmark ringtail partial sign.
static BenchResult bench_ringtail_sign(
    LuxBackendContext* cpu_ctx, const lux_gpu_backend_vtbl* cpu_vtbl,
    LuxBackendContext* gpu_ctx, const lux_gpu_backend_vtbl* gpu_vtbl,
    size_t count)
{
    BenchResult r = {"ringtail_partial_sign", count, 0, 0, false, false};

    // RingtailShare = 1024, RingtailMessage = 32, RingtailPartialSig = 1024
    std::vector<uint8_t> shares(count * 1024);
    std::vector<uint8_t> messages(count * 32);
    fill_random(shares.data(), shares.size());
    fill_random(messages.data(), messages.size());

    std::vector<uint8_t> cpu_out(count * 1024, 0);
    std::vector<uint8_t> gpu_out(count * 1024, 0);

    double t0 = now_ms();
    LuxBackendError err = cpu_vtbl->op_ringtail_partial_sign_batch(
        cpu_ctx, shares.data(), messages.data(), cpu_out.data(), count);
    r.cpu_ms = now_ms() - t0;
    r.cpu_ok = (err == LUX_BACKEND_OK);

    if (!gpu_vtbl->op_ringtail_partial_sign_batch || !gpu_ctx)
    {
        r.gpu_ok = false;
        return r;
    }

    t0 = now_ms();
    err = gpu_vtbl->op_ringtail_partial_sign_batch(
        gpu_ctx, shares.data(), messages.data(), gpu_out.data(), count);
    r.gpu_ms = now_ms() - t0;
    r.gpu_ok = (err == LUX_BACKEND_OK);

    return r;
}

/// Benchmark cggmp21 partial sign.
static BenchResult bench_cggmp21_sign(
    LuxBackendContext* cpu_ctx, const lux_gpu_backend_vtbl* cpu_vtbl,
    LuxBackendContext* gpu_ctx, const lux_gpu_backend_vtbl* gpu_vtbl,
    size_t count)
{
    BenchResult r = {"cggmp21_partial_sign", count, 0, 0, false, false};

    // CGGMP21Input = 128, r_x = 32, CGGMP21PartialSig = 32
    std::vector<uint8_t> inputs(count * 128);
    std::vector<uint8_t> r_x(32);
    fill_random(inputs.data(), inputs.size());
    fill_random(r_x.data(), r_x.size());

    std::vector<uint8_t> cpu_out(count * 32, 0);
    std::vector<uint8_t> gpu_out(count * 32, 0);

    double t0 = now_ms();
    LuxBackendError err = cpu_vtbl->op_cggmp21_partial_sign_batch(
        cpu_ctx, inputs.data(), r_x.data(), cpu_out.data(), count);
    r.cpu_ms = now_ms() - t0;
    r.cpu_ok = (err == LUX_BACKEND_OK);

    if (!gpu_vtbl->op_cggmp21_partial_sign_batch || !gpu_ctx)
    {
        r.gpu_ok = false;
        return r;
    }

    t0 = now_ms();
    err = gpu_vtbl->op_cggmp21_partial_sign_batch(
        gpu_ctx, inputs.data(), r_x.data(), gpu_out.data(), count);
    r.gpu_ms = now_ms() - t0;
    r.gpu_ok = (err == LUX_BACKEND_OK);

    return r;
}

/// Benchmark FROST partial verify.
static BenchResult bench_frost_verify(
    LuxBackendContext* cpu_ctx, const lux_gpu_backend_vtbl* cpu_vtbl,
    LuxBackendContext* gpu_ctx, const lux_gpu_backend_vtbl* gpu_vtbl,
    size_t count)
{
    BenchResult r = {"frost_partial_verify", count, 0, 0, false, false};

    // FROSTCommitment=66, FROSTPartialSig=32, FROSTPublicKey=33, FROSTChallenge=32
    std::vector<uint8_t> commitments(count * 66);
    std::vector<uint8_t> sigs(count * 32);
    std::vector<uint8_t> pks(count * 33);
    std::vector<uint8_t> challenges(count * 32);
    fill_random(commitments.data(), commitments.size());
    fill_random(sigs.data(), sigs.size());
    fill_random(pks.data(), pks.size());
    fill_random(challenges.data(), challenges.size());

    std::vector<uint32_t> cpu_results(count, 0);
    std::vector<uint32_t> gpu_results(count, 0);

    double t0 = now_ms();
    LuxBackendError err = cpu_vtbl->op_frost_partial_verify_batch(
        cpu_ctx, commitments.data(), sigs.data(), pks.data(),
        challenges.data(), cpu_results.data(), count);
    r.cpu_ms = now_ms() - t0;
    r.cpu_ok = (err == LUX_BACKEND_OK);

    if (!gpu_vtbl->op_frost_partial_verify_batch || !gpu_ctx)
    {
        r.gpu_ok = false;
        return r;
    }

    t0 = now_ms();
    err = gpu_vtbl->op_frost_partial_verify_batch(
        gpu_ctx, commitments.data(), sigs.data(), pks.data(),
        challenges.data(), gpu_results.data(), count);
    r.gpu_ms = now_ms() - t0;
    r.gpu_ok = (err == LUX_BACKEND_OK);

    return r;
}

// =============================================================================
// Main
// =============================================================================

int main()
{
    printf("================================================================\n");
    printf("Lux GPU Crypto Benchmark - ALL Operations\n");
    printf("================================================================\n\n");

    // Initialize backends
    lux_gpu_backend_desc cpu_desc = {};
    lux_gpu_backend_desc gpu_desc = {};

    bool cpu_avail = cpu_backend_init(&cpu_desc);
    bool gpu_avail = lux_gpu_backend_init(&gpu_desc);

    if (!cpu_avail)
    {
        fprintf(stderr, "FATAL: CPU backend init failed\n");
        return 1;
    }

    printf("CPU backend: %s v%s\n", cpu_desc.backend_name, cpu_desc.backend_version);

    if (!gpu_avail)
    {
        printf("GPU backend: not available (Metal not supported)\n");
        printf("Running CPU-only benchmarks.\n\n");
    }
    else
    {
        printf("GPU backend: %s v%s\n", gpu_desc.backend_name, gpu_desc.backend_version);
    }

    const auto* cpu_vtbl = cpu_desc.vtbl;
    const auto* gpu_vtbl = gpu_avail ? gpu_desc.vtbl : nullptr;

    LuxBackendContext* cpu_ctx = cpu_vtbl->create_context(0);
    LuxBackendContext* gpu_ctx = gpu_avail ? gpu_vtbl->create_context(0) : nullptr;

    if (!cpu_ctx)
    {
        fprintf(stderr, "FATAL: CPU context creation failed\n");
        return 1;
    }

    if (gpu_avail && gpu_ctx)
    {
        LuxBackendDeviceInfo info = {};
        gpu_vtbl->get_device_info(gpu_ctx, &info);
        printf("GPU device: %s (%s)\n", info.name, info.vendor);
        printf("GPU memory: %llu MB\n", (unsigned long long)(info.memory_total / (1024 * 1024)));
        printf("Unified memory: %s\n", info.is_unified_memory ? "yes" : "no");
    }

    constexpr size_t COUNT = 10000;
    printf("\nBenchmarking %zu operations per test...\n", COUNT);

    std::vector<BenchResult> results;

    // --- Hash operations ---

    results.push_back(bench_hash("keccak256", cpu_ctx, cpu_vtbl, gpu_ctx, gpu_vtbl,
                                 cpu_vtbl->op_keccak256_hash,
                                 gpu_vtbl ? gpu_vtbl->op_keccak256_hash : nullptr,
                                 COUNT));

    results.push_back(bench_hash("blake3", cpu_ctx, cpu_vtbl, gpu_ctx, gpu_vtbl,
                                 cpu_vtbl->op_blake3_hash,
                                 gpu_vtbl ? gpu_vtbl->op_blake3_hash : nullptr,
                                 COUNT));

    // --- ECDSA ---

    results.push_back(bench_ecrecover(cpu_ctx, cpu_vtbl, gpu_ctx, gpu_vtbl, COUNT));

    // --- Ed25519 / sr25519 ---

    results.push_back(bench_verify("ed25519_verify", cpu_ctx, gpu_ctx,
                                   cpu_vtbl->op_ed25519_verify_batch,
                                   gpu_vtbl ? gpu_vtbl->op_ed25519_verify_batch : nullptr,
                                   32, 64, 64, COUNT));

    results.push_back(bench_verify("sr25519_verify", cpu_ctx, gpu_ctx,
                                   cpu_vtbl->op_sr25519_verify_batch,
                                   gpu_vtbl ? gpu_vtbl->op_sr25519_verify_batch : nullptr,
                                   32, 64, 64, COUNT));

    // --- Post-quantum ---

    results.push_back(bench_verify("mldsa_verify", cpu_ctx, gpu_ctx,
                                   cpu_vtbl->op_mldsa_verify_batch,
                                   gpu_vtbl ? gpu_vtbl->op_mldsa_verify_batch : nullptr,
                                   1952, 64, 3360, COUNT));

    results.push_back(bench_verify("slhdsa_verify", cpu_ctx, gpu_ctx,
                                   cpu_vtbl->op_slhdsa_verify_batch,
                                   gpu_vtbl ? gpu_vtbl->op_slhdsa_verify_batch : nullptr,
                                   32, 32, 17088, COUNT));

    results.push_back(bench_mlkem(cpu_ctx, cpu_vtbl, gpu_ctx, gpu_vtbl, COUNT));

    // --- Threshold signatures ---

    results.push_back(bench_ringtail_sign(cpu_ctx, cpu_vtbl, gpu_ctx, gpu_vtbl, COUNT));

    results.push_back(bench_frost_verify(cpu_ctx, cpu_vtbl, gpu_ctx, gpu_vtbl, COUNT));

    results.push_back(bench_cggmp21_sign(cpu_ctx, cpu_vtbl, gpu_ctx, gpu_vtbl, COUNT));

    // Print results
    print_table(results);

    // Cleanup
    cpu_vtbl->destroy_context(cpu_ctx);
    if (gpu_ctx && gpu_vtbl)
        gpu_vtbl->destroy_context(gpu_ctx);

    return 0;
}
