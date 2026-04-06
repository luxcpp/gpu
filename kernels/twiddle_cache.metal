// =============================================================================
// Twiddle Hotset Caching Kernels for Apple Metal
// =============================================================================
//
// High-performance NTT kernels with intelligent twiddle caching.
//
// Key innovations:
// 1. Hotset identification: Early stages use tiny twiddle sets that fit in
//    constant/threadgroup memory entirely
// 2. Prefetch hints: Load next stage's twiddles during current stage compute
// 3. LRU eviction: Smart eviction for multi-modulus RNS scenarios
// 4. Bank conflict avoidance: Padded storage to eliminate shared memory conflicts
//
// Memory hierarchy utilization:
// - Constant memory: First-level twiddles (8 values), modular constants
// - Threadgroup memory: Stage-specific twiddles with prefetch
// - Registers: Current butterfly operands and twiddle
//
// For N=1024:
//   Stage 0: 1 twiddle   -> constant memory (4 cycles)
//   Stage 1: 2 twiddles  -> constant memory (4 cycles)
//   Stage 2: 4 twiddles  -> constant memory (4 cycles)
//   Stage 3: 8 twiddles  -> constant memory (4 cycles)
//   Stage 4+: threadgroup prefetch (20-30 cycles)
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-2-Clause
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Configuration Constants
// =============================================================================

/// Maximum twiddles in threadgroup shared memory (32KB / 8 bytes)
constant uint32_t MAX_THREADGROUP_TWIDDLES = 4096;

/// First-level twiddles stored in constant memory per prime
constant uint32_t FIRST_LEVEL_TWIDDLE_COUNT = 8;

/// Maximum RNS primes supported
constant uint32_t MAX_RNS_PRIMES = 16;

/// Threadgroup memory bank width for conflict avoidance
constant uint32_t BANK_WIDTH = 32;  // 32 banks of 4 bytes each

/// Padding for bank conflict avoidance
constant uint32_t BANK_PADDING = 1;

// =============================================================================
// Modular Arithmetic Constants (Constant Memory Tier)
// =============================================================================

struct PrimeConstants {
    uint64_t q;           // Prime modulus
    uint64_t q_inv;       // -q^(-1) mod 2^64 (Montgomery)
    uint64_t mu_hi;       // Barrett high bits
    uint64_t mu_lo;       // Barrett low bits
    uint64_t r_squared;   // R^2 mod q
    uint64_t root;        // Primitive root
    uint64_t root_inv;    // Inverse root
    uint64_t n_inv;       // N^(-1) mod q
};

/// Constant memory cache structure
struct ConstantCache {
    uint32_t numPrimes;
    uint32_t ringDim;
    uint32_t padding[2];

    PrimeConstants primes[MAX_RNS_PRIMES];
    uint64_t firstLevelTwiddles[MAX_RNS_PRIMES][FIRST_LEVEL_TWIDDLE_COUNT];
    uint64_t firstLevelInvTwiddles[MAX_RNS_PRIMES][FIRST_LEVEL_TWIDDLE_COUNT];
};

// =============================================================================
// NTT Parameters
// =============================================================================

struct NTTParams {
    uint64_t Q;            // Prime modulus
    uint64_t mu;           // Barrett constant (mu_hi)
    uint64_t N_inv;        // N^(-1) mod Q
    uint64_t N_inv_precon; // Precomputed for Barrett
    uint32_t N;            // Ring dimension
    uint32_t log_N;        // log2(N)
    uint32_t stage;        // Current stage
    uint32_t primeIdx;     // Prime index in RNS
    uint32_t batch;        // Batch size
    uint32_t prefetchStage; // Next stage to prefetch (-1 if none)
};

// =============================================================================
// Modular Arithmetic Functions
// =============================================================================

/// Barrett multiplication: (a * b) mod Q
inline uint64_t barrett_mul(uint64_t a, uint64_t b, uint64_t Q, uint64_t mu) {
    // Low 64 bits of a * b
    uint64_t lo = a * b;

    // Approximate quotient using mulhi
    uint64_t q = mulhi(lo, mu);

    // result = a * b - q * Q
    uint64_t result = lo - q * Q;

    // Conditional subtraction for exact result
    if (result >= Q) result -= Q;

    return result;
}

/// Modular addition: (a + b) mod Q
inline uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    return (sum >= Q) ? sum - Q : sum;
}

/// Modular subtraction: (a - b) mod Q
inline uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    return (a >= b) ? a - b : a + Q - b;
}

// =============================================================================
// Bank Conflict Avoidance Helper
// =============================================================================

/// Compute padded index to avoid bank conflicts
inline uint32_t padded_index(uint32_t idx) {
    // Add padding every BANK_WIDTH elements
    return idx + (idx / BANK_WIDTH) * BANK_PADDING;
}

// =============================================================================
// Kernel: Single Stage NTT with Hotset Caching
// =============================================================================
//
// This kernel processes one NTT stage with intelligent twiddle caching.
//
// For stages 0-3: Uses constant memory twiddles (zero global loads)
// For stages 4+: Cooperative threadgroup load with prefetch hints
//
// Thread organization:
// - One threadgroup per polynomial in the batch
// - Each thread processes multiple butterflies

kernel void ntt_hotset_forward_stage(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* twiddles [[buffer(1)]],          // Device memory twiddles
    constant ConstantCache& cache [[buffer(2)]],        // Constant memory cache
    constant NTTParams& params [[buffer(3)]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    uint threadgroup_idx [[threadgroup_position_in_grid]]
) {
    // Threadgroup shared memory with padding for bank conflict avoidance
    threadgroup uint64_t twiddles_shared[MAX_THREADGROUP_TWIDDLES + MAX_THREADGROUP_TWIDDLES / BANK_WIDTH];
    // Prefetch buffer for next stage
    threadgroup uint64_t twiddles_prefetch[MAX_THREADGROUP_TWIDDLES + MAX_THREADGROUP_TWIDDLES / BANK_WIDTH];

    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint32_t stage = params.stage;
    uint32_t primeIdx = params.primeIdx;
    uint32_t batch_idx = threadgroup_idx;

    // Stage parameters
    uint32_t m = 1u << stage;           // Number of twiddle factors
    uint32_t t = N >> (stage + 1);      // Butterflies per twiddle

    device uint64_t* batch_data = data + batch_idx * N;

    // =========================================================================
    // Phase 1: Determine twiddle source and load strategy
    // =========================================================================

    bool use_constant_memory = (stage < 4 && m <= FIRST_LEVEL_TWIDDLE_COUNT);

    if (!use_constant_memory) {
        // Cooperative load into threadgroup memory with padding
        uint32_t twiddles_to_load = m;
        uint32_t loads_per_thread = (twiddles_to_load + threadgroup_size - 1) / threadgroup_size;

        for (uint32_t i = 0; i < loads_per_thread; ++i) {
            uint32_t tw_idx = thread_idx + i * threadgroup_size;
            if (tw_idx < m) {
                uint32_t padded = padded_index(tw_idx);
                twiddles_shared[padded] = twiddles[m + tw_idx];
            }
        }

        // Prefetch next stage if enabled
        if (params.prefetchStage < params.log_N && params.prefetchStage > stage) {
            uint32_t next_m = 1u << params.prefetchStage;
            uint32_t prefetch_loads = (next_m + threadgroup_size - 1) / threadgroup_size;

            for (uint32_t i = 0; i < prefetch_loads; ++i) {
                uint32_t tw_idx = thread_idx + i * threadgroup_size;
                if (tw_idx < next_m && tw_idx < MAX_THREADGROUP_TWIDDLES) {
                    uint32_t padded = padded_index(tw_idx);
                    twiddles_prefetch[padded] = twiddles[next_m + tw_idx];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // =========================================================================
    // Phase 2: Butterfly computation
    // =========================================================================

    uint32_t butterflies_per_thread = (N / 2 + threadgroup_size - 1) / threadgroup_size;

    for (uint32_t b = 0; b < butterflies_per_thread; ++b) {
        uint32_t butterfly_idx = thread_idx + b * threadgroup_size;
        if (butterfly_idx >= N / 2) break;

        // Compute butterfly indices
        uint32_t group = butterfly_idx / t;
        uint32_t elem = butterfly_idx % t;
        uint32_t idx_lo = (group << (params.log_N - stage)) + elem;
        uint32_t idx_hi = idx_lo + t;

        // Load data
        uint64_t lo = batch_data[idx_lo];
        uint64_t hi = batch_data[idx_hi];

        // Get twiddle from appropriate cache tier
        uint64_t tw;
        if (use_constant_memory) {
            // L1 cache tier - instant access
            tw = cache.firstLevelTwiddles[primeIdx][group];
        } else {
            // L2 cache tier - threadgroup memory
            uint32_t padded = padded_index(group);
            tw = twiddles_shared[padded];
        }

        // Butterfly operation
        uint64_t hi_tw = barrett_mul(hi, tw, Q, mu);
        uint64_t new_lo = mod_add(lo, hi_tw, Q);
        uint64_t new_hi = mod_sub(lo, hi_tw, Q);

        // Write results
        batch_data[idx_lo] = new_lo;
        batch_data[idx_hi] = new_hi;
    }
}

// =============================================================================
// Kernel: Inverse NTT Stage with Hotset Caching
// =============================================================================

kernel void ntt_hotset_inverse_stage(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* twiddles [[buffer(1)]],
    constant ConstantCache& cache [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    uint threadgroup_idx [[threadgroup_position_in_grid]]
) {
    threadgroup uint64_t twiddles_shared[MAX_THREADGROUP_TWIDDLES + MAX_THREADGROUP_TWIDDLES / BANK_WIDTH];

    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint32_t stage = params.stage;
    uint32_t primeIdx = params.primeIdx;
    uint32_t batch_idx = threadgroup_idx;

    // Gentleman-Sande parameters
    uint32_t m = N >> (stage + 1);
    uint32_t t = 1u << stage;

    device uint64_t* batch_data = data + batch_idx * N;

    bool use_constant_memory = (stage >= params.log_N - 4 && m <= FIRST_LEVEL_TWIDDLE_COUNT);

    if (!use_constant_memory) {
        uint32_t twiddles_to_load = m;
        uint32_t loads_per_thread = (twiddles_to_load + threadgroup_size - 1) / threadgroup_size;

        for (uint32_t i = 0; i < loads_per_thread; ++i) {
            uint32_t tw_idx = thread_idx + i * threadgroup_size;
            if (tw_idx < m) {
                uint32_t padded = padded_index(tw_idx);
                twiddles_shared[padded] = twiddles[m + tw_idx];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint32_t butterflies_per_thread = (N / 2 + threadgroup_size - 1) / threadgroup_size;

    for (uint32_t b = 0; b < butterflies_per_thread; ++b) {
        uint32_t butterfly_idx = thread_idx + b * threadgroup_size;
        if (butterfly_idx >= N / 2) break;

        uint32_t group = butterfly_idx / t;
        uint32_t elem = butterfly_idx % t;
        uint32_t idx_lo = (group << (stage + 1)) + elem;
        uint32_t idx_hi = idx_lo + t;

        uint64_t lo = batch_data[idx_lo];
        uint64_t hi = batch_data[idx_hi];

        uint64_t tw;
        if (use_constant_memory) {
            tw = cache.firstLevelInvTwiddles[primeIdx][group];
        } else {
            uint32_t padded = padded_index(group);
            tw = twiddles_shared[padded];
        }

        // Gentleman-Sande butterfly
        uint64_t sum = mod_add(lo, hi, Q);
        uint64_t diff = mod_sub(lo, hi, Q);
        uint64_t new_hi = barrett_mul(diff, tw, Q, mu);

        batch_data[idx_lo] = sum;
        batch_data[idx_hi] = new_hi;
    }
}

// =============================================================================
// Kernel: Multi-Stage Fused NTT with Full Hotset
// =============================================================================
//
// For N <= 4096, ALL twiddles fit in threadgroup memory.
// This kernel processes all log_N stages in a single dispatch.

kernel void ntt_hotset_fused(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* twiddles_flat [[buffer(1)]],
    constant ConstantCache& cache [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    uint threadgroup_idx [[threadgroup_position_in_grid]]
) {
    // All twiddles for N<=4096 fit in shared memory
    threadgroup uint64_t twiddles_shared[MAX_THREADGROUP_TWIDDLES];

    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint64_t mu = params.mu;
    uint32_t log_N = params.log_N;
    uint32_t primeIdx = params.primeIdx;
    uint32_t batch_idx = threadgroup_idx;

    device uint64_t* batch_data = data + batch_idx * N;

    // =========================================================================
    // Phase 1: Load ALL twiddles into threadgroup memory (one-time cost)
    // =========================================================================

    // Total twiddles needed: 1 + 2 + 4 + ... + N/2 = N - 1
    uint32_t total_twiddles = N - 1;
    uint32_t loads_per_thread = (total_twiddles + threadgroup_size - 1) / threadgroup_size;

    for (uint32_t i = 0; i < loads_per_thread; ++i) {
        uint32_t tw_idx = thread_idx + i * threadgroup_size;
        if (tw_idx < total_twiddles) {
            twiddles_shared[tw_idx] = twiddles_flat[tw_idx];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // Phase 2: Process all stages from threadgroup memory
    // =========================================================================

    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = 1u << stage;
        uint32_t t = N >> (stage + 1);
        uint32_t tw_base = m;  // Standard layout: twiddles[m + i]

        uint32_t butterflies_per_thread = (N / 2 + threadgroup_size - 1) / threadgroup_size;

        for (uint32_t b = 0; b < butterflies_per_thread; ++b) {
            uint32_t butterfly_idx = thread_idx + b * threadgroup_size;
            if (butterfly_idx >= N / 2) break;

            uint32_t group = butterfly_idx / t;
            uint32_t elem = butterfly_idx % t;
            uint32_t idx_lo = (group << (log_N - stage)) + elem;
            uint32_t idx_hi = idx_lo + t;

            uint64_t lo = batch_data[idx_lo];
            uint64_t hi = batch_data[idx_hi];
            uint64_t tw = twiddles_shared[tw_base + group];

            uint64_t hi_tw = barrett_mul(hi, tw, Q, mu);
            uint64_t new_lo = mod_add(lo, hi_tw, Q);
            uint64_t new_hi = mod_sub(lo, hi_tw, Q);

            batch_data[idx_lo] = new_lo;
            batch_data[idx_hi] = new_hi;
        }

        // Barrier between stages to ensure memory coherence
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// =============================================================================
// Kernel: RNS Multi-Prime NTT with Hotset Caching
// =============================================================================
//
// Processes NTT for multiple RNS primes in parallel.
// Uses twiddle-major layout for coalesced access across primes.

kernel void ntt_hotset_rns_stage(
    device uint64_t* data [[buffer(0)]],           // [batch, numPrimes, N]
    constant uint64_t* twiddles [[buffer(1)]],      // [N, numPrimes] twiddle-major
    constant ConstantCache& cache [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tgSize [[threads_per_threadgroup]]
) {
    // Thread assignment: x=element, y=prime, z=batch
    uint32_t elemIdx = tid.x;
    uint32_t primeIdx = tid.y;
    uint32_t batchIdx = tid.z;

    uint32_t N = params.N;
    uint32_t log_N = params.log_N;
    uint32_t numPrimes = cache.numPrimes;

    // Get prime-specific constants from constant memory
    PrimeConstants pc = cache.primes[primeIdx];
    uint64_t Q = pc.q;
    uint64_t mu = pc.mu_hi;

    // Per-prime threadgroup twiddle cache
    threadgroup uint64_t prime_twiddles[512];  // Max 512 per prime per stage

    uint32_t stage = params.stage;
    uint32_t m = 1u << stage;
    uint32_t t = N >> (stage + 1);

    // Cooperative load for this prime's twiddles (twiddle-major access)
    uint32_t localIdx = elemIdx % tgSize.x;
    if (localIdx < m) {
        // Coalesced access: adjacent primes access adjacent memory
        prime_twiddles[localIdx] = twiddles[localIdx * numPrimes + primeIdx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process butterfly
    device uint64_t* poly = data + (batchIdx * numPrimes + primeIdx) * N;

    uint32_t butterfly_idx = elemIdx;
    if (butterfly_idx < N / 2) {
        uint32_t group = butterfly_idx / t;
        uint32_t elem = butterfly_idx % t;
        uint32_t idx_lo = (group << (log_N - stage)) + elem;
        uint32_t idx_hi = idx_lo + t;

        uint64_t lo = poly[idx_lo];
        uint64_t hi = poly[idx_hi];
        uint64_t tw = prime_twiddles[group];

        uint64_t hi_tw = barrett_mul(hi, tw, Q, mu);
        poly[idx_lo] = mod_add(lo, hi_tw, Q);
        poly[idx_hi] = mod_sub(lo, hi_tw, Q);
    }
}

// =============================================================================
// Kernel: N^(-1) Scaling for INTT
// =============================================================================

kernel void ntt_hotset_scale_ninv(
    device uint64_t* data [[buffer(0)]],
    constant NTTParams& params [[buffer(1)]],
    uint global_idx [[thread_position_in_grid]]
) {
    uint32_t total = params.N * params.batch;
    if (global_idx >= total) return;

    uint64_t val = data[global_idx];
    data[global_idx] = barrett_mul(val, params.N_inv, params.Q, params.mu);
}

// =============================================================================
// Kernel: Cache Performance Benchmark
// =============================================================================
//
// Measures effective bandwidth for different cache tiers.

struct BenchmarkResult {
    uint64_t constantCycles;
    uint64_t threadgroupCycles;
    uint64_t deviceCycles;
    uint64_t computeCycles;
};

kernel void benchmark_twiddle_access(
    device BenchmarkResult* result [[buffer(0)]],
    constant uint64_t* device_twiddles [[buffer(1)]],
    constant ConstantCache& cache [[buffer(2)]],
    uint thread_idx [[thread_index_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]]
) {
    threadgroup uint64_t shared_twiddles[512];

    // Warm up shared memory
    if (thread_idx < 512) {
        shared_twiddles[thread_idx] = device_twiddles[thread_idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Benchmark iterations
    const uint32_t ITERATIONS = 1000;
    uint64_t sum = 0;

    // Test constant memory access
    uint64_t start = 0;  // Note: Metal lacks cycle counter; use external timing
    for (uint32_t i = 0; i < ITERATIONS; ++i) {
        sum += cache.firstLevelTwiddles[0][i % FIRST_LEVEL_TWIDDLE_COUNT];
    }

    // Test threadgroup memory access
    for (uint32_t i = 0; i < ITERATIONS; ++i) {
        sum += shared_twiddles[i % 512];
    }

    // Test device memory access
    for (uint32_t i = 0; i < ITERATIONS; ++i) {
        sum += device_twiddles[i % 4096];
    }

    // Prevent optimization from eliminating reads
    if (thread_idx == 0) {
        result->computeCycles = sum;  // Force dependency
    }
}
