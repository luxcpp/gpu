// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// ZK Operations - High-level Poseidon2, Merkle, Commitment, Nullifier
//
// This file provides the high-level ZK API. All cryptographic operations
// delegate to the backend vtable (cpu_backend.cpp for CPU, etc.).
// No duplicate field arithmetic here - one canonical implementation per operation.

#include "gpu_internal.h"
#include <cstring>
#include <cstdint>
#include <vector>

// =============================================================================
// Helper: Next power of 2
// =============================================================================

static inline size_t next_pow2(size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

// =============================================================================
// C API - ZK Operations
// =============================================================================

extern "C" {

// Poseidon2 hash: out[i] = Poseidon2(left[i], right[i])
// Delegates to backend op_poseidon2_hash with rate=2
LuxError lux_gpu_poseidon2(LuxGPU* gpu, LuxFr256* out, const LuxFr256* left, const LuxFr256* right, size_t n) {
    if (!out || !left || !right || n == 0) {
        return LUX_ERROR_INVALID_ARGUMENT;
    }

    if (!gpu || !gpu->vtbl || !gpu->ctx) {
        return LUX_ERROR_INVALID_ARGUMENT;
    }

    if (!gpu->vtbl->op_poseidon2_hash) {
        return LUX_ERROR_NOT_SUPPORTED;
    }

    // The backend expects interleaved [left, right] pairs for rate=2
    // Each pair: 2 field elements (2 * 4 * uint64_t = 8 uint64_t per hash)
    std::vector<uint64_t> inputs(n * 2 * 4);
    for (size_t i = 0; i < n; i++) {
        memcpy(&inputs[i * 8], left[i].limbs, 32);
        memcpy(&inputs[i * 8 + 4], right[i].limbs, 32);
    }

    std::vector<uint64_t> outputs(n * 4);

    LuxBackendError err = gpu->vtbl->op_poseidon2_hash(gpu->ctx, inputs.data(), outputs.data(), 2, n);
    if (err != LUX_BACKEND_OK) {
        return static_cast<LuxError>(err);
    }

    // Copy outputs to LuxFr256 array
    for (size_t i = 0; i < n; i++) {
        memcpy(out[i].limbs, &outputs[i * 4], 32);
    }

    return LUX_OK;
}

// Merkle root: compute root from leaves using Poseidon2
LuxError lux_gpu_merkle_root(LuxGPU* gpu, LuxFr256* out, const LuxFr256* leaves, size_t n) {
    if (!out || !leaves || n == 0) {
        return LUX_ERROR_INVALID_ARGUMENT;
    }

    if (n == 1) {
        memcpy(out, leaves, sizeof(LuxFr256));
        return LUX_OK;
    }

    // Pad to power of 2
    size_t padded_n = next_pow2(n);
    std::vector<LuxFr256> current(padded_n);

    // Copy leaves
    memcpy(current.data(), leaves, n * sizeof(LuxFr256));

    // Zero-pad remaining
    for (size_t i = n; i < padded_n; i++) {
        memset(&current[i], 0, sizeof(LuxFr256));
    }

    // Build tree bottom-up using Poseidon2 compression
    std::vector<LuxFr256> next_level(padded_n / 2);

    while (padded_n > 1) {
        size_t half = padded_n / 2;

        // Prepare left/right arrays for batch Poseidon2
        std::vector<LuxFr256> left_arr(half);
        std::vector<LuxFr256> right_arr(half);

        for (size_t i = 0; i < half; i++) {
            left_arr[i] = current[i * 2];
            right_arr[i] = current[i * 2 + 1];
        }

        LuxError err = lux_gpu_poseidon2(gpu, next_level.data(), left_arr.data(), right_arr.data(), half);
        if (err != LUX_OK) {
            return err;
        }

        // Swap buffers
        for (size_t i = 0; i < half; i++) {
            current[i] = next_level[i];
        }
        padded_n = half;
    }

    memcpy(out, &current[0], sizeof(LuxFr256));
    return LUX_OK;
}

// Commitment: out[i] = Poseidon2(Poseidon2(values[i], blindings[i]), salts[i])
LuxError lux_gpu_commitment(LuxGPU* gpu, LuxFr256* out, const LuxFr256* values, const LuxFr256* blindings, const LuxFr256* salts, size_t n) {
    if (!out || !values || !blindings || !salts || n == 0) {
        return LUX_ERROR_INVALID_ARGUMENT;
    }

    // First hash: inner[i] = Poseidon2(values[i], blindings[i])
    std::vector<LuxFr256> inner(n);
    LuxError err = lux_gpu_poseidon2(gpu, inner.data(), values, blindings, n);
    if (err != LUX_OK) {
        return err;
    }

    // Second hash: out[i] = Poseidon2(inner[i], salts[i])
    return lux_gpu_poseidon2(gpu, out, inner.data(), salts, n);
}

// Nullifier: out[i] = Poseidon2(Poseidon2(keys[i], commitments[i]), indices[i])
LuxError lux_gpu_nullifier(LuxGPU* gpu, LuxFr256* out, const LuxFr256* keys, const LuxFr256* commitments, const LuxFr256* indices, size_t n) {
    // Same structure as commitment
    return lux_gpu_commitment(gpu, out, keys, commitments, indices, n);
}

} // extern "C"
