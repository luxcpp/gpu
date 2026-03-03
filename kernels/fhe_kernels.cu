// =============================================================================
// Lux FHE GPU Kernels - CUDA Port of fhe_kernels.metal
// =============================================================================
// Byte-identical arithmetic output.
//
// Kernel Architecture:
//   1. Fused NTT with shared memory optimization
//   2. Fused CMux (rotate + decompose + external product)
//   3. Pipelined Key Switching
//   4. Barrett/Montgomery modular arithmetic

#include <cstdint>

#ifdef __CUDA_ARCH__
#define FHE_DEVICE __device__ __forceinline__
#else
#define FHE_DEVICE inline
#define __global__
#define __shared__
static inline uint64_t __umul64hi(uint64_t a, uint64_t b) {
    __uint128_t r = (__uint128_t)a * b; return (uint64_t)(r >> 64);
}
static inline void __syncthreads() {}
#endif

// Compile-time constants passed via template or runtime params on CUDA
// (Metal uses function_constant; CUDA uses kernel arguments)

struct FHEParams {
    uint32_t N;
    uint32_t LOG_N;
    uint32_t L;
    uint32_t BASE_LOG;
    uint32_t n_lwe;
    uint64_t Q;
    uint64_t Q_INV;
    uint64_t BARRETT_MU;
};

// Shared memory struct for NTT
struct NTTSharedMem {
    uint64_t data[1024];
    uint64_t twiddles[512];
};

FHE_DEVICE uint64_t barrett_reduce(uint64_t x, uint64_t BARRETT_MU, uint64_t Q) {
    uint64_t q_hat = __umul64hi(x, BARRETT_MU);
    uint64_t r = x - q_hat * Q;
    return r >= Q ? r - Q : r;
}

FHE_DEVICE uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    return sum >= Q ? sum - Q : sum;
}

FHE_DEVICE uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    return a >= b ? a - b : Q - b + a;
}

FHE_DEVICE uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t BARRETT_MU, uint64_t Q) {
    uint64_t product = a * b;
    return barrett_reduce(product, BARRETT_MU, Q);
}

FHE_DEVICE uint64_t mod_neg(uint64_t a, uint64_t Q) {
    return a == 0 ? 0 : Q - a;
}

// =============================================================================
// KERNEL 1: FORWARD NTT (Cooley-Tukey)
// =============================================================================

extern "C" __global__ void ntt_forward_fused(
    uint64_t* data, const uint64_t* twiddles,
    uint32_t ring_dim, uint32_t log_ring_dim, uint32_t batch_size,
    uint64_t Q, uint64_t BARRETT_MU)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t smem[];
    // smem layout: [0, ring_dim) = data, [ring_dim, ring_dim + ring_dim/2) = twiddles
    uint64_t* s_data = smem;
    uint64_t* s_twiddles = smem + ring_dim;

    uint32_t tg_size = blockDim.x;
    uint32_t batch_idx = blockIdx.y;
    uint32_t local_idx = threadIdx.x;

    if (batch_idx >= batch_size) return;

    uint64_t* poly = data + batch_idx * ring_dim;

    // Load with bit-reversal
    for (uint32_t i = local_idx; i < ring_dim; i += tg_size) {
        uint32_t j = 0;
        uint32_t temp = i;
        for (uint32_t k = 0; k < log_ring_dim; k++) {
            j = (j << 1) | (temp & 1);
            temp >>= 1;
        }
        s_data[j] = poly[i];
    }

    for (uint32_t i = local_idx; i < ring_dim / 2; i += tg_size) {
        s_twiddles[i] = twiddles[i];
    }
    __syncthreads();

    // Cooley-Tukey butterfly stages
    for (uint32_t stage = 0; stage < log_ring_dim; stage++) {
        uint32_t len = 1u << (stage + 1);
        uint32_t half_len = len >> 1;
        uint32_t step = ring_dim / len;

        for (uint32_t idx = local_idx; idx < ring_dim / 2; idx += tg_size) {
            uint32_t group = idx / half_len;
            uint32_t j = idx % half_len;
            uint32_t i = group * len + j;

            uint64_t w = s_twiddles[j * step];
            uint64_t a = s_data[i];
            uint64_t b = s_data[i + half_len];
            uint64_t wb = mod_mul(w, b, BARRETT_MU, Q);
            s_data[i] = mod_add(a, wb, Q);
            s_data[i + half_len] = mod_sub(a, wb, Q);
        }
        __syncthreads();
    }

    for (uint32_t i = local_idx; i < ring_dim; i += tg_size) {
        poly[i] = s_data[i];
    }
#endif
}

// =============================================================================
// KERNEL 2: INVERSE NTT (Gentleman-Sande)
// =============================================================================

extern "C" __global__ void ntt_inverse_fused(
    uint64_t* data, const uint64_t* inv_twiddles, uint64_t n_inv,
    uint32_t ring_dim, uint32_t log_ring_dim, uint32_t batch_size,
    uint64_t Q, uint64_t BARRETT_MU)
{
#ifdef __CUDA_ARCH__
    extern __shared__ uint64_t smem[];
    uint64_t* s_data = smem;
    uint64_t* s_twiddles = smem + ring_dim;

    uint32_t tg_size = blockDim.x;
    uint32_t batch_idx = blockIdx.y;
    uint32_t local_idx = threadIdx.x;

    if (batch_idx >= batch_size) return;

    uint64_t* poly = data + batch_idx * ring_dim;

    for (uint32_t i = local_idx; i < ring_dim; i += tg_size)
        s_data[i] = poly[i];
    for (uint32_t i = local_idx; i < ring_dim / 2; i += tg_size)
        s_twiddles[i] = inv_twiddles[i];
    __syncthreads();

    for (int stage = log_ring_dim - 1; stage >= 0; stage--) {
        uint32_t len = 1u << (stage + 1);
        uint32_t half_len = len >> 1;
        uint32_t step = ring_dim / len;

        for (uint32_t idx = local_idx; idx < ring_dim / 2; idx += tg_size) {
            uint32_t group = idx / half_len;
            uint32_t j = idx % half_len;
            uint32_t i = group * len + j;
            uint64_t w = s_twiddles[j * step];
            uint64_t a = s_data[i];
            uint64_t b = s_data[i + half_len];
            s_data[i] = mod_add(a, b, Q);
            s_data[i + half_len] = mod_mul(mod_sub(a, b, Q), w, BARRETT_MU, Q);
        }
        __syncthreads();
    }

    for (uint32_t i = local_idx; i < ring_dim; i += tg_size) {
        uint32_t j = 0;
        uint32_t temp = i;
        for (uint32_t k = 0; k < log_ring_dim; k++) {
            j = (j << 1) | (temp & 1);
            temp >>= 1;
        }
        poly[j] = mod_mul(s_data[i], n_inv, BARRETT_MU, Q);
    }
#endif
}

// =============================================================================
// KERNEL 3: INITIALIZE ACCUMULATOR
// =============================================================================

extern "C" __global__ void init_accumulator(
    uint64_t* acc, const uint64_t* test_poly, const int32_t* rotations,
    uint32_t ring_dim, uint32_t batch_size, uint64_t Q)
{
#ifdef __CUDA_ARCH__
    uint32_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = blockIdx.z;

    if (coeff_idx >= ring_dim || batch_idx >= batch_size) return;

    int32_t rotation = rotations[batch_idx];
    uint32_t two_n = 2 * ring_dim;
    rotation = ((rotation % (int32_t)two_n) + (int32_t)two_n) % (int32_t)two_n;

    int32_t src_idx = (int32_t)coeff_idx + rotation;
    bool negate = false;

    if (src_idx >= (int32_t)two_n) src_idx -= two_n;
    if (src_idx >= (int32_t)ring_dim) {
        src_idx -= ring_dim;
        negate = true;
    }

    uint64_t val = test_poly[src_idx];
    if (negate) val = mod_neg(val, Q);

    uint32_t out_idx_c0 = batch_idx * 2 * ring_dim + coeff_idx;
    uint32_t out_idx_c1 = batch_idx * 2 * ring_dim + ring_dim + coeff_idx;

    acc[out_idx_c0] = 0;
    acc[out_idx_c1] = val;
#endif
}

// =============================================================================
// KERNEL 4: FUSED CMUX GATE
// =============================================================================

extern "C" __global__ void cmux_fused(
    uint64_t* acc, const uint64_t* bsk, const int32_t* rotations,
    uint32_t ring_dim, uint32_t num_levels, uint32_t decomp_log,
    uint32_t batch_size, uint64_t Q, uint64_t BARRETT_MU)
{
#ifdef __CUDA_ARCH__
    uint32_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t out_comp = blockIdx.y;
    uint32_t batch_idx = blockIdx.z;

    if (coeff_idx >= ring_dim || out_comp >= 2 || batch_idx >= batch_size) return;

    int32_t rotation = rotations[batch_idx];
    uint64_t mask = (1ULL << decomp_log) - 1;
    uint32_t two_n = 2 * ring_dim;

    if (rotation == 0) return;

    rotation = ((rotation % (int32_t)two_n) + (int32_t)two_n) % (int32_t)two_n;

    int32_t src_idx = (int32_t)coeff_idx - rotation;
    bool negate = false;
    if (src_idx < 0) src_idx += two_n;
    if (src_idx >= (int32_t)ring_dim) {
        src_idx -= ring_dim;
        negate = true;
    }

    uint64_t acc_orig[2], acc_rot[2], diff[2];
    for (int c = 0; c < 2; c++) {
        uint32_t orig_idx = batch_idx * 2 * ring_dim + c * ring_dim + coeff_idx;
        uint32_t rot_idx = batch_idx * 2 * ring_dim + c * ring_dim + src_idx;
        acc_orig[c] = acc[orig_idx];
        acc_rot[c] = acc[rot_idx];
        if (negate) acc_rot[c] = mod_neg(acc_rot[c], Q);
        diff[c] = mod_sub(acc_rot[c], acc_orig[c], Q);
    }

    uint64_t ext_prod_sum = 0;
    for (uint32_t in_comp = 0; in_comp < 2; in_comp++) {
        uint64_t val = diff[in_comp];
        for (uint32_t l = 0; l < num_levels; l++) {
            uint64_t digit = (val >> (l * decomp_log)) & mask;
            uint32_t bsk_idx = in_comp * num_levels * 2 * ring_dim +
                              l * 2 * ring_dim + out_comp * ring_dim + coeff_idx;
            uint64_t bsk_val = bsk[bsk_idx];
            ext_prod_sum = mod_add(ext_prod_sum, mod_mul(digit, bsk_val, BARRETT_MU, Q), Q);
        }
    }

    uint32_t out_idx = batch_idx * 2 * ring_dim + out_comp * ring_dim + coeff_idx;
    acc[out_idx] = mod_add(acc_orig[out_comp], ext_prod_sum, Q);
#endif
}

// =============================================================================
// KERNEL 5: KEY SWITCHING
// =============================================================================

extern "C" __global__ void key_switch_decompose(
    uint64_t* digits, const uint64_t* rlwe,
    uint32_t ring_dim, uint32_t num_levels, uint32_t decomp_log, uint32_t batch_size)
{
#ifdef __CUDA_ARCH__
    uint32_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = blockIdx.y;
    if (coeff_idx >= ring_dim || batch_idx >= batch_size) return;

    uint64_t mask = (1ULL << decomp_log) - 1;
    uint32_t rlwe_idx = batch_idx * 2 * ring_dim + coeff_idx;
    uint64_t val = rlwe[rlwe_idx];

    for (uint32_t l = 0; l < num_levels; l++) {
        uint32_t digit_idx = batch_idx * ring_dim * num_levels + coeff_idx * num_levels + l;
        digits[digit_idx] = (val >> (l * decomp_log)) & mask;
    }
#endif
}

extern "C" __global__ void key_switch_accumulate(
    uint64_t* lwe, const uint64_t* digits, const uint64_t* ksk,
    const uint64_t* rlwe_c1,
    uint32_t ring_dim, uint32_t lwe_dim, uint32_t num_levels, uint32_t batch_size,
    uint64_t Q, uint64_t BARRETT_MU)
{
#ifdef __CUDA_ARCH__
    uint32_t lwe_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = blockIdx.y;
    if (lwe_idx > lwe_dim || batch_idx >= batch_size) return;

    uint64_t sum = 0;
    for (uint32_t j = 0; j < ring_dim; j++) {
        for (uint32_t l = 0; l < num_levels; l++) {
            uint32_t digit_idx = batch_idx * ring_dim * num_levels + j * num_levels + l;
            uint64_t digit = digits[digit_idx];
            if (digit == 0) continue;
            uint32_t ksk_idx = j * num_levels * (lwe_dim + 1) + l * (lwe_dim + 1) + lwe_idx;
            sum = mod_add(sum, mod_mul(digit, ksk[ksk_idx], BARRETT_MU, Q), Q);
        }
    }

    if (lwe_idx == lwe_dim) {
        uint32_t c1_idx = batch_idx * ring_dim;
        sum = mod_add(rlwe_c1[c1_idx], sum, Q);
    }

    uint32_t out_idx = batch_idx * (lwe_dim + 1) + lwe_idx;
    lwe[out_idx] = sum;
#endif
}

// =============================================================================
// KERNEL 6: POINTWISE POLYNOMIAL MULTIPLICATION (NTT Domain)
// =============================================================================

extern "C" __global__ void pointwise_mul(
    uint64_t* result, const uint64_t* poly_a, const uint64_t* poly_b,
    uint32_t ring_dim, uint32_t batch_size, uint64_t Q, uint64_t BARRETT_MU)
{
#ifdef __CUDA_ARCH__
    uint32_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = blockIdx.y;
    if (coeff_idx >= ring_dim || batch_idx >= batch_size) return;
    uint32_t idx = batch_idx * ring_dim + coeff_idx;
    result[idx] = mod_mul(poly_a[idx], poly_b[idx], BARRETT_MU, Q);
#endif
}

// =============================================================================
// KERNEL 7: BATCH MODULAR OPERATIONS
// =============================================================================

extern "C" __global__ void batch_mod_add(
    uint64_t* result, const uint64_t* a, const uint64_t* b,
    uint32_t size, uint64_t Q)
{
#ifdef __CUDA_ARCH__
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    result[gid] = mod_add(a[gid], b[gid], Q);
#endif
}

extern "C" __global__ void batch_mod_sub(
    uint64_t* result, const uint64_t* a, const uint64_t* b,
    uint32_t size, uint64_t Q)
{
#ifdef __CUDA_ARCH__
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    result[gid] = mod_sub(a[gid], b[gid], Q);
#endif
}
