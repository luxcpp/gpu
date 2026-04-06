// PAT-FHE-012: EVM256 Parallel Processing - Metal Implementation
// uint256 operations using 4x uint64 limbs (little-endian)

#include <metal_stdlib>
using namespace metal;

#define LIMBS 4

// uint256 represented as 4x uint64 limbs
struct uint256_t {
    ulong limbs[LIMBS];
};

// ============================================================================
// Helper functions for multi-limb arithmetic
// ============================================================================

inline ulong add_carry(ulong a, ulong b, thread ulong* carry) {
    ulong sum = a + b;
    ulong c1 = sum < a ? 1 : 0;
    ulong result = sum + *carry;
    ulong c2 = result < sum ? 1 : 0;
    *carry = c1 | c2;
    return result;
}

inline ulong sub_borrow(ulong a, ulong b, thread ulong* borrow) {
    ulong diff = a - b;
    ulong b1 = diff > a ? 1 : 0;
    ulong result = diff - *borrow;
    ulong b2 = result > diff ? 1 : 0;
    *borrow = b1 | b2;
    return result;
}

// Full 64x64 -> 128 bit multiplication
inline void mul64_wide(ulong a, ulong b, thread ulong* hi, thread ulong* lo) {
    ulong a_lo = a & 0xFFFFFFFF;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFF;
    ulong b_hi = b >> 32;

    ulong p0 = a_lo * b_lo;
    ulong p1 = a_lo * b_hi;
    ulong p2 = a_hi * b_lo;
    ulong p3 = a_hi * b_hi;

    ulong cy = ((p0 >> 32) + (p1 & 0xFFFFFFFF) + (p2 & 0xFFFFFFFF)) >> 32;
    *lo = p0 + (p1 << 32) + (p2 << 32);
    *hi = p3 + (p1 >> 32) + (p2 >> 32) + cy;
}

// ============================================================================
// Kernel: Batch Add256
// ============================================================================

kernel void metal_add256(
    const device uint256_t* a [[buffer(0)]],
    const device uint256_t* b [[buffer(1)]],
    device uint256_t* result [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    ulong carry = 0;
    for (int i = 0; i < LIMBS; i++) {
        result[idx].limbs[i] = add_carry(a[idx].limbs[i], b[idx].limbs[i], &carry);
    }
}

// ============================================================================
// Kernel: Batch Sub256
// ============================================================================

kernel void metal_sub256(
    const device uint256_t* a [[buffer(0)]],
    const device uint256_t* b [[buffer(1)]],
    device uint256_t* result [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    ulong borrow = 0;
    for (int i = 0; i < LIMBS; i++) {
        result[idx].limbs[i] = sub_borrow(a[idx].limbs[i], b[idx].limbs[i], &borrow);
    }
}

// ============================================================================
// Kernel: Batch Mul256
// ============================================================================

kernel void metal_mul256(
    const device uint256_t* a [[buffer(0)]],
    const device uint256_t* b [[buffer(1)]],
    device uint256_t* result [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    // Schoolbook multiplication with 8 limbs intermediate result
    ulong product[LIMBS * 2];
    for (int i = 0; i < LIMBS * 2; i++) {
        product[i] = 0;
    }

    for (int i = 0; i < LIMBS; i++) {
        ulong carry = 0;
        for (int j = 0; j < LIMBS; j++) {
            ulong hi, lo;
            mul64_wide(a[idx].limbs[i], b[idx].limbs[j], &hi, &lo);

            // Add to product[i+j]
            ulong sum = product[i + j] + lo + carry;
            carry = (sum < product[i + j]) ? 1 : 0;
            carry += hi;
            product[i + j] = sum;
        }
        product[i + LIMBS] += carry;
    }

    // Take lower 256 bits
    for (int i = 0; i < LIMBS; i++) {
        result[idx].limbs[i] = product[i];
    }
}

// ============================================================================
// Helper: Compare uint256
// ============================================================================

inline int cmp256(const thread uint256_t* a, const thread uint256_t* b) {
    for (int i = LIMBS - 1; i >= 0; i--) {
        if (a->limbs[i] > b->limbs[i]) return 1;
        if (a->limbs[i] < b->limbs[i]) return -1;
    }
    return 0;
}

inline bool is_zero(const thread uint256_t* a) {
    for (int i = 0; i < LIMBS; i++) {
        if (a->limbs[i] != 0) return false;
    }
    return true;
}

// ============================================================================
// Helper: Div256 Implementation
// ============================================================================

inline void div256_impl(const thread uint256_t* numerator, const thread uint256_t* denominator,
                        thread uint256_t* quotient, thread uint256_t* remainder) {
    // Handle division by zero
    if (is_zero(denominator)) {
        for (int i = 0; i < LIMBS; i++) {
            quotient->limbs[i] = 0;
            remainder->limbs[i] = 0;
        }
        return;
    }

    // Initialize
    uint256_t q, r;
    for (int i = 0; i < LIMBS; i++) {
        q.limbs[i] = 0;
        r.limbs[i] = 0;
    }

    // Long division algorithm
    for (int i = 255; i >= 0; i--) {
        // r <<= 1
        ulong carry = 0;
        for (int j = 0; j < LIMBS; j++) {
            ulong temp = (r.limbs[j] << 1) | carry;
            carry = r.limbs[j] >> 63;
            r.limbs[j] = temp;
        }

        // r[0] = numerator[i]
        int limb_idx = i / 64;
        int bit_idx = i % 64;
        ulong bit = (numerator->limbs[limb_idx] >> bit_idx) & 1;
        r.limbs[0] |= bit;

        // if r >= denominator
        if (cmp256(&r, denominator) >= 0) {
            // r -= denominator
            ulong borrow = 0;
            for (int j = 0; j < LIMBS; j++) {
                r.limbs[j] = sub_borrow(r.limbs[j], denominator->limbs[j], &borrow);
            }

            // q[i] = 1
            limb_idx = i / 64;
            bit_idx = i % 64;
            q.limbs[limb_idx] |= (1UL << bit_idx);
        }
    }

    *quotient = q;
    *remainder = r;
}

// ============================================================================
// Kernel: Batch Div256
// ============================================================================

kernel void metal_div256(
    const device uint256_t* a [[buffer(0)]],
    const device uint256_t* b [[buffer(1)]],
    device uint256_t* result [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    uint256_t numerator = a[idx];
    uint256_t denominator = b[idx];
    uint256_t quotient, remainder;

    div256_impl(&numerator, &denominator, &quotient, &remainder);
    result[idx] = quotient;
}

// ============================================================================
// Kernel: Batch Mod256
// ============================================================================

kernel void metal_mod256(
    const device uint256_t* a [[buffer(0)]],
    const device uint256_t* b [[buffer(1)]],
    device uint256_t* result [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    uint256_t numerator = a[idx];
    uint256_t denominator = b[idx];
    uint256_t quotient, remainder;

    div256_impl(&numerator, &denominator, &quotient, &remainder);
    result[idx] = remainder;
}

// ============================================================================
// Montgomery multiplication helpers
// ============================================================================

inline void montgomery_reduce(const thread ulong* t, const thread uint256_t* m,
                              ulong m_inv, thread uint256_t* result) {
    ulong a[LIMBS * 2];
    for (int i = 0; i < LIMBS * 2; i++) {
        a[i] = t[i];
    }

    for (int i = 0; i < LIMBS; i++) {
        ulong u = a[i] * m_inv;
        ulong carry = 0;

        for (int j = 0; j < LIMBS; j++) {
            ulong hi, lo;
            mul64_wide(u, m->limbs[j], &hi, &lo);

            ulong sum = a[i + j] + lo + carry;
            carry = (sum < a[i + j]) ? 1 : 0;
            carry += hi;
            a[i + j] = sum;
        }

        for (int j = LIMBS; j < LIMBS * 2 - i && carry; j++) {
            ulong sum = a[i + j] + carry;
            carry = (sum < a[i + j]) ? 1 : 0;
            a[i + j] = sum;
        }
    }

    // Result is in upper half
    bool needs_sub = false;
    for (int i = LIMBS - 1; i >= 0; i--) {
        if (a[LIMBS + i] > m->limbs[i]) {
            needs_sub = true;
            break;
        }
        if (a[LIMBS + i] < m->limbs[i]) break;
    }

    if (needs_sub) {
        ulong borrow = 0;
        for (int i = 0; i < LIMBS; i++) {
            result->limbs[i] = sub_borrow(a[LIMBS + i], m->limbs[i], &borrow);
        }
    } else {
        for (int i = 0; i < LIMBS; i++) {
            result->limbs[i] = a[LIMBS + i];
        }
    }
}

// ============================================================================
// Kernel: Montgomery Multiplication
// ============================================================================

kernel void metal_mont_mul(
    const device uint256_t* a [[buffer(0)]],
    const device uint256_t* b [[buffer(1)]],
    const device uint256_t* m [[buffer(2)]],
    const device ulong* m_inv [[buffer(3)]],
    device uint256_t* result [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    // Compute full 512-bit product
    ulong product[LIMBS * 2];
    for (int i = 0; i < LIMBS * 2; i++) {
        product[i] = 0;
    }

    for (int i = 0; i < LIMBS; i++) {
        ulong carry = 0;
        for (int j = 0; j < LIMBS; j++) {
            ulong hi, lo;
            mul64_wide(a[idx].limbs[i], b[idx].limbs[j], &hi, &lo);

            ulong sum = product[i + j] + lo + carry;
            carry = (sum < product[i + j]) ? 1 : 0;
            carry += hi;
            product[i + j] = sum;
        }
        product[i + LIMBS] += carry;
    }

    // Montgomery reduce
    uint256_t mod = m[idx];
    uint256_t res;
    montgomery_reduce(product, &mod, *m_inv, &res);
    result[idx] = res;
}

// ============================================================================
// Kernel: Modular Exponentiation
// ============================================================================

kernel void metal_modexp256(
    const device uint256_t* base [[buffer(0)]],
    const device uint256_t* exponent [[buffer(1)]],
    const device uint256_t* modulus [[buffer(2)]],
    device uint256_t* result [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    // Simple square-and-multiply
    uint256_t res;
    res.limbs[0] = 1;
    res.limbs[1] = 0;
    res.limbs[2] = 0;
    res.limbs[3] = 0;

    uint256_t b = base[idx];
    uint256_t mod = modulus[idx];

    for (int i = 0; i < 256; i++) {
        int limb_idx = i / 64;
        int bit_idx = i % 64;
        ulong bit = (exponent[idx].limbs[limb_idx] >> bit_idx) & 1;

        if (bit) {
            // res = (res * b) % modulus
            ulong product[LIMBS * 2];
            for (int k = 0; k < LIMBS * 2; k++) {
                product[k] = 0;
            }

            for (int j = 0; j < LIMBS; j++) {
                ulong carry = 0;
                for (int k = 0; k < LIMBS; k++) {
                    ulong hi, lo;
                    mul64_wide(res.limbs[j], b.limbs[k], &hi, &lo);

                    ulong sum = product[j + k] + lo + carry;
                    carry = (sum < product[j + k]) ? 1 : 0;
                    carry += hi;
                    product[j + k] = sum;
                }
                product[j + LIMBS] += carry;
            }

            // Take product % modulus
            uint256_t temp;
            for (int j = 0; j < LIMBS; j++) {
                temp.limbs[j] = product[j];
            }

            uint256_t quot, rem;
            div256_impl(&temp, &mod, &quot, &rem);
            res = rem;
        }

        // b = (b * b) % modulus
        ulong product[LIMBS * 2];
        for (int k = 0; k < LIMBS * 2; k++) {
            product[k] = 0;
        }

        for (int j = 0; j < LIMBS; j++) {
            ulong carry = 0;
            for (int k = 0; k < LIMBS; k++) {
                ulong hi, lo;
                mul64_wide(b.limbs[j], b.limbs[k], &hi, &lo);

                ulong sum = product[j + k] + lo + carry;
                carry = (sum < product[j + k]) ? 1 : 0;
                carry += hi;
                product[j + k] = sum;
            }
            product[j + LIMBS] += carry;
        }

        uint256_t temp;
        for (int j = 0; j < LIMBS; j++) {
            temp.limbs[j] = product[j];
        }

        uint256_t quot, rem;
        div256_impl(&temp, &mod, &quot, &rem);
        b = rem;
    }

    result[idx] = res;
}
