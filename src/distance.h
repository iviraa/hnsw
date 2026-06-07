#pragma once
#include <cstddef>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// Squared distance. The square root would not change the ordering.
inline float l2(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

#if defined(__ARM_NEON)

// Four accumulators so each add does not wait on the one before it.
inline float l2_simd(const float* a, const float* b, size_t dim) {
    float32x4_t s0 = vdupq_n_f32(0);
    float32x4_t s1 = vdupq_n_f32(0);
    float32x4_t s2 = vdupq_n_f32(0);
    float32x4_t s3 = vdupq_n_f32(0);

    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));

        s0 = vfmaq_f32(s0, d0, d0);
        s1 = vfmaq_f32(s1, d1, d1);
        s2 = vfmaq_f32(s2, d2, d2);
        s3 = vfmaq_f32(s3, d3, d3);
    }

    float32x4_t pair0 = vaddq_f32(s0, s1);
    float32x4_t pair1 = vaddq_f32(s2, s3);
    float sum = vaddvq_f32(vaddq_f32(pair0, pair1));

    for (; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

#else
inline float l2_simd(const float* a, const float* b, size_t dim) { return l2(a, b, dim); }
#endif
