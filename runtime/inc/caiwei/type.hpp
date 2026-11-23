#ifndef CAIWEI_RUNTIME_TYPE_HPP
#define CAIWEI_RUNTIME_TYPE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#else
#if CAIWEI_OS_WIN
#if !defined(HAS_AVX2) || !defined(HAS_F16C)
#error "Require AVX2 + F16C"
#endif
#else
#if !defined(__AVX2__) || !defined(__F16C__)
#error "Require AVX2 + F16C"
#endif
#endif
#include <immintrin.h>
#endif

namespace caiwei {
namespace type {

inline void u8_to_i8(const uint8_t* __restrict src, const size_t size, int8_t* __restrict dst) {
    // 不用SIMD指令Release优化效果一样几乎没有性能提升
    for (size_t i = 0; i < size; ++i) {
        dst[i] = static_cast<int8_t>(static_cast<int>(src[i]) - 128);
    }
}

inline void u8_to_f32(const uint8_t* __restrict src, const size_t size, float* __restrict dst, const float scale) {
    if (scale == 0.0F) {
        CW_LOG_W("缩放系数无效: %f", scale);
        return;
    }
    // 不用SIMD指令Release优化效果一样几乎没有性能提升
    const float scale_inv = 1.0F / scale;
    for (size_t i = 0; i < size; ++i) {
        dst[i] = static_cast<float>(src[i]) * scale_inv;
    }
}

inline int8_t qnt_affine_to_i8(const float f32, const int32_t zp, const float scale) {
    // clang-format off
    const float   f32v = std::clamp((f32 / scale) + static_cast<float>(zp), -128.0F, 127.0F);
    const int32_t ret  = static_cast<int32_t>(std::round(f32v));
    // clang-format on
    return static_cast<int8_t>(ret);
}

inline float deqnt_affine_to_f32(const int8_t qnt, const int32_t zp, const float scale) {
    return ((static_cast<float>(qnt) - static_cast<float>(zp)) * scale);
}

#if defined(__aarch64__)
inline void f32_to_fp16(uint16_t* __restrict dst, const float* __restrict src, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t f32v = vld1q_f32(src + i);
        float16x4_t f16v = vcvt_f16_f32(f32v);
        vst1_f16(reinterpret_cast<__fp16*>(dst + i), f16v);
    }
    for (; i < count; ++i) {
        __fp16 v = static_cast<__fp16>(src[i]);
        std::memcpy(dst + i, &v, sizeof(uint16_t));
    }
}

inline void fp16_to_f32(float* __restrict dst, const uint16_t* __restrict src, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float16x4_t f16v = vld1_f16(reinterpret_cast<const __fp16*>(src + i));
        float32x4_t f32v = vcvt_f32_f16(f16v);
        vst1q_f32(dst + i, f32v);
    }
    for (; i < count; ++i) {
        __fp16 v;
        std::memcpy(&v, src + i, sizeof(uint16_t));
        dst[i] = static_cast<float>(v);
    }
}
#else
inline void f32_to_fp16(uint16_t* __restrict dst, const float* __restrict src, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        // clang-format off
        __m256  vf = _mm256_loadu_ps(src + i);
        __m128i vh = _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // clang-format on
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), vh);
    }
    for (; i < count; ++i) {
        // clang-format off
        __m128  vf = _mm_set_ss(src[i]);
        __m128i vh = _mm_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // clang-format on
        dst[i] = static_cast<uint16_t>(_mm_extract_epi16(vh, 0));
    }
}

inline void fp16_to_f32(float* __restrict dst, const uint16_t* __restrict src, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        // clang-format off
        __m128i vh = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256  vf = _mm256_cvtph_ps(vh);
        // clang-format on
        _mm256_storeu_ps(dst + i, vf);
    }
    for (; i < count; ++i) {
        // clang-format off
        __m128i vh = _mm_set1_epi16(static_cast<int16_t>(src[i]));
        __m128  vf = _mm_cvtph_ps(vh);
        // clang-format on
        dst[i] = _mm_cvtss_f32(vf);
    }
}
#endif

} // namespace type
} // namespace caiwei

#endif // CAIWEI_RUNTIME_TYPE_HPP
