#ifndef CAIWEI_RUNTIME_TYPE_HPP
#define CAIWEI_RUNTIME_TYPE_HPP

#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>

#if defined(__aarch64__)
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif

namespace caiwei {
namespace type   {

inline void i8_to_f32(const uint8_t* __restrict src, const uint32_t size, float* __restrict dst) {
    std::transform(src, src + size, dst, [](uint8_t v) {
        return static_cast<float>(v);
    });
}

inline void i8_to_f32(const uint8_t* __restrict src, const uint32_t size, float* __restrict dst, const float scale) {
    std::transform(src, src + size, dst, [scale](uint8_t v) {
        return static_cast<float>(v) / scale;
    });
}

inline float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
    return ((static_cast<float>(qnt) - static_cast<float>(zp)) * scale);
}

inline int8_t qnt_affine_to_i8(float f32, int32_t zp, float scale) {
    int32_t ret = std::round(std::clamp((f32 / scale) + zp, -128.0F, 127.0F));
    return static_cast<int8_t>(ret);
}

#if defined(__aarch64__)
inline void f32_to_fp16(uint16_t* __restrict dst, const float* __restrict src, int count) {
    int i = 0;
    for (; i <= count - 4; i += 4) {
        float32x4_t f32v = vld1q_f32(src + i);
        float16x4_t f16v = vcvt_f16_f32(f32v);
        vst1_f16((__fp16*) (dst + i), f16v);
    }
    for (; i < count; ++i) {
        __fp16 h = static_cast<__fp16>(src[i]);
        std::memcpy(dst + i, &h, sizeof(uint16_t));
    }
}

inline void fp16_to_f32(float* __restrict dst, const uint16_t* __restrict src, int count) {
    int i = 0;
    for (; i <= count - 4; i += 4) {
        float16x4_t f16v = vld1_f16((const __fp16*) (src + i));
        float32x4_t f32v = vcvt_f32_f16(f16v);
        vst1q_f32(dst + i, f32v);
    }
    for (; i < count; ++i) {
        __fp16 h;
        std::memcpy(&h, src + i, sizeof(uint16_t));
        dst[i] = static_cast<float>(h);
    }
}
#else
inline void f32_to_fp16(uint16_t* __restrict dst, const float* __restrict src, int count) {
    int i = 0;
    for (; i <= count - 8; i += 8) {
        __m256  vf = _mm256_loadu_ps(src + i);
        __m128i vh = _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i*) (dst + i), vh);
    }
    for (; i < count; i++) {
        __m128  vf = _mm_set_ss(src[i]);
        __m128i vh = _mm_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        dst[i] = (uint16_t) _mm_extract_epi16(vh, 0);
    }
}

inline void fp16_to_f32(float* __restrict dst, const uint16_t* __restrict src, int count) {
    int i = 0;
    for (; i <= count - 8; i += 8) {
        __m128i vh = _mm_loadu_si128((const __m128i*)(src + i));
        __m256  vf = _mm256_cvtph_ps(vh);
        _mm256_storeu_ps(dst + i, vf);
    }
    for (; i < count; i++) {
        __m128i vh = _mm_set1_epi16(src[i]);
        __m128  vf = _mm_cvtph_ps(vh);
        dst[i] = _mm_cvtss_f32(vf);
    }
}
#endif

} // type
} // caiwei

#endif // CAIWEI_RUNTIME_TYPE_HPP
