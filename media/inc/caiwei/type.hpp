/**
 * 类型转换
 */
#ifndef CAIWEI_MEDIA_TYPE_HPP
#define CAIWEI_MEDIA_TYPE_HPP

#include <bit>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace caiwei {
namespace type   {

inline void i8_to_f32(const uint8_t* src, const uint32_t size, float* dst) {
    std::transform(src, src + size, dst, [](uint8_t v) {
        return static_cast<float>(v);
    });
}

inline void i8_to_f32(const uint8_t* src, uint32_t size, float* dst, const float scale) {
    std::transform(src, src + size, dst, [scale](uint8_t v) {
        return static_cast<float>(v) / scale;
    });
}

inline float clip(float val, float min, float max) {
    return val <= min ? min : (val >= max ? max : val);
}

inline float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
    return ((static_cast<float>(qnt) - static_cast<float>(zp)) * scale);
}

inline int8_t qnt_affine_to_i8(float f32, int32_t zp, float scale) {
    int32_t ret = std::round(clip((f32 / scale) + zp, -128.0F, 127.0F));
    return static_cast<int8_t>(ret);
}

#if defined(__aarch64__)
#include <arm_neon.h>

inline void f32_to_fp16(uint16_t* dst, const float* src, int count) {
    int i = 0;
    for (; i <= count - 4; i += 4) {
        float32x4_t f32v = vld1q_f32(src + i);
        float16x4_t f16v = vcvt_f16_f32(f32v);
        vst1_f16((__fp16*)(dst + i), f16v);
    }
    for (; i < count; ++i) {
        __fp16 h = static_cast<__fp16>(src[i]);
        std::memcpy(dst + i, &h, sizeof(uint16_t));
    }
}

inline void fp16_to_f32(float* dst, const uint16_t* src, int count) {
    int i = 0;
    for (; i <= count - 4; i += 4) {
        float16x4_t f16v = vld1_f16((const __fp16*)(src + i));
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
inline uint16_t f32_to_f16_fallback(float value) {
    uint32_t x;
    std::memcpy(&x, &value, sizeof(x));
    uint32_t sign = x & 0x80000000u;
    uint32_t exp  = x & 0x7F800000u;
    uint32_t man  = x & 0x007FFFFFu;
    if (exp == 0x7F800000u) {
        uint32_t nan_bit = (man == 0) ? 0 : 0x0200u;
        return static_cast<uint16_t>((sign >> 16) | 0x7C00u | nan_bit | (man >> 13));
    }
    uint32_t half_sign   = sign >> 16;
    int32_t unbiased_exp = static_cast<int32_t>(exp >> 23) - 127;
    int32_t half_exp     = unbiased_exp + 15;
    if (half_exp >= 0x1F) {
        return static_cast<uint16_t>(half_sign | 0x7C00u);
    }
    if (half_exp <= 0) {
        if ((14 - half_exp) > 24) {
            return static_cast<uint16_t>(half_sign);
        }
        man |= 0x00800000u;
        uint32_t half_man = man >> (14 - half_exp);
        uint32_t round_bit = 1u << (13 - half_exp);
        if ((man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0) {
            half_man++;
        }
        return static_cast<uint16_t>(half_sign | half_man);
    }
    half_exp = static_cast<uint32_t>(half_exp) << 10;
    uint32_t half_man = man >> 13;
    uint32_t round_bit = 0x00001000u;
    if ((man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0) {
        return static_cast<uint16_t>((half_sign | half_exp | half_man) + 1u);
    } else {
        return static_cast<uint16_t>(half_sign | half_exp | half_man);
    }
}

inline float f16_to_f32_fallback(uint16_t i) {
    if ((i & 0x7FFF) == 0) {
        uint32_t result = static_cast<uint32_t>(i) << 16;
        float f;
        std::memcpy(&f, &result, sizeof(result));
        return f;
    }
    uint32_t half_sign = i & 0x8000u;
    uint32_t half_exp  = i & 0x7C00u;
    uint32_t half_man  = i & 0x03FFu;
    if (half_exp == 0x7C00u) {
        uint32_t result;
        if (half_man == 0) {
            result = (half_sign << 16) | 0x7F800000u;
        } else {
            result = (half_sign << 16) | 0x7FC00000u | (half_man << 13);
        }
        float f;
        std::memcpy(&f, &result, sizeof(result));
        return f;
    }
    uint32_t sign = half_sign << 16;
    int32_t unbiased_exp = static_cast<int32_t>(half_exp >> 10) - 15;
    if (half_exp == 0) {
        int e = std::countl_zero(half_man) - 6;
        uint32_t exp = (127 - 15 - e) << 23;
        uint32_t man = (half_man << (14 + e)) & 0x7FFFFFu;
        uint32_t result = sign | exp | man;
        float f;
        std::memcpy(&f, &result, sizeof(result));
        return f;
    }
    uint32_t exp = (static_cast<uint32_t>(unbiased_exp + 127)) << 23;
    uint32_t man = (half_man & 0x03FFu) << 13;
    uint32_t result = sign | exp | man;
    float f;
    std::memcpy(&f, &result, sizeof(result));
    return f;
}

inline void f32_to_fp16(uint16_t* dst, const float* src, int count) {
    for(int i = 0; i < count; ++i) {
        dst[i] = f32_to_f16_fallback(src[i]);
    }
}

inline void fp16_to_f32(float* dst, const uint16_t* src, int count) {
    for(int i = 0; i < count; ++i) {
        dst[i] = f16_to_f32_fallback(src[i]);
    }
}
#endif

}
}

#endif // CAIWEI_MEDIA_TYPE_HPP
