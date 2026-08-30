/**
 * 数据转换
 */
#ifndef CAIWEI_TRANSFORM_HPP
#define CAIWEI_TRANSFORM_HPP

#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>

#include "stb/stb_image.h"
#include "stb/stb_image_resize2.h"

namespace caiwei    {
namespace transform {

template<typename T>
inline void crop(const T* src, const int src_w, const int src_h, T* dst, const int x, const int y, const int w, const int h, const int channels = 3) {
    const size_t row_size = static_cast<size_t>(w) * channels;
    for (int row = 0; row < h; ++row) {
        const size_t src_offset = (static_cast<size_t>(y + row) * src_w + x) * channels;
        const size_t dst_offset = static_cast<size_t>(row) * w * channels;
        const T* src_row = src + src_offset;
              T* dst_row = dst + dst_offset;
        std::copy_n(src_row, row_size, dst_row);
    }
}

inline void draw(uint8_t* dst, const int src_w, const int src_h, const int x, const int y, const int w, const int h, const uint8_t color = 0, const int channels = 3) {
    int x1 = std::max(0, x);
    int y1 = std::max(0, y);
    int x2 = std::min(src_w, x + w);
    int y2 = std::min(src_h, y + h);
    for (int i = x1; i < x2; ++i) {
        const size_t offset = (static_cast<size_t>(y1) * src_w + i) * channels;
        std::fill_n(dst + offset, channels, color);
    }
    for (int i = x1; i < x2; ++i) {
        size_t offset = (static_cast<size_t>(y2 - 1) * src_w + i) * channels;
        std::fill_n(dst + offset, channels, color);
    }
    for (int i = y1 + 1; i < y2 - 1; ++i) {
        size_t offset = (static_cast<size_t>(i) * src_w + x1) * channels;
        std::fill_n(dst + offset, channels, color);
    }
    for (int i = y1 + 1; i < y2 - 1; ++i) {
        size_t offset = (static_cast<size_t>(i) * src_w + (x2 - 1)) * channels;
        std::fill_n(dst + offset, channels, color);
    }
}

inline void resize(const int src_w, const int src_h, const int w, const int h, int& dst_w, int& dst_h, int& pad_w, int& pad_h, float& scale) {
    scale = std::min(static_cast<float>(w) / src_w, static_cast<float>(h) / src_h);
    dst_w = static_cast<int>(std::lround(src_w * scale)) / 2 * 2;
    dst_h = static_cast<int>(std::lround(src_h * scale)) / 2 * 2;
    pad_w = (w - dst_w) / 2;
    pad_h = (h - dst_h) / 2;
}

inline void resize(const uint8_t* src, uint8_t* dst, const int src_w, const int src_h, const int dst_w, const int dst_h, const int channels = 3) {
    if (src_h == dst_h && src_w == dst_w) {
        std::copy_n(src, static_cast<size_t>(src_w) * src_h * channels, dst);
    } else {
        stbir_resize_uint8_linear(src, src_w, src_h, 0, dst, dst_w, dst_h, 0, STBIR_RGB);
    }
}

inline void padding(const uint8_t* src, uint8_t* pad, const int src_w, const int src_h, const int pad_w, const int pad_h, const int w, const int h, const int channels = 3) {
    if (pad_w == 0 && pad_h == 0) {
        std::copy_n(src, static_cast<size_t>(src_w) * src_h * channels, pad);
    } else {
        for (int y = 0; y < src_h; ++y) {
            const size_t dst_offset = (static_cast<size_t>(pad_h + y) * w + pad_w) * static_cast<size_t>(channels);
            const size_t src_offset = static_cast<size_t>(y) * static_cast<size_t>(src_w) * static_cast<size_t>(channels);
            std::copy_n(src + src_offset, static_cast<size_t>(src_w) * channels, pad + dst_offset);
        }
    }
}

template<typename T>
inline std::vector<T> nms_boxes(std::vector<T> boxes, const float iou_threshold) {
    if (boxes.empty()) {
        return {};
    }
    std::sort(boxes.begin(), boxes.end(), [](const T& a, const T& z) {
        return a.score > z.score;
    });
    std::vector<T> result;
    while (!boxes.empty()) {
        T top = boxes[0];
        result.push_back(top);
        std::vector<T> rest;
        for (size_t i = 1; i < boxes.size(); ++i) {
            const T& cur = boxes[i];
            if (cur.class_id != top.class_id) {
                rest.push_back(cur);
                continue;
            }
            float inter_x1 = std::max(top.x1, cur.x1);
            float inter_y1 = std::max(top.y1, cur.y1);
            float inter_x2 = std::min(top.x2, cur.x2);
            float inter_y2 = std::min(top.y2, cur.y2);
            float w = std::max(0.0F, inter_x2 - inter_x1);
            float h = std::max(0.0F, inter_y2 - inter_y1);
            float inter = w * h;
            float area_top = (top.x2 - top.x1) * (top.y2 - top.y1);
            float area_cur = (cur.x2 - cur.x1) * (cur.y2 - cur.y1);
            float union_area = area_top + area_cur - inter;
            float iou = union_area <= 0.0F ? 0.0F : inter / union_area;
            if (iou < iou_threshold) {
                rest.push_back(cur);
            }
        }
        boxes.swap(rest);
    }
    return result;
}

template <typename T>
inline void min_loc(const T* score, const int size, T& min_score, int& min_index) {
    auto [min, max] = std::minmax_element(score, score + size);
    min_score = *min;
    min_index = std::distance(score, min);
}

template <typename T>
inline void max_loc(const T* score, const int size, T& max_score, int& max_index) {
    auto [min, max] = std::minmax_element(score, score + size);
    max_score = *max;
    max_index = std::distance(score, max);
}

template <typename T>
inline void min_max_loc(const T* score, const int size, T& min_score, int& min_index, T& max_score, int& max_index) {
    auto [min, max] = std::minmax_element(score, score + size);
    min_score = *min;
    min_index = std::distance(score, min);
    max_score = *max;
    max_index = std::distance(score, max);
}

template <typename T>
inline void transpose(const T* src, T* dst, const int C, const int N) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            dst[n * C + c] = src[c * N + n];
        }
    }
}

inline void f32_to_f16(const float   * src, const uint32_t size, uint16_t* dst);
inline void f16_to_f32(const uint16_t* src, const uint32_t size, float   * dst);

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

template<typename T>
void hwc_to_chw(const T* hwc, T* chw, const int H, const int W, const int C) {
    const size_t plane_size = static_cast<size_t>(H) * W;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t pixel_idx = static_cast<size_t>(y) * W + x;
            const T*     hwc_pixel = hwc + pixel_idx * C;
            for (int c = 0; c < C; ++c) {
                chw[c * plane_size + pixel_idx] = hwc_pixel[c];
            }
        }
    }
}

template<typename T>
void chw_to_hwc(const T* chw, T* hwc, const int H, const int W, const int C) {
    const size_t plane_size = static_cast<size_t>(H) * W;
    for (int c = 0; c < C; ++c) {
        const T* src_plane = chw + c * plane_size;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const size_t chw_idx = static_cast<size_t>(y) * W + x;
                const size_t hwc_idx = chw_idx * C + c;
                hwc[hwc_idx] = src_plane[chw_idx];
            }
        }
    }
}

} // namespace transform
} // namespace caiwei
#endif // CAIWEI_TRANSFORM_HPP
