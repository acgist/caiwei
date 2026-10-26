#ifndef CAIWEI_RUNTIME_IMAGE_TOOL_HPP
#define CAIWEI_RUNTIME_IMAGE_TOOL_HPP

#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>

#include "stb/stb_image.h"
#include "stb/stb_image_resize2.h"

#ifdef ENABLE_CAIWEI_TEST
#include "stb/stb_image_write.h"
#endif

namespace caiwei {
namespace image  {

inline constexpr uint8_t DEFAULT_PADDING = 114;

inline void resize(
    const int src_w,
    const int src_h,
    const int w,
    const int h,
    int& dst_w,
    int& dst_h,
    int& pad_w,
    int& pad_h,
    float& scale
) {
    scale = std::min(static_cast<float>(w) / src_w, static_cast<float>(h) / src_h);
    dst_w = static_cast<int>(std::lround(src_w * scale)) / 2 * 2;
    dst_h = static_cast<int>(std::lround(src_h * scale)) / 2 * 2;
    pad_w = (w - dst_w) / 2;
    pad_h = (h - dst_h) / 2;
}

inline void resize(
    const uint8_t* __restrict src,
          uint8_t* __restrict dst,
    const int src_w,
    const int src_h,
    const int dst_w,
    const int dst_h,
    const int channels = 3
) {
    if (src_h == dst_h && src_w == dst_w) {
        std::copy_n(src, static_cast<size_t>(src_w) * src_h * channels, dst);
    } else {
        stbir_resize_uint8_linear(src, src_w, src_h, 0, dst, dst_w, dst_h, 0, STBIR_RGB);
    }
}

inline void padding(
    const uint8_t* __restrict src,
          uint8_t* __restrict pad,
    const int src_w,
    const int src_h,
    const int pad_w,
    const int pad_h,
    const int w,
    const int /* h */,
    const int channels = 3
) {
    if (pad_w == 0 && pad_h == 0) {
        std::copy_n(src, static_cast<size_t>(src_w) * src_h * channels, pad);
    } else {
        const size_t stride = static_cast<size_t>(src_w) * channels;
        for (int y = 0; y < src_h; ++y) {
            const size_t src_offset = stride * y;
            const size_t dst_offset = (static_cast<size_t>(pad_h + y) * w + pad_w) * channels;
            std::copy_n(src + src_offset, stride, pad + dst_offset);
        }
    }
}

template<typename T>
inline std::vector<size_t> nms_boxes(const std::vector<T>& boxes, const float iou_threshold) {
    if (boxes.empty()) {
        return {};
    }
    std::vector<size_t> all_index;
    std::vector<size_t> ret_index;
    all_index.reserve(boxes.size());
    ret_index.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        all_index.push_back(i);
    }
    std::sort(all_index.begin(), all_index.end(), [&](size_t a, size_t z) {
        return boxes[a].score > boxes[z].score;
    });
    while (!all_index.empty()) {
        const size_t top_index = all_index[0];
        const T& top = boxes[top_index];
        ret_index.push_back(top_index);
        std::vector<size_t> tmp_index;
        for (size_t i = 1; i < all_index.size(); ++i) {
            const size_t cur_index = all_index[i];
            const T& cur = boxes[cur_index];
            if (cur.class_id != top.class_id) {
                tmp_index.push_back(cur_index);
                continue;
            }
            const float inter_x1 = std::max(top.x1, cur.x1);
            const float inter_y1 = std::max(top.y1, cur.y1);
            const float inter_x2 = std::min(top.x2, cur.x2);
            const float inter_y2 = std::min(top.y2, cur.y2);
            const float w = std::max(0.0F, inter_x2 - inter_x1);
            const float h = std::max(0.0F, inter_y2 - inter_y1);
            const float area_inter = w * h;
            const float area_top = (top.x2 - top.x1) * (top.y2 - top.y1);
            const float area_cur = (cur.x2 - cur.x1) * (cur.y2 - cur.y1);
            const float area_union = area_top + area_cur - area_inter;
            const float iou = area_union <= 0.0F ? 0.0F : area_inter / area_union;
            if (iou < iou_threshold) {
                tmp_index.push_back(cur_index);
            }
        }
        all_index.swap(tmp_index);
    }
    return ret_index;
}

template <typename T>
inline void min_loc(const T* score, const int size, T& min_score, int& min_index) {
    auto min = std::min_element(score, score + size);
    min_score = *min;
    min_index = std::distance(score, min);
}

template <typename T>
inline void max_loc(const T* score, const int size, T& max_score, int& max_index) {
    auto max = std::max_element(score, score + size);
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

template <typename T, typename R>
inline std::vector<R> top_k(const T* score, const uint32_t size, const uint32_t top_k, const T confidence_threshold) {
    if (size == 0 || top_k == 0) {
        return {};
    }
    std::vector<R> ret;
    ret.reserve(size);
    for (int i = 0; i < size; ++i) {
        if (score[i] >= confidence_threshold) {
            ret.push_back({
                .class_id = i,
                .score    = score[i]
            });
        }
    }
    if (ret.size() <= top_k) {
        std::sort(ret.begin(), ret.end(), [](const R& a, const R& z) {
            return a.score > z.score;
        });
        return ret;
    }
    std::partial_sort(ret.begin(), ret.begin() + top_k, ret.end(), [](const R& a, const R& z) {
        return a.score > z.score;
    });
    ret.resize(top_k);
    return ret;
}

template <typename T>
inline void transpose(const T* __restrict src, T* __restrict dst, const int C, const int N) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            dst[n * C + c] = src[c * N + n];
        }
    }
}

template<typename T>
void hwc_to_chw(const T* __restrict hwc, T* __restrict chw, const int H, const int W, const int C) {
    const size_t plane_size = static_cast<size_t>(H) * W;
    for (int y = 0; y < H; ++y) {
        const size_t stride = static_cast<size_t>(y) * W;
        for (int x = 0; x < W; ++x) {
            const size_t pixel_idx = stride + x;
            const T*     pixel_hwc = hwc + pixel_idx * C;
            #if defined(__GNUC__)
            #pragma GCC unroll 3
            #endif
            for (int c = 0; c < C; ++c) {
                chw[c * plane_size + pixel_idx] = pixel_hwc[c];
            }
        }
    }
}

template<typename T>
void chw_to_hwc(const T* __restrict chw, T* __restrict hwc, const int H, const int W, const int C) {
    const size_t plane_size = static_cast<size_t>(H) * W;
    for (int c = 0; c < C; ++c) {
        const T* src_plane = chw + c * plane_size;
        for (int y = 0; y < H; ++y) {
            const size_t stride = static_cast<size_t>(y) * W;
            for (int x = 0; x < W; ++x) {
                const size_t chw_idx = stride + x;
                const size_t hwc_idx = chw_idx * C + c;
                hwc[hwc_idx] = src_plane[chw_idx];
            }
        }
    }
}

inline void hwc_to_chw_i8_f32(const uint8_t* __restrict hwc, float* __restrict chw, const int H, const int W, const int C,const float scale) {
    const float inv_scale = 1.0F / scale;
    const size_t plane_size = static_cast<size_t>(H) * W;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t   pixel_idx = static_cast<size_t>(y) * W + x;
            const uint8_t* hwc_pixel = hwc + pixel_idx * C;
            #if defined(__GNUC__)
            #pragma GCC unroll 3
            #endif
            for (int c = 0; c < C; ++c) {
                chw[c * plane_size + pixel_idx] = static_cast<float>(hwc_pixel[c]) * inv_scale;
            }
        }
    }
}

} // image
} // caiwei

#endif // CAIWEI_RUNTIME_IMAGE_TOOL_HPP
