/**
 * 图片工具
 */
#ifndef CAIWEI_MEDIA_IMAGE_TOOL_HPP
#define CAIWEI_MEDIA_IMAGE_TOOL_HPP

#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>

#include "stb/stb_image.h"
#include "stb/stb_image_resize2.h"

#ifdef ENABLE_CAIWEI_TEST
#include "stb/stb_image_write.h"
#endif

#include "caiwei/image_data.hpp"
#include "caiwei/media_data.hpp"

namespace caiwei {
namespace image  {

const uint8_t DEFAULT_PADDING = 114;

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

inline void resize(const int src_w, const int src_h, const int w, const int h, int& dst_w, int& dst_h, int& pad_w, int& pad_h, float& scale) {
    scale = std::min(static_cast<float>(w) / src_w, static_cast<float>(h) / src_h);
    dst_w = static_cast<int>(std::lround(src_w * scale)) / 2 * 2;
    dst_h = static_cast<int>(std::lround(src_h * scale)) / 2 * 2;
    pad_w = (w - dst_w) / 2;
    pad_h = (h - dst_h) / 2;
}

inline void resize(const uint8_t* src, uint8_t* dst, const int src_w, const int src_h, const int dst_w, const int dst_h) {
    if (src_h == dst_h && src_w == dst_w) {
        std::copy_n(src, static_cast<size_t>(src_w) * src_h * 3, dst);
    } else {
        stbir_resize_uint8_linear(src, src_w, src_h, 0, dst, dst_w, dst_h, 0, STBIR_RGB);
    }
}

inline void resize(const float* src, float* dst, const int src_w, const int src_h, const int dst_w, const int dst_h) {
    if (src_h == dst_h && src_w == dst_w) {
        std::copy_n(src, static_cast<size_t>(src_w) * src_h, dst);
    } else {
        stbir_resize_float_linear(src, src_w, src_h, 0, dst, dst_w, dst_h, 0, STBIR_1CHANNEL);
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

inline float sigmoid(float x) {
    if (x > 88.0F) {
        return 1.0F;
    };
    if (x < -88.0F) {
        return 0.0F;
    };
    return 1.0F / (1.0F + std::exp(-x));
}

inline void coeff_to_mask(const float* coeff, const int box_x, const int box_y, const int box_w, const int box_h, const float* proto_data, const int proto_c, const int proto_h, const int proto_w, const int proto_pad_h, const int proto_pad_w, float* mask) {
    const int length = proto_h * proto_w;
    const int box_x2 = box_x + box_w;
    const int box_y2 = box_y + box_h;
    int j = 0;
    for (int i = 0; i < length; ++i) {
        const int y = i / proto_w;
        const int x = i % proto_w;
        if (y < box_y || y >= box_y2 || x < box_x || x >= box_x2) {
            continue;
        }
        float sum = 0.0F;
        for (int c = 0; c < proto_c; ++c) {
            sum += coeff[c] * proto_data[c * length + i];
        }
        mask[j++] = sigmoid(sum);
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
        size_t top_index = all_index[0];
        const T& top = boxes[top_index];
        ret_index.push_back(top_index);
        std::vector<size_t> rest_index;
        for (size_t i = 1; i < all_index.size(); ++i) {
            size_t cur_index = all_index[i];
            const T& cur = boxes[cur_index];
            if (cur.class_id != top.class_id) {
                rest_index.push_back(cur_index);
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
                rest_index.push_back(cur_index);
            }
        }
        all_index.swap(rest_index);
    }
    return ret_index;
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
inline std::vector<std::pair<uint32_t, T>> top_k(const T* score, const uint32_t size, const uint32_t top_k, const T confidence_threshold) {
    if (size == 0 || top_k == 0 || top_k > size) {
        return {};
    }
    std::vector<std::pair<uint32_t, T>> ret;
    ret.reserve(size);
    for (uint32_t i = 0; i < size; ++i) {
        if (score[i] >= confidence_threshold) {
            ret.emplace_back(i, score[i]);
        }
    }
    if (ret.size() <= top_k) {
        std::sort(ret.begin(), ret.end(), [](const std::pair<uint32_t, T>& a, const std::pair<uint32_t, T>& z) {
            return a.second > z.second;
        });
        return ret;
    }
    std::partial_sort(ret.begin(), ret.begin() + top_k, ret.end(), [](const std::pair<uint32_t, T>& a, const std::pair<uint32_t, T>& z) {
        return a.second > z.second;
    });
    ret.resize(top_k);
    return ret;
}

template <typename T>
inline void transpose(const T* src, T* dst, const int C, const int N) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            dst[n * C + c] = src[c * N + n];
        }
    }
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

} // namespace image
} // namespace caiwei

#endif // CAIWEI_MEDIA_IMAGE_TOOL_HPP
