#include "player.hpp"

#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/media.hpp"
#include "caiwei/caiwei.hpp"
#include "caiwei/image_tool.hpp"
#include "caiwei/context_manager.hpp"

#include <chrono>
#include <cstdlib>
#include <source_location>

#define CAIWEI_FOR_EACH(caiwei_length)                    \
    auto caiwei_count = (caiwei_length);                  \
    auto caiwei_for_a = std::chrono::system_clock::now(); \
    for (uint32_t caiwei_index = 0; caiwei_index < (caiwei_length); ++caiwei_index) {

#define CAIWEI_FOR_EACH_END                                                                                          \
    }                                                                                                                \
    auto caiwei_for_z  = std::chrono::system_clock::now();                                                           \
    auto caiwei_for_ms = std::chrono::duration_cast<std::chrono::milliseconds>(caiwei_for_z - caiwei_for_a).count(); \
    auto loc = std::source_location::current();                                                                      \
    CW_LOG_I("耗时: %6d ms = %48s", caiwei_for_ms / caiwei_count, loc.function_name());

namespace caiwei {
namespace test   {

inline void draw_line(uint8_t* dst, const int src_w, const int src_h, int x0, int y0, const int x1, const int y1, const uint8_t color = 0, const int channels = 3) {
    int dx = std::abs(x1 - x0);
    int dy = std::abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int er = dx - dy;
    while (true) {
        std::fill_n(dst + (static_cast<size_t>(y0) * src_w + x0) * channels, channels, color);
        if (x0 == x1 && y0 == y1) {
            break;
        }
        int e2 = 2 * er;
        if (e2 > -dy) {
            er -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            er += dx;
            y0 += sy;
        }
    }
}

inline void draw_mask(uint8_t* src, const int src_w, const int src_h, float const* mask, const int mask_x, const int mask_y, const int mask_w, const int mask_h, float conf = 0.4F, const uint8_t color[3] = nullptr, const int channels = 3, const float alpha = 0.5F) {
    const uint8_t default_color[3] = { 0, 255, 0 };
    if (!color) {
        color = default_color;
    }
    const int pixel_size = src_w * src_h;
    const float beta = 1.0F - alpha;
    for (int index = 0; index < pixel_size; ++index) {
        int x = index % src_w;
        int y = index / src_w;
        if (x >= mask_x && x < mask_x + mask_w && y >= mask_y && y < mask_y + mask_h) {
            if (*mask > conf) {
                for (int c = 0; c < channels; ++c) {
                    float v = static_cast<float>(color[c]) * alpha + static_cast<float>(src[c]) * beta;
                    src[c] = static_cast<uint8_t>(std::clamp(v, 0.0F, 255.F));
                }
            }
            ++mask;
        }
        src += channels;
    }
}

inline void draw_rect(uint8_t* dst, const int src_w, const int src_h, const int x, const int y, const int w, const int h, const uint8_t color = 0, const int channels = 3) {
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

inline void draw_point(uint8_t* dst, const int src_w, const int src_h, const int x, const int y, const uint8_t color = 0, const int channels = 3) {
    size_t offset = static_cast<size_t>(y) * src_w * channels + static_cast<size_t>(x) * channels;
    std::fill_n(dst + offset, channels, color);
}

inline void init_test() {
    #if CAIWEI_OS_WIN
    system("chcp 65001");
    #endif
    CW_LOG_I("测试开始");
    caiwei::init();
}

inline void stop_test() {
    caiwei::stop();
    CW_LOG_I("测试结束");
}

}
}
