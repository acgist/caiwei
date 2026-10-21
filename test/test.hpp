#include "player.hpp"

#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/caiwei.hpp"
#include "caiwei/manager.hpp"
#include "caiwei/image_tool.hpp"
#include "caiwei/media_tool.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <source_location>

#define CAIWEI_FOR_EACH(caiwei_length)                    \
    auto caiwei_count = (caiwei_length);                  \
    auto caiwei_for_a = std::chrono::steady_clock::now(); \
    for (uint32_t caiwei_index = 0; caiwei_index < (caiwei_length); ++caiwei_index) {

#define CAIWEI_FOR_EACH_END                                                                                          \
    }                                                                                                                \
    auto caiwei_for_z  = std::chrono::steady_clock::now();                                                           \
    auto caiwei_for_ms = std::chrono::duration_cast<std::chrono::milliseconds>(caiwei_for_z - caiwei_for_a).count(); \
    auto loc = std::source_location::current();                                                                      \
    CW_LOG_I("耗时: %6d / %4d = %4d ms = %48s", caiwei_for_ms, caiwei_count, caiwei_for_ms / caiwei_count, loc.function_name());

inline void caiwei_wait_enter() {
    std::printf("回车继续\n");
    std::cin.get();
}

namespace caiwei {
namespace test   {

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
