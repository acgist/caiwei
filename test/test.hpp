#ifndef CAIWEI_TEST_TEST_HPP
#define CAIWEI_TEST_TEST_HPP

#include "player.hpp"

#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/caiwei.hpp"
#include "caiwei/manager.hpp"
#include "caiwei/text_tool.hpp"
#include "caiwei/audio_tool.hpp"
#include "caiwei/image_tool.hpp"
#include "caiwei/media_tool.hpp"

#include <chrono>
#include <cassert>
#include <numeric>
#include <iostream>

#if CAIWEI_OS_WIN
#include <Windows.h>
#endif

#ifndef CAIWEI_FOR_EACH
#define CAIWEI_FOR_EACH(caiwei_length)                          \
    const size_t caiwei_count = (caiwei_length);                \
    const auto caiwei_for_a = std::chrono::steady_clock::now(); \
    for (size_t caiwei_index = 0; caiwei_index < caiwei_count; ++caiwei_index) {
#endif

#ifndef CAIWEI_FOR_EACH_END
#define CAIWEI_FOR_EACH_END                                                                                                   \
    }                                                                                                                         \
    const auto    loc = std::source_location::current();                                                                      \
    const auto    caiwei_for_z  = std::chrono::steady_clock::now();                                                           \
    const int64_t caiwei_for_ms = std::chrono::duration_cast<std::chrono::milliseconds>(caiwei_for_z - caiwei_for_a).count(); \
    const size_t  caiwei_per_ms = caiwei_count == 0 ? 0 : static_cast<size_t>(caiwei_for_ms) / caiwei_count;                  \
    CW_LOG_I(                                                                                                                 \
        "耗时: %6" PRId64 " / %4zu = %4zu ms = %32s",                                                                         \
        caiwei_for_ms,                                                                                                        \
        caiwei_count,                                                                                                         \
        caiwei_per_ms,                                                                                                        \
        loc.function_name()                                                                                                   \
    );
#endif

namespace caiwei {
namespace test   {

inline void caiwei_wait_enter() {
    CW_LOG_I("回车继续");
    std::cin.get();
}

inline void draw_rect(
    uint8_t* dst,
    const int src_w,
    const int src_h,
    const int x,
    const int y,
    const int w,
    const int h,
    const int channels  = 3,
    const uint8_t color = 255
) {
    if (dst == nullptr || src_w <= 0 || src_h <= 0 || channels <= 0) {
        return;
    }
    const int x1 = std::clamp(x,     0, src_w - 1);
    const int y1 = std::clamp(y,     0, src_h - 1);
    const int x2 = std::clamp(x + w, 0, src_w    );
    const int y2 = std::clamp(y + h, 0, src_h    );
    if (x2 <= x1 || y2 <= y1) {
        return;
    }
    for (int i = x1; i < x2; ++i) {
        const size_t offset = (static_cast<size_t>(y1) * src_w + i) * channels;
        std::fill_n(dst + offset, channels, color);
    }
    for (int i = x1; i < x2; ++i) {
        const size_t offset = (static_cast<size_t>(y2 - 1) * src_w + i) * channels;
        std::fill_n(dst + offset, channels, color);
    }
    for (int i = y1 + 1; i < y2 - 1; ++i) {
        const size_t offset = (static_cast<size_t>(i) * src_w + x1) * channels;
        std::fill_n(dst + offset, channels, color);
    }
    for (int i = y1 + 1; i < y2 - 1; ++i) {
        const size_t offset = (static_cast<size_t>(i) * src_w + (x2 - 1)) * channels;
        std::fill_n(dst + offset, channels, color);
    }
}

inline void init_test() {
    #if CAIWEI_OS_WIN
    SetConsoleOutputCP(CP_UTF8);
    #endif
    CW_LOG_I("测试开始");
    caiwei::init();
}

inline void stop_test() {
    caiwei::stop();
    CW_LOG_I("测试结束");
}

} // test
} // caiwei

#endif // CAIWEI_TEST_TEST_HPP
