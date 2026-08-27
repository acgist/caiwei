#include "caiwei/log.hpp"
#include "caiwei/caiwei.hpp"

#include <chrono>
#include <cstdlib>

#define CAIWEI_FOR_EACH(caiwei_length)                                                \
    auto caiwei_for_a = std::chrono::system_clock::now();                             \
    for (uint32_t caiwei_index = 0; caiwei_index < (caiwei_length); ++caiwei_index) { \

#define CAIWEI_FOR_EACH_END                                                                                          \
    }                                                                                                                \
    auto caiwei_for_z  = std::chrono::system_clock::now();                                                           \
    auto caiwei_for_ms = std::chrono::duration_cast<std::chrono::milliseconds>(caiwei_for_z - caiwei_for_a).count(); \
    LOG_INFO("耗时: %d ms", caiwei_for_ms);                                                                          \

inline void init_test() {
    #if OS_WIN
    system("chcp 65001");
    #endif
    LOG_INFO("测试开始");
    caiwei::init();
}

inline void stop_test() {
    caiwei::stop();
    LOG_INFO("测试结束");
}
