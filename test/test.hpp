#include "caiwei/env.hpp"
#include "caiwei/log.hpp"

#include <chrono>
#include <cstdlib>
#include <source_location>

#define CAIWEI_FOR_EACH(caiwei_length)                                                \
    auto caiwei_for_a = std::chrono::system_clock::now();                             \
    for (uint32_t caiwei_index = 0; caiwei_index < (caiwei_length); ++caiwei_index) { \

#define CAIWEI_FOR_EACH_END                                                                                          \
    }                                                                                                                \
    auto caiwei_for_z  = std::chrono::system_clock::now();                                                           \
    auto caiwei_for_ms = std::chrono::duration_cast<std::chrono::milliseconds>(caiwei_for_z - caiwei_for_a).count(); \
    auto loc = std::source_location::current();                                                                      \
    LOG_INFO("耗时: %6d ms = %48s", caiwei_for_ms, loc.function_name());                                             \
