/**
 * 日志
 */
#ifndef CAIWEI_LOG_HPP
#define CAIWEI_LOG_HPP

#include <cinttypes>
#include <source_location>

namespace caiwei {
namespace log    {

enum class Level {

    D,
    I,
    W,
    E,

};

extern void log(Level level, const std::source_location& loc, const char* fmt, ...);

extern void set_log_level(Level level);

} // namespace log
} // namespace caiwei

#define LOG_DEBUG(fmt, ...) caiwei::log::log(caiwei::log::Level::D, std::source_location::current(), fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  caiwei::log::log(caiwei::log::Level::I, std::source_location::current(), fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  caiwei::log::log(caiwei::log::Level::W, std::source_location::current(), fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) caiwei::log::log(caiwei::log::Level::E, std::source_location::current(), fmt, ##__VA_ARGS__)

#endif // CAIWEI_LOG_HPP
