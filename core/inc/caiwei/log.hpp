/**
 * 日志模块
 */
#ifndef CAIWEI_LOG_HPP
#define CAIWEI_LOG_HPP

#include <cinttypes>
#include <source_location>

namespace caiwei {
namespace log    {

enum class Level {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

extern void log(Level level, const std::source_location& loc, const char* fmt, ...);

extern void set_log_level(Level level);

} // namespace log
} // namespace caiwei

#define LOG_DEBUG(fmt, ...) caiwei::log::log(caiwei::log::Level::DEBUG, std::source_location::current(), fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  caiwei::log::log(caiwei::log::Level::INFO,  std::source_location::current(), fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  caiwei::log::log(caiwei::log::Level::WARN,  std::source_location::current(), fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) caiwei::log::log(caiwei::log::Level::ERROR, std::source_location::current(), fmt, ##__VA_ARGS__)

#endif // CAIWEI_LOG_HPP
