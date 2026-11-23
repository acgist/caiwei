#ifndef CAIWEI_RUNTIME_LOG_HPP
#define CAIWEI_RUNTIME_LOG_HPP

#include <cinttypes>
#include <source_location>

#if defined(__clang__) || defined(__GNUC__)
#define CW_LOG_FORMAT(...) __attribute__((format(printf, 3, 4)))
#else
#define CW_LOG_FORMAT(...)
#endif

namespace caiwei {
namespace log {

enum class Level {
    D,
    I,
    W,
    E,
};

CW_LOG_FORMAT()
void log(Level level, const std::source_location& loc, const char* fmt, ...);

void set_log_level(Level level);

} // namespace log
} // namespace caiwei

// clang-format off
#if defined(__cpp_va_opt) || (defined(_MSVC_TRADITIONAL) && (_MSVC_TRADITIONAL == 0))
#if CAIWEI_DEBUG
#define CW_LOG_D(fmt, ...) caiwei::log::log(caiwei::log::Level::D, std::source_location::current(), fmt __VA_OPT__(,) __VA_ARGS__)
#else
#define CW_LOG_D(fmt, ...) do { } while (0)
#endif
#define CW_LOG_I(fmt, ...) caiwei::log::log(caiwei::log::Level::I, std::source_location::current(), fmt __VA_OPT__(,) __VA_ARGS__)
#define CW_LOG_W(fmt, ...) caiwei::log::log(caiwei::log::Level::W, std::source_location::current(), fmt __VA_OPT__(,) __VA_ARGS__)
#define CW_LOG_E(fmt, ...) caiwei::log::log(caiwei::log::Level::E, std::source_location::current(), fmt __VA_OPT__(,) __VA_ARGS__)
#else
#if CAIWEI_DEBUG
#define CW_LOG_D(fmt, ...) caiwei::log::log(caiwei::log::Level::D, std::source_location::current(), fmt, ##__VA_ARGS__)
#else
#define CW_LOG_D(fmt, ...) do { } while (0)
#endif
#define CW_LOG_I(fmt, ...) caiwei::log::log(caiwei::log::Level::I, std::source_location::current(), fmt, ##__VA_ARGS__)
#define CW_LOG_W(fmt, ...) caiwei::log::log(caiwei::log::Level::W, std::source_location::current(), fmt, ##__VA_ARGS__)
#define CW_LOG_E(fmt, ...) caiwei::log::log(caiwei::log::Level::E, std::source_location::current(), fmt, ##__VA_ARGS__)
#endif
// clang-format on

#endif // CAIWEI_RUNTIME_LOG_HPP
