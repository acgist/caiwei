#include "caiwei/log.hpp"

#include <ctime>
#include <mutex>
#include <string>
#include <vector>
#include <cstdarg>
#include <string_view>

static std::mutex mutex;
#ifdef CAIWEI_LOG_LEVEL
static caiwei::log::Level level = caiwei::log::Level::CAIWEI_LOG_LEVEL;
#else
static caiwei::log::Level level = caiwei::log::Level::D;
#endif

inline const char* format_level(caiwei::log::Level level) {
    switch (level) {
    case caiwei::log::Level::D: return "D";
    case caiwei::log::Level::I: return "I";
    case caiwei::log::Level::W: return "W";
    case caiwei::log::Level::E: return "E";
    default                   : return "?";
    }
}

void caiwei::log::set_log_level(caiwei::log::Level level) {
    ::level = level;
}

void caiwei::log::log(caiwei::log::Level level, const std::source_location& loc, const char* fmt, ...) {
    if (level < ::level) {
        return;
    }
    std::lock_guard<std::mutex> lock(::mutex);
    // 日志时间
    static char        time_buf[16] = {};
    static std::time_t time_cached  = 0;
    time_t time_tt = std::time(nullptr);
    if (time_tt != time_cached) {
        time_cached = time_tt;
        struct tm time_tm;
        #if CAIWEI_OS_WIN
        localtime_s(&time_tm, &time_tt);
        #else
        localtime_r(&time_tt, &time_tm);
        #endif
        std::strftime(time_buf, sizeof(time_buf), "%m-%d %H:%M:%S", &time_tm);
    }
    // 文件名称
    const std::string_view file = loc.file_name();
    const auto             pos  = file.find_last_of("/\\");
    const std::string_view name = (pos == std::string_view::npos) ? file : file.substr(pos + 1);
    // 日志内容
    constexpr int MESSAGE_SIZE = 2048;
    static std::vector<char> message(MESSAGE_SIZE);
    va_list ap;
    va_start(ap, fmt);
    int ret = vsnprintf(message.data(), MESSAGE_SIZE, fmt, ap);
    va_end(ap);
    if (ret < 0) {
        return;
    }
    if (ret >= MESSAGE_SIZE) {
        message[MESSAGE_SIZE - 4] = '.';
        message[MESSAGE_SIZE - 3] = '.';
        message[MESSAGE_SIZE - 2] = '.';
        message[MESSAGE_SIZE - 1] = '\0';
    }
    if (level == caiwei::log::Level::E || level == caiwei::log::Level::W) {
        std::fprintf(stderr, "[%s] %s %-16.*s:%4d | %s\n", format_level(level), time_buf, static_cast<int>(name.size()), name.data(), loc.line(), message.data()); 
        std::fflush(stderr);
    } else {
        std::fprintf(stdout, "[%s] %s %-16.*s:%4d | %s\n", format_level(level), time_buf, static_cast<int>(name.size()), name.data(), loc.line(), message.data()); 
        std::fflush(stdout);
    }
}
