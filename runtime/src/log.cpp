#include "caiwei/log.hpp"

#include <ctime>
#include <mutex>
#include <string>
#include <vector>
#include <cstdarg>

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
    default                   : return "W";
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
    static std::string time;
    time.resize(15);
    time_t time_tt = std::time(nullptr);
    struct tm time_tm;
    #if CAIWEI_OS_WIN
    localtime_s(&time_tm, &time_tt);
    #else
    localtime_r(&time_tt, &time_tm);
    #endif
    std::strftime(time.data(), time.size(), "%m-%d %H:%M:%S", &time_tm);
    // 文件名称
    std::string filename = loc.file_name();
#if CAIWEI_OS_WIN
    auto pos = filename.find_last_of('\\');
#else
    auto pos = filename.find_last_of('/');
#endif
    if (pos != std::string::npos) {
        filename = filename.substr(pos + 1);
    }
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
    std::printf("[%s] %s %-16s:%4d | %s\n", format_level(level), time.c_str(), filename.c_str(), loc.line(), message.data()); 
}
