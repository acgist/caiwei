#include "caiwei/log.hpp"

#include <array>
#include <ctime>
#include <string>
#include <cstdarg>
#include <cstring>
#include <cinttypes>

#if CAIWEI_OS_WIN
#include <Windows.h>
#elif CAIWEI_OS_UNIX
#include <unistd.h>
#include <sys/syscall.h>
#else
#endif

#ifdef CAIWEI_LOG_LEVEL
static caiwei::log::Level level = caiwei::log::Level::CAIWEI_LOG_LEVEL;
#else
static caiwei::log::Level level = caiwei::log::Level::D;
#endif

inline constexpr int MESSAGE_SIZE = 2048;

inline uint64_t thread_id() {
#if CAIWEI_OS_WIN
    return static_cast<uint64_t>(GetCurrentThreadId());
#elif CAIWEI_OS_UNIX
    return static_cast<uint64_t>(syscall(SYS_gettid));
#else
    return 0;
#endif
}

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
    // 日志时间
    thread_local char        time_buf[16] = {};
    thread_local std::time_t time_old     = 0;
    time_t time_now = std::time(nullptr);
    if (time_now != time_old) {
        time_old = time_now;
        struct tm time_tm{};
        #if CAIWEI_OS_WIN
        localtime_s(&time_tm, &time_now);
        #else
        localtime_r(&time_now, &time_tm);
        #endif
        std::strftime(time_buf, sizeof(time_buf), "%m-%d %H:%M:%S", &time_tm);
    }
    // 文件名称
    const char* file = loc.file_name();
    #if CAIWEI_OS_WIN
    const char* name = std::strrchr(file, '\\');
    #else
    const char* name = std::strrchr(file, '/');
    #endif
    if (name == nullptr) {
        name = file;
    } else {
        name++;
    }
    // 日志内容
    thread_local std::array<char, MESSAGE_SIZE> message{};
    va_list ap;
    va_start(ap, fmt);
    int ret = vsnprintf(message.data(), message.size(), fmt, ap);
    va_end(ap);
    if (ret < 0) {
        message[0] = 'E';
        message[1] = 'R';
        message[2] = 'R';
        message[3] = 'O';
        message[4] = 'R';
        message[5] = '\0';
    } else if (static_cast<size_t>(ret) >= message.size()) {
        message[message.size() - 4] = '.';
        message[message.size() - 3] = '.';
        message[message.size() - 2] = '.';
        message[message.size() - 1] = '\0';
    }
    #if CAIWEI_DEBUG
    if (level == caiwei::log::Level::D || level == caiwei::log::Level::I) {
    #else
    if (level == caiwei::log::Level::I) {
    #endif
        std::fprintf(stdout, "[%s] %s [%" PRIu64 "] %-16s:%4u | %s\n", format_level(level), time_buf, thread_id(), name, loc.line(), message.data()); 
    } else {
        std::fprintf(stderr, "[%s] %s [%" PRIu64 "] %-16s:%4u | %s\n", format_level(level), time_buf, thread_id(), name, loc.line(), message.data()); 
    }
}
