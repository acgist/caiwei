#include "caiwei/log.hpp"

#include <mutex>
#include <vector>
#include <cstdarg>

static std::mutex mutex;
#ifdef CAIWEI_LOG_DEBUG
static caiwei::log::Level level = caiwei::log::Level::CAIWEI_LOG_DEBUG;
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
    constexpr int BUFFER_SIZE = 2048;
    static std::vector<char> buffer(BUFFER_SIZE);
    va_list ap;
    va_start(ap, fmt);
    int ret = vsnprintf(buffer.data(), BUFFER_SIZE, fmt, ap);
    va_end(ap);
    if (ret < 0) {
        return;
    }
    if (ret >= BUFFER_SIZE) {
        buffer[BUFFER_SIZE - 4] = '.';
        buffer[BUFFER_SIZE - 3] = '.';
        buffer[BUFFER_SIZE - 2] = '.';
        buffer[BUFFER_SIZE - 1] = '\0';
    }
    std::string filename = loc.file_name();
#if OS_WIN
    auto pos = filename.find_last_of('\\');
#else
    auto pos = filename.find_last_of('/');
#endif
    if (pos != std::string::npos) {
        filename = filename.substr(pos + 1);
    }
    std::printf("[%s] %-16s:%4d | %s\n", format_level(level), filename.c_str(), loc.line(), buffer.data()); 
}
