/**
 * 日志模块
 */

#ifndef CAIWEI_LOGGER_HPP
#define CAIWEI_LOGGER_HPP

#define LOG_DEBUG() logger::debug()
#define LOG_INFO()  logger::info ()
#define LOG_ERROR() logger::error()
#define LOG_WARN()  logger::warn ()

namespace caiwei::logger {

extern void debug();
extern void info ();
extern void error();
extern void warn ();

}
#endif
