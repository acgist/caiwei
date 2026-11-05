#include "caiwei/env.hpp"
#include "caiwei/caiwei.hpp"
#include "caiwei/context.hpp"
#include "caiwei/runtime.hpp"
#include "caiwei/manager.hpp"
#include "caiwei/media_tool.hpp"

bool caiwei::init() {
#if CAIWEI_OS_WIN
#else
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::setvbuf(stderr, nullptr, _IOLBF, 0);
#endif
    CW_LOG_I("系统启动");
    try {
        caiwei::env::init();
        caiwei::media::init();
        caiwei::runtime::init();
        caiwei::context::init();
        caiwei::manager::init();
        return true;
    } catch (const std::exception& e) {
        CW_LOG_E("系统启动异常: %s", e.what());
    } catch (...) {
        CW_LOG_E("系统启动异常");
    }
    return false;
}

void caiwei::stop() {
    caiwei::manager::stop();
    caiwei::context::stop();
    caiwei::runtime::stop();
    caiwei::media::stop();
    caiwei::env::stop();
    CW_LOG_I("系统关闭");
    std::fflush(stdout);
    std::fflush(stderr);
}
