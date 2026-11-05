#include "caiwei/env.hpp"
#include "caiwei/caiwei.hpp"
#include "caiwei/context.hpp"
#include "caiwei/runtime.hpp"
#include "caiwei/manager.hpp"
#include "caiwei/media_tool.hpp"

void caiwei::init() {
    CW_LOG_I("系统启动");
    caiwei::env::init();
    caiwei::media::init();
    caiwei::runtime::init();
    caiwei::context::init();
    caiwei::manager::init();
}

void caiwei::stop() {
    caiwei::manager::stop();
    caiwei::context::stop();
    caiwei::runtime::stop();
    caiwei::media::stop();
    CW_LOG_I("系统关闭");
}
