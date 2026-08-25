#include "caiwei/log.hpp"
#include "caiwei/caiwei.hpp"

#include <cstdlib>

inline void init_test() {
    #if OS_WIN
    system("chcp 65001");
    #endif
    LOG_INFO("测试开始");
    caiwei::init();
}

inline void stop_test() {
    caiwei::stop();
    LOG_INFO("测试结束");
}
