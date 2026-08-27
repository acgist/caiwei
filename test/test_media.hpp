#include "test.hpp"

#include "caiwei/media.hpp"

inline void init_test() {
    #if OS_WIN
    system("chcp 65001");
    #endif
    LOG_INFO("测试开始");
    caiwei::media::init();
}

inline void stop_test() {
    caiwei::media::stop();
    LOG_INFO("测试结束");
}
