#include "test.hpp"

#include "caiwei/media.hpp"

inline void init_test() {
    #if CAIWEI_OS_WIN
    system("chcp 65001");
    #endif
    CW_LOG_I("测试开始");
    caiwei::media::init();
}

inline void stop_test() {
    caiwei::media::stop();
    CW_LOG_I("测试结束");
}
