#include "test.hpp"

#include "caiwei/media.hpp"
#include "caiwei/caiwei.hpp"
#include "caiwei/player.hpp"
#include "caiwei/context.hpp"
#include "caiwei/runtime.hpp"
#include "caiwei/transform.hpp"

inline void init_test() {
    #if OS_WIN
    system("chcp 65001");
    #endif
    CW_LOG_I("测试开始");
    caiwei::init();
}

inline void stop_test() {
    caiwei::stop();
    CW_LOG_I("测试结束");
}
