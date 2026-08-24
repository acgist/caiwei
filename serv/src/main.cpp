#include "caiwei/log.hpp"

#include <cstdlib>

int main(int argc, char* argv[]) {
    #if OS_WIN
    system("chcp 65001");
    #endif
    LOG_DEBUG("audio frame pts=%lld sample=%d", 10000LL, 48000);
    LOG_ERROR("open stream 测试 failed code=%d", -100);
    LOG_ERROR("open stream 测试 failed code=%d", -100);
    LOG_ERROR("open stream 测试 failed code=%s", "测试");
    return 0;
}