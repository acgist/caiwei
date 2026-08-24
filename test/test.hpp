#include "caiwei/caiwei.hpp"

#include <cstdlib>

inline void init_test() {
    #if OS_WIN
    system("chcp 65001");
    #endif
    caiwei::init();
}

inline void stop_test() {
    caiwei::stop();
}
