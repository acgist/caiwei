#include <cstdlib>

inline void init_test() {
    #if OS_WIN
    system("chcp 65001");
    #endif
}
