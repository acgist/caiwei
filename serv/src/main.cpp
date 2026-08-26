#include "caiwei/rest.hpp"
#include "caiwei/caiwei.hpp"

#if OS_WIN
#include <cstdlib>
#endif

int main(int argc, char* argv[]) {
#if OS_WIN
    system("chcp 65001");
#endif
    caiwei::init();
    caiwei::rest::open();
    caiwei::stop();
    return 0;
}
