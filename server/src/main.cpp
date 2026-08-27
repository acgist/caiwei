#include "caiwei/env.hpp"
#ifdef ENABLE_CAIWEI_REST
#include "caiwei/rest.hpp"
#endif
#include "caiwei/caiwei.hpp"

#if OS_WIN
#include <cstdlib>
#endif
#include <cstring>

int main(int argc, char* argv[]) {
#if OS_WIN
    system("chcp 65001");
#endif
    if (argc >= 2 && std::strcmp(argv[1], "-h") == 0) {
        caiwei::env::print_all_env();
        return 0;
    }
    caiwei::init();
#ifdef ENABLE_CAIWEI_REST
    caiwei::rest::open();
#endif
    caiwei::stop();
    return 0;
}
