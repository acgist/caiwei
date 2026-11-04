#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#if ENABLE_CAIWEI_REST
#include "caiwei/rest.hpp"
#endif
#include "caiwei/caiwei.hpp"

#if CAIWEI_OS_WIN
#include <Windows.h>
#endif
#include <atomic>
#include <thread>
#include <csignal>
#include <cstring>

static std::atomic<int> caiwei_signal{ 0 };

#if CAIWEI_OS_WIN
static BOOL WINAPI on_signal(DWORD type) {
    switch (type) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_LOGOFF_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        caiwei_signal = 1;
        return TRUE;
    default:
        return FALSE;
    }
}
#else
static void on_signal(int sig) {
    caiwei_signal = sig;
}
#endif

static void install_signal_handler() {
#if CAIWEI_OS_WIN
    SetConsoleCtrlHandler(on_signal, TRUE);
#else
    struct sigaction sa{};
    sa.sa_flags   = SA_RESTART;
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGHUP,  &sa, nullptr);
    sigaction(SIGINT,  &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
#endif
}

int main(int argc, char* argv[]) {
#if CAIWEI_OS_WIN
    SetConsoleOutputCP(CP_UTF8);
#endif
    if (argc >= 2 && std::strcmp(argv[1], "?") == 0) {
        caiwei::env::print_all_env();
        return 0;
    }
    install_signal_handler();
    caiwei::init();
    if (caiwei_signal != 0) {
        caiwei::stop();
        return 0;
    }
    std::atomic<bool> running{ true };
    std::atomic<bool> success{ true };
#if ENABLE_CAIWEI_REST
    std::thread server_thread([&running, &success]() {
        try {
            success.store(caiwei::rest::open(), std::memory_order_release);
        } catch (...) {
            CW_LOG_E("注册REST服务异常");
            success.store(false, std::memory_order_release);
        }
        running.store(false, std::memory_order_release);
    });
    while (caiwei_signal == 0 && running.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    if (running.load(std::memory_order_acquire)) {
        caiwei::rest::stop();
    }
    if (server_thread.joinable()) {
        server_thread.join();
    }
#endif
    caiwei::stop();
    return success.load(std::memory_order_acquire) ? 0 : 1;
}
