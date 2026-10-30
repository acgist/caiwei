#include "caiwei/log.hpp"
#include "caiwei/env.hpp"
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

volatile std::sig_atomic_t caiwei_signal = 0;

void on_signal(int signal) {
    CW_LOG_D("收到信号: %d", signal);
    caiwei_signal = signal;
}

void install_signal_handler() {
#if CAIWEI_OS_WIN
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);
#else
    std::signal(SIGHUP,  on_signal);
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);
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
            success.store(caiwei::rest::open());
        } catch (...) {
            success.store(false);
        }
        running.store(true, std::memory_order_release);
    });
    while (caiwei_signal == 0 && running.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    if (caiwei_signal != 0) {
        caiwei::rest::stop();
    }
    if (server_thread.joinable()) {
        server_thread.join();
    }
#endif
    caiwei::stop();
    return success ? 0 : 1;
}
