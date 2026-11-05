#include "caiwei/manager.hpp"

std::atomic<bool> caiwei::manager::available{ false };

void caiwei::manager::init() {
    caiwei::manager::available.store(true, std::memory_order_release);
}

void caiwei::manager::stop() {
    caiwei::manager::available.store(false, std::memory_order_release);
}
