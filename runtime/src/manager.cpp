#include "caiwei/manager.hpp"

static std::atomic<bool> available {false};

void caiwei::manager::init() {
    available.store(true, std::memory_order_release);
}

void caiwei::manager::stop() {
    available.store(false, std::memory_order_release);
}

bool caiwei::manager::is_available() {
    return available.load(std::memory_order_acquire);
}
