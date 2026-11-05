#include "caiwei/manager.hpp"

#include <mutex>

bool caiwei::manager::available = false;

static std::mutex mutex;

void caiwei::manager::init() {
    std::lock_guard<std::mutex> lock(mutex);
    caiwei::manager::available = true;
}

void caiwei::manager::stop() {
    std::lock_guard<std::mutex> lock(mutex);
    caiwei::manager::available = false;
}
