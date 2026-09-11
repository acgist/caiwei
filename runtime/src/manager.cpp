#include "caiwei/manager.hpp"

#include <mutex>
#include <thread>
#include <condition_variable>

static void runtime_optimize_timer();
static bool runtime_optimize_running = false;
static std::mutex runtime_optimize_mutex;
static std::thread runtime_optimize_thread;
static std::condition_variable runtime_optimize_cv;

void caiwei::manager::init() {
    runtime_optimize_timer();
}

void caiwei::manager::stop() {
    {
        std::lock_guard<std::mutex> lock(caiwei::runtime::runtime_mutex);
        caiwei::runtime::runtime_map.clear();
    }
    {
        std::lock_guard<std::mutex> lock(runtime_optimize_mutex);
        runtime_optimize_running = false;
        runtime_optimize_cv.notify_one();
    }
    if (runtime_optimize_thread.joinable()) {
        runtime_optimize_thread.join();
    }
}

static void runtime_optimize_timer() {
    runtime_optimize_running = true;
    runtime_optimize_thread = std::thread([]() {
        while (runtime_optimize_running) {
            std::unique_lock<std::mutex> lock(runtime_optimize_mutex);
            runtime_optimize_cv.wait_for(lock, std::chrono::seconds(30));
            if (!runtime_optimize_running) {
                break;
            }
            std::lock_guard<std::mutex> runtime_lock(caiwei::runtime::runtime_mutex);
            for (auto iter = caiwei::runtime::runtime_map.begin(); iter != caiwei::runtime::runtime_map.end();) {
                if (iter->second->optimize() == 0) {
                    CW_LOG_I("优化释放runtime: %s", iter->second->id.c_str());
                    iter = caiwei::runtime::runtime_map.erase(iter);
                } else {
                    ++iter;
                }
            }
        }
    });
}