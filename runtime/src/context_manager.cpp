#include "caiwei/context_manager.hpp"

std::mutex caiwei::context::runtime_mutex;
std::map<caiwei::runtime::Type, std::shared_ptr<caiwei::runtime::Runtime>> caiwei::context::runtime_map;

void caiwei::context::free() {
    std::lock_guard<std::mutex> runtime_lock(runtime_mutex);
    runtime_map.clear();
}
// TODO 空闲资源释放：最后保留min