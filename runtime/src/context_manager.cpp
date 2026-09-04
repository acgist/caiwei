#include "caiwei/context_manager.hpp"

std::mutex caiwei::context::context_mutex;
std::mutex caiwei::context::runtime_mutex;
std::map<std::string,           std::shared_ptr<caiwei::context::Context>> caiwei::context::context_map;
std::map<caiwei::runtime::Type, std::shared_ptr<caiwei::runtime::Runtime>> caiwei::context::runtime_map;

void caiwei::context::free() {
    std::lock_guard<std::mutex> context_lock(context_mutex);
    context_map.clear();
    std::lock_guard<std::mutex> runtime_lock(runtime_mutex);
    runtime_map.clear();
}
