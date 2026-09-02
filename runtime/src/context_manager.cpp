#include "caiwei/context_manager.hpp"

std::mutex caiwei::context::context_mutex;
std::mutex caiwei::context::runtime_mutex;
std::map<std::string,           std::shared_ptr<caiwei::context::Context>> caiwei::context::context_map;
std::map<caiwei::runtime::Type, std::shared_ptr<caiwei::runtime::Runtime>> caiwei::context::runtime_map;
