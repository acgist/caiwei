#include "caiwei/log.hpp"
#include "caiwei/runtime.hpp"

caiwei::runtime::LlamaCPPRuntime::LlamaCPPRuntime() : Runtime(caiwei::runtime::Type::LLAMACPP) {
    LOG_INFO("LLlamaCPPRuntime init");
}

caiwei::runtime::LlamaCPPRuntime::~LlamaCPPRuntime() {
    LOG_INFO("LLlamaCPPRuntime stop");
}

template<>
std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    return std::make_shared<caiwei::runtime::LlamaCPPRuntime>();
}
