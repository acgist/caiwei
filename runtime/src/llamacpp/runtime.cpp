#include "caiwei/log.hpp"
#include "caiwei/runtime.hpp"

caiwei::runtime::LlamaCPPRuntime::LlamaCPPRuntime() : Runtime(caiwei::runtime::Type::LLAMACPP) {
    CW_LOG_I("LLlamaCPPRuntime init");
}

caiwei::runtime::LlamaCPPRuntime::~LlamaCPPRuntime() {
    CW_LOG_I("LLlamaCPPRuntime stop");
}

template<>
std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    return std::make_shared<caiwei::runtime::LlamaCPPRuntime>();
}
