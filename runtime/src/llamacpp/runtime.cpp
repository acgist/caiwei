#include "caiwei/runtime/llamacpp.hpp"

template<>
std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    return std::make_shared<caiwei::runtime::LlamaCPPRuntime>();
}

caiwei::runtime::LlamaCPPRuntime::LlamaCPPRuntime() : Runtime(caiwei::runtime::Type::LLAMACPP) {
    CW_LOG_I("LlamaCPPRuntime init");
}

caiwei::runtime::LlamaCPPRuntime::~LlamaCPPRuntime() {
    CW_LOG_I("LlamaCPPRuntime stop");
}
