#include "onnxruntime.hpp"

caiwei::runtime::ONNXRuntimeRuntime::ONNXRuntimeRuntime() : Runtime(caiwei::runtime::Type::ONNXRUNTIME) {
    LOG_INFO("ONNXRuntimeRuntime init");
    this->env = new Ort::Env(caiwei::context::onnxruntime_log_level, "caiwei");
}

caiwei::runtime::ONNXRuntimeRuntime::~ONNXRuntimeRuntime() {
    if (this->env != nullptr) {
        LOG_INFO("ONNXRuntimeRuntime stop");
        delete this->env;
        this->env = nullptr;
    }
}

template<>
std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    return std::make_shared<caiwei::runtime::ONNXRuntimeRuntime>();
}
