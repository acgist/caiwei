#include "caiwei/log.hpp"
#include "caiwei/runtime.hpp"

#include "onnxruntime_cxx_api.h"

caiwei::runtime::Runtime::Runtime(caiwei::runtime::Type type) : type(type) {
}

caiwei::runtime::Runtime::~Runtime() {
}

uint32_t caiwei::runtime::Runtime::ref() {
    auto old = this->ref_count.fetch_add(1);
    return old + 1;
}

uint32_t caiwei::runtime::Runtime::unref() {
    auto old = this->ref_count.fetch_sub(1);
    return old - 1;
}

static void print_onnx_runtime_info();

void caiwei::runtime::init() {
    print_onnx_runtime_info();
}

void caiwei::runtime::stop() {
}

void print_onnx_runtime_info() {
    LOG_INFO("ONNXRuntime Version: %s", Ort::GetVersionString().c_str());
    LOG_INFO("ONNXRuntime Build Info: %s", Ort::GetBuildInfoString().c_str());
    for(const auto& provider : Ort::GetAvailableProviders()) {
        LOG_INFO("ONNXRuntime Provider: %s", provider.c_str());
    }
}
