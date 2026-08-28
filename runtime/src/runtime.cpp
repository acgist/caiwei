#include "caiwei/runtime.hpp"

std::shared_ptr<caiwei::runtime::Runtime> caiwei::runtime::get_runtime(caiwei::context::Type type) {
    std::shared_ptr<Runtime> runtime = nullptr;
    #ifdef ENABLE_CAIWEI_RUNTIME_MNN
    runtime = caiwei::runtime::get_mnn_runtime(type);
    if (runtime != nullptr) {
        return runtime;
    }
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_CANN
    runtime = caiwei::runtime::get_cann_runtime(type);
    if (runtime != nullptr) {
        return runtime;
    }
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_RKNN
    runtime = caiwei::runtime::get_rknn_runtime(type);
    if (runtime != nullptr) {
        return runtime;
    }
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_LLAMACPP
    runtime = caiwei::runtime::get_llamacpp_runtime(type);
    if (runtime != nullptr) {
        return runtime;
    }
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
    runtime = caiwei::runtime::get_onnxruntime_runtime(type);
    if (runtime != nullptr) {
        return runtime;
    }
    #endif
    return runtime;
}
