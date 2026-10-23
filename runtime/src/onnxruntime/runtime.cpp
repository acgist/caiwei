#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/runtime.hpp"
#include "caiwei/runtime/onnxruntime.hpp"

#include <filesystem>

template<>
std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    if (type != caiwei::runtime::Type::ONNXRUNTIME) {
        CW_LOG_W("类型无效: %d", type);
        return nullptr;
    }
    const int  min_pool  = caiwei::env::get_int("CAIWEI_ONNXRUNTIME_MIN_POOL");
    const int  max_pool  = caiwei::env::get_int("CAIWEI_ONNXRUNTIME_MAX_POOL");
    const int  timeout   = caiwei::env::get_int("CAIWEI_RUNTIME_TIMEOUT");
    const int  keepalive = caiwei::env::get_int("CAIWEI_RUNTIME_KEEPALIVE");
    const bool share     = caiwei::env::get_bool("CAIWEI_ONNXRUNTIME_SHARE");
    return std::make_shared<caiwei::runtime::ONNXRuntimeRuntime>(min_pool, max_pool, timeout, keepalive, share);
}

caiwei::runtime::ONNXRuntimeRuntime::ONNXRuntimeRuntime(int min_pool, int max_pool, int timeout, int keepalive, bool share) : Runtime(min_pool, max_pool, timeout, keepalive, share, caiwei::runtime::Type::ONNXRUNTIME) {
    CW_LOG_I("创建ONNXRuntimeRuntime");
    this->env = new Ort::Env(caiwei::context::onnxruntime_log_level, "caiwei");
}

caiwei::runtime::ONNXRuntimeRuntime::~ONNXRuntimeRuntime() {
    CW_LOG_I("释放ONNXRuntimeRuntime");
    if (this->env != nullptr) {
        delete this->env;
        this->env = nullptr;
    }
}

std::shared_ptr<caiwei::context::ClsContext> caiwei::runtime::ONNXRuntimeRuntime::get_cls_context(const caiwei::context::ContextInfo* info) {
    if (!std::filesystem::exists(info->path)) {
        CW_LOG_W("模型无效: %s", info->path.c_str());
        return nullptr;
    }
    int c = caiwei::env::get_int("CAIWEI_CLS_C");
    int h = caiwei::env::get_int("CAIWEI_CLS_H");
    int w = caiwei::env::get_int("CAIWEI_CLS_W");
    int class_size = caiwei::env::get_int("CAIWEI_CLS_CLASS_SIZE");
    int top_k = caiwei::env::get_int("CAIWEI_CLS_TOP_K");
    float confidence_threshold = caiwei::env::get_float("CAIWEI_CLS_CONFIDENCE_THRESHOLD");
    return std::make_shared<caiwei::context::ClsONNXRuntimeContext>(info->path, c, h, w, class_size, top_k, confidence_threshold, this->env, this);
}

std::shared_ptr<caiwei::context::DetContext> caiwei::runtime::ONNXRuntimeRuntime::get_det_context(const caiwei::context::ContextInfo* info) {
    if (!std::filesystem::exists(info->path)) {
        CW_LOG_W("模型无效: %s", info->path.c_str());
        return nullptr;
    }
    int c = caiwei::env::get_int("CAIWEI_DET_C");
    int h = caiwei::env::get_int("CAIWEI_DET_H");
    int w = caiwei::env::get_int("CAIWEI_DET_W");
    int class_size = caiwei::env::get_int("CAIWEI_DET_CLASS_SIZE");
    float iou_threshold = caiwei::env::get_float("CAIWEI_DET_IOU_THRESHOLD");
    float confidence_threshold = caiwei::env::get_float("CAIWEI_DET_CONFIDENCE_THRESHOLD");
    return std::make_shared<caiwei::context::DetONNXRuntimeContext>(info->path, c, h, w, class_size, iou_threshold, confidence_threshold, this->env, this);
}
