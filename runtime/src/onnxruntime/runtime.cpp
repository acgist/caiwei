#include "caiwei/runtime.hpp"
#include "caiwei/runtime/onnxruntime.hpp"

template<>
std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    int min_pool  = caiwei::env::get_int("CAIWEI_ONNXRUNTIME_MIN_POOL");
    int max_pool  = caiwei::env::get_int("CAIWEI_ONNXRUNTIME_MAX_POOL");
    int timeout   = caiwei::env::get_int("CAIWEI_RUNTIME_TIMEOUT");
    int keepalive = caiwei::env::get_int("CAIWEI_RUNTIME_KEEPALIVE");
    return std::make_shared<caiwei::runtime::ONNXRuntimeRuntime>(min_pool, max_pool, timeout, keepalive);
}

caiwei::runtime::ONNXRuntimeRuntime::ONNXRuntimeRuntime(int min_pool, int max_pool, int timeout, int keepalive) : Runtime(min_pool, max_pool, timeout, keepalive, caiwei::runtime::Type::ONNXRUNTIME) {
    CW_LOG_I("ONNXRuntimeRuntime init");
    this->env = new Ort::Env(caiwei::context::onnxruntime_log_level, "caiwei");
}

caiwei::runtime::ONNXRuntimeRuntime::~ONNXRuntimeRuntime() {
    if (this->env != nullptr) {
        CW_LOG_I("ONNXRuntimeRuntime stop");
        delete this->env;
        this->env = nullptr;
    }
}

std::shared_ptr<caiwei::context::ClsContext> caiwei::runtime::ONNXRuntimeRuntime::get_cls_context(const caiwei::context::ContextInfo* info) {
    int c = caiwei::env::get_int("CAIWEI_CLS_C");
    int h = caiwei::env::get_int("CAIWEI_CLS_H");
    int w = caiwei::env::get_int("CAIWEI_CLS_W");
    int top_k = caiwei::env::get_int("CAIWEI_CLS_TOP_K");
    int class_size = caiwei::env::get_int("CAIWEI_CLS_CLASS_SIZE");
    float confidence_threshold = caiwei::env::get_float("CAIWEI_CLS_CONFIDENCE_THRESHOLD");
    if (!std::filesystem::exists(info->path)) {
        CW_LOG_W("ClsContext模型无效: %s", info->path.c_str());
        return nullptr;
    }
    return std::make_shared<caiwei::context::ClsONNXRuntimeContext>(info->path, c, h, w, top_k, class_size, confidence_threshold, this->env, this);
}

std::shared_ptr<caiwei::context::DetContext> caiwei::runtime::ONNXRuntimeRuntime::get_det_context(const caiwei::context::ContextInfo* info) {
    int c = caiwei::env::get_int("CAIWEI_DET_C");
    int h = caiwei::env::get_int("CAIWEI_DET_H");
    int w = caiwei::env::get_int("CAIWEI_DET_W");
    int class_size = caiwei::env::get_int("CAIWEI_DET_CLASS_SIZE");
    float iou_threshold = caiwei::env::get_float("CAIWEI_DET_IOU_THRESHOLD");
    float confidence_threshold = caiwei::env::get_float("CAIWEI_DET_CONFIDENCE_THRESHOLD");
    if (!std::filesystem::exists(info->path)) {
        CW_LOG_W("DetContext模型无效: %s", info->path.c_str());
        return nullptr;
    }
    return std::make_shared<caiwei::context::DetONNXRuntimeContext>(info->path, c, h, w, class_size, iou_threshold, confidence_threshold, this->env, this);
}

std::shared_ptr<caiwei::context::SegContext> caiwei::runtime::ONNXRuntimeRuntime::get_seg_context(const caiwei::context::ContextInfo* info) {
    int c = caiwei::env::get_int("CAIWEI_SEG_C");
    int h = caiwei::env::get_int("CAIWEI_SEG_H");
    int w = caiwei::env::get_int("CAIWEI_SEG_W");
    int class_size = caiwei::env::get_int("CAIWEI_SEG_CLASS_SIZE");
    float iou_threshold = caiwei::env::get_float("CAIWEI_SEG_IOU_THRESHOLD");
    float confidence_threshold = caiwei::env::get_float("CAIWEI_SEG_CONFIDENCE_THRESHOLD");
    if (!std::filesystem::exists(info->path)) {
        CW_LOG_W("SegContext模型无效: %s", info->path.c_str());
        return nullptr;
    }
    return std::make_shared<caiwei::context::SegONNXRuntimeContext>(info->path, c, h, w, class_size, iou_threshold, confidence_threshold, this->env, this);
}

std::shared_ptr<caiwei::context::PoseContext> caiwei::runtime::ONNXRuntimeRuntime::get_pose_context(const caiwei::context::ContextInfo* info) {
    int c = caiwei::env::get_int("CAIWEI_POSE_C");
    int h = caiwei::env::get_int("CAIWEI_POSE_H");
    int w = caiwei::env::get_int("CAIWEI_POSE_W");
    int class_size = caiwei::env::get_int("CAIWEI_POSE_CLASS_SIZE");
    float iou_threshold = caiwei::env::get_float("CAIWEI_POSE_IOU_THRESHOLD");
    float confidence_threshold = caiwei::env::get_float("CAIWEI_POSE_CONFIDENCE_THRESHOLD");
    if (!std::filesystem::exists(info->path)) {
        CW_LOG_W("PoseContext模型无效: %s", info->path.c_str());
        return nullptr;
    }
    return std::make_shared<caiwei::context::PoseONNXRuntimeContext>(info->path, c, h, w, class_size, iou_threshold, confidence_threshold, this->env, this);
}
