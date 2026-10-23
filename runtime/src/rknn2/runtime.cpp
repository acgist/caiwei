#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/runtime.hpp"
#include "caiwei/runtime/rknn2.hpp"

#include <filesystem>

template<>
std::shared_ptr<caiwei::runtime::RKNN2Runtime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    if (type != caiwei::runtime::Type::RKNN2) {
        CW_LOG_W("类型无效: %d", type);
        return nullptr;
    }
    const int  min_pool  = caiwei::env::get_int("CAIWEI_RKNN2_MIN_POOL");
    const int  max_pool  = caiwei::env::get_int("CAIWEI_RKNN2_MAX_POOL");
    const int  timeout   = caiwei::env::get_int("CAIWEI_RUNTIME_TIMEOUT");
    const int  keepalive = caiwei::env::get_int("CAIWEI_RUNTIME_KEEPALIVE");
    const bool share     = caiwei::env::get_bool("CAIWEI_RKNN2_SHARE");
    return std::make_shared<caiwei::runtime::RKNN2Runtime>(min_pool, max_pool, timeout, keepalive, share);
}

caiwei::runtime::RKNN2Runtime::RKNN2Runtime(int min_pool, int max_pool, int timeout, int keepalive, bool share) : Runtime(min_pool, max_pool, timeout, keepalive, share, caiwei::runtime::Type::RKNN2) {
    CW_LOG_I("创建RKNN2Runtime");
}

caiwei::runtime::RKNN2Runtime::~RKNN2Runtime() {
    CW_LOG_I("释放RKNN2Runtime");
}

std::shared_ptr<caiwei::context::ClsContext> caiwei::runtime::RKNN2Runtime::get_cls_context(const caiwei::context::ContextInfo* info) {
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
    return std::make_shared<caiwei::context::ClsRKNN2Context>(info->path, c, h, w, class_size, top_k, confidence_threshold, this);
}

std::shared_ptr<caiwei::context::DetContext> caiwei::runtime::RKNN2Runtime::get_det_context(const caiwei::context::ContextInfo* info) {
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
    return std::make_shared<caiwei::context::DetRKNN2Context>(info->path, c, h, w, class_size, iou_threshold, confidence_threshold, this);
}
