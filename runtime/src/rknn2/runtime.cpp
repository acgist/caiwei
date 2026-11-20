#include "caiwei/runtime.hpp"
#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/runtime/rknn2.hpp"

template<>
std::shared_ptr<caiwei::runtime::RKNN2Runtime> caiwei::runtime::get_runtime() {
    return std::make_shared<caiwei::runtime::RKNN2Runtime>(
        caiwei::env::get_int("CAIWEI_RKNN2_MIN_POOL"),
        caiwei::env::get_int("CAIWEI_RKNN2_MAX_POOL"),
        caiwei::env::get_int("CAIWEI_RUNTIME_TIMEOUT"),
        caiwei::env::get_int("CAIWEI_RUNTIME_KEEPALIVE"),
        caiwei::env::get_bool("CAIWEI_RKNN2_SHARE")
    );
}

caiwei::runtime::RKNN2Runtime::RKNN2Runtime(int min_pool, int max_pool, int timeout, int keepalive, bool share) :
    Runtime(min_pool, max_pool, timeout, keepalive, share, caiwei::runtime::Type::RKNN2) {
    CW_LOG_I("创建RKNN2Runtime");
}

caiwei::runtime::RKNN2Runtime::~RKNN2Runtime() {
    CW_LOG_I("释放RKNN2Runtime");
}

std::shared_ptr<caiwei::context::ClsContext>
caiwei::runtime::RKNN2Runtime::get_cls_context(const caiwei::context::ContextInfo* info) {
    if (!caiwei::context::check_model_path(info)) {
        return nullptr;
    }
    return std::make_shared<caiwei::context::ClsRKNN2Context>(
        info->path,
        caiwei::env::get_int("CAIWEI_CLS_C"),
        caiwei::env::get_int("CAIWEI_CLS_H"),
        caiwei::env::get_int("CAIWEI_CLS_W"),
        caiwei::env::get_int("CAIWEI_CLS_CLASS_SIZE"),
        caiwei::env::get_int("CAIWEI_CLS_TOP_K"),
        caiwei::env::get_float("CAIWEI_CLS_CONFIDENCE_THRESHOLD"),
        this
    );
}

std::shared_ptr<caiwei::context::DetContext>
caiwei::runtime::RKNN2Runtime::get_det_context(const caiwei::context::ContextInfo* info) {
    if (!caiwei::context::check_model_path(info)) {
        return nullptr;
    }
    return std::make_shared<caiwei::context::DetRKNN2Context>(
        info->path,
        caiwei::env::get_int("CAIWEI_DET_C"),
        caiwei::env::get_int("CAIWEI_DET_H"),
        caiwei::env::get_int("CAIWEI_DET_W"),
        caiwei::env::get_int("CAIWEI_DET_CLASS_SIZE"),
        caiwei::env::get_float("CAIWEI_DET_IOU_THRESHOLD"),
        caiwei::env::get_float("CAIWEI_DET_CONFIDENCE_THRESHOLD"),
        this
    );
}
