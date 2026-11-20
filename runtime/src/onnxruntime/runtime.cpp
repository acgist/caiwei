#include "caiwei/runtime.hpp"
#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/runtime/onnxruntime.hpp"

static void onnxruntime_log(
    void* /* param */,
    OrtLoggingLevel severity,
    const char* category,
    const char* logid,
    const char* code_location,
    const char* message
) {
    if (severity == ORT_LOGGING_LEVEL_VERBOSE) {
        CW_LOG_D("[%s:%s] %s - %s", logid, category, code_location, message);
    } else if (severity == ORT_LOGGING_LEVEL_INFO) {
        CW_LOG_I("[%s:%s] %s - %s", logid, category, code_location, message);
    } else if (severity == ORT_LOGGING_LEVEL_WARNING) {
        CW_LOG_W("[%s:%s] %s - %s", logid, category, code_location, message);
    } else if (severity == ORT_LOGGING_LEVEL_ERROR || severity == ORT_LOGGING_LEVEL_FATAL) {
        CW_LOG_E("[%s:%s] %s - %s", logid, category, code_location, message);
    } else {
        // -
    }
}

template<>
std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> caiwei::runtime::get_runtime() {
    return std::make_shared<caiwei::runtime::ONNXRuntimeRuntime>(
        caiwei::env::get_int("CAIWEI_ONNXRUNTIME_MIN_POOL"),
        caiwei::env::get_int("CAIWEI_ONNXRUNTIME_MAX_POOL"),
        caiwei::env::get_int("CAIWEI_RUNTIME_TIMEOUT"),
        caiwei::env::get_int("CAIWEI_RUNTIME_KEEPALIVE"),
        caiwei::env::get_bool("CAIWEI_ONNXRUNTIME_SHARE")
    );
}

caiwei::runtime::ONNXRuntimeRuntime::ONNXRuntimeRuntime(int min_pool, int max_pool, int timeout, int keepalive, bool share) :
    Runtime(min_pool, max_pool, timeout, keepalive, share, caiwei::runtime::Type::ONNXRUNTIME) {
    CW_LOG_I("创建ONNXRuntimeRuntime");
    this->env = new Ort::Env(caiwei::context::onnxruntime_log_level, "caiwei", onnxruntime_log, nullptr);
}

caiwei::runtime::ONNXRuntimeRuntime::~ONNXRuntimeRuntime() {
    CW_LOG_I("释放ONNXRuntimeRuntime");
    this->context_map.clear();
    if (this->env != nullptr) {
        CW_LOG_I("释放ONNXRuntimeRuntime: env");
        delete this->env;
        this->env = nullptr;
    }
}

std::shared_ptr<caiwei::context::ClsContext>
caiwei::runtime::ONNXRuntimeRuntime::get_cls_context(const caiwei::context::ContextInfo* info) {
    if (!caiwei::context::check_model_path(info)) {
        return nullptr;
    }
    return std::make_shared<caiwei::context::ClsONNXRuntimeContext>(
        info->path,
        caiwei::env::get_int("CAIWEI_CLS_C"),
        caiwei::env::get_int("CAIWEI_CLS_H"),
        caiwei::env::get_int("CAIWEI_CLS_W"),
        caiwei::env::get_int("CAIWEI_CLS_CLASS_SIZE"),
        caiwei::env::get_int("CAIWEI_CLS_TOP_K"),
        caiwei::env::get_float("CAIWEI_CLS_CONFIDENCE_THRESHOLD"),
        this->env,
        this
    );
}

std::shared_ptr<caiwei::context::DetContext>
caiwei::runtime::ONNXRuntimeRuntime::get_det_context(const caiwei::context::ContextInfo* info) {
    if (!caiwei::context::check_model_path(info)) {
        return nullptr;
    }
    return std::make_shared<caiwei::context::DetONNXRuntimeContext>(
        info->path,
        caiwei::env::get_int("CAIWEI_DET_C"),
        caiwei::env::get_int("CAIWEI_DET_H"),
        caiwei::env::get_int("CAIWEI_DET_W"),
        caiwei::env::get_int("CAIWEI_DET_CLASS_SIZE"),
        caiwei::env::get_float("CAIWEI_DET_IOU_THRESHOLD"),
        caiwei::env::get_float("CAIWEI_DET_CONFIDENCE_THRESHOLD"),
        this->env,
        this
    );
}
