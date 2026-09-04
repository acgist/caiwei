#include "caiwei/log.hpp"
#include "caiwei/runtime.hpp"

#ifdef ENABLE_CAIWEI_RUNTIME_LLAMACPP
#include "llama.h"
#endif
#ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
#include "onnxruntime_cxx_api.h"
#endif

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

#ifdef ENABLE_CAIWEI_RUNTIME_LLAMACPP
static void init_llamacpp();
#endif
#ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
static void init_onnxruntime();
#endif

void caiwei::runtime::init() {
    #ifdef ENABLE_CAIWEI_RUNTIME_LLAMACPP
    init_llamacpp();
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
    init_onnxruntime();
    #endif
}

void caiwei::runtime::stop() {
}

#ifdef ENABLE_CAIWEI_RUNTIME_LLAMACPP
static void init_llamacpp() {
    llama_log_set([](enum ggml_log_level level, const char* text, void* /* user_data */) {
        size_t n = std::strlen(text);
        if (n == 0) {
            return;
        }
        n--;
        if (level == GGML_LOG_LEVEL_DEBUG) {
            CW_LOG_D("%.*s", n, text);
        } else if (level == GGML_LOG_LEVEL_INFO) {
            CW_LOG_I("%.*s", n, text);
        } else if (level == GGML_LOG_LEVEL_WARN) {
            CW_LOG_W("%.*s", n, text);
        } else if (level == GGML_LOG_LEVEL_ERROR) {
            CW_LOG_E("%.*s", n, text);
        } else {
            // -
        }
    }, nullptr);
    ggml_backend_load_all();
    CW_LOG_I("llama.cpp version: %s", ggml_version());
    const int backend_count = ggml_backend_reg_count();
    for (int i = 0; i < backend_count; i++) {
        ggml_backend_reg* reg = ggml_backend_reg_get(i);
        if (!reg) {
            continue;
        }
        const char* name = ggml_backend_reg_name(reg);
        CW_LOG_I("llama.cpp backend name: %s", name);
    }
}
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
static void init_onnxruntime() {
    CW_LOG_I("ONNXRuntime Version: %s", Ort::GetVersionString().c_str());
    CW_LOG_I("ONNXRuntime Build Info: %s", Ort::GetBuildInfoString().c_str());
    for(const auto& provider : Ort::GetAvailableProviders()) {
        CW_LOG_I("ONNXRuntime Provider: %s", provider.c_str());
    }
}
#endif
