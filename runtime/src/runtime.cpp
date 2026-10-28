#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/runtime.hpp"

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN3
#include "rknn3/rknn3_api.h"
#endif
#ifdef ENABLE_CAIWEI_RUNTIME_LLAMACPP
#include "llama.h"

#include "mtmd.h"
#include "mtmd-helper.h"
#endif
#ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
#include "onnxruntime_cxx_api.h"
#endif

std::mutex caiwei::runtime::runtime_mutex;
std::map<caiwei::runtime::Type, std::shared_ptr<caiwei::runtime::Runtime>> caiwei::runtime::runtime_map;

static const char* get_type_name(caiwei::runtime::Type type) {
    switch (type) {
    case caiwei::runtime::Type::RKNN2:       return "RKNN2";
    case caiwei::runtime::Type::RKNN3:       return "RKNN3";
    case caiwei::runtime::Type::LLAMACPP:    return "LLAMACPP";
    case caiwei::runtime::Type::ONNXRUNTIME: return "ONNXRUNTIME";
    default: return "UNKNOWN";
    }
}

caiwei::runtime::Runtime::Runtime(int min_pool, int max_pool, int timeout, int keepalive, bool share, caiwei::runtime::Type type)
  : id(caiwei::env::id()),
    min_pool(min_pool),
    max_pool(max_pool),
    timeout(timeout),
    keepalive(keepalive),
    share(share),
    type(type) {
}

caiwei::runtime::Runtime::~Runtime() {
}

std::shared_ptr<caiwei::context::Context> caiwei::runtime::Runtime::get_context(const caiwei::context::ContextInfo* info) {
    if (info == nullptr) {
        return nullptr;
    }
    std::unique_lock<std::mutex> lock(mutex);
    auto iter = context_map.find(info->name);
    if (iter == context_map.end()) {
        iter = context_map.emplace(info->name, std::vector<std::shared_ptr<caiwei::context::Context>>()).first;
    }
    for (auto& context : iter->second) {
        if (this->share) {
            if (context->ref_count >= this->max_pool) {
                CW_LOG_W("获取共享context失败: %s = %d = %s", info->name.c_str(), context->ref_count.load(), context->id.c_str());
                return nullptr;
            }
            CW_LOG_D("返回共享context: %s = %d = %s", info->name.c_str(), iter->second.size(), context->id.c_str());
            context->ref_count++;
            return context;
        }
        if (context->usage) {
            continue;
        }
        CW_LOG_D("返回独占context: %s = %d = %s", info->name.c_str(), iter->second.size(), context->id.c_str());
        context->usage = true;
        context->ref_count++;
        return context;
    }
    std::shared_ptr<caiwei::context::Context> ptr{ nullptr };
    if (iter->second.size() < this->max_pool) {
        switch (info->type) {
        case caiwei::context::Type::CLS: ptr = this->get_cls_context(info); break;
        case caiwei::context::Type::DET: ptr = this->get_det_context(info); break;
        case caiwei::context::Type::ASR: ptr = this->get_asr_context(info); break;
        case caiwei::context::Type::LLM: ptr = this->get_llm_context(info); break;
        case caiwei::context::Type::VLM: ptr = this->get_vlm_context(info); break;
        case caiwei::context::Type::EMBEDDING: ptr = this->get_embedding_context(info); break;
        case caiwei::context::Type::RERANKING: ptr = this->get_reranking_context(info); break;
        default: break;
        }
        if (ptr == nullptr) {
            CW_LOG_D("新建context失败: %s = %s", get_type_name(this->type), info->name.c_str());
            return nullptr;
        }
        if (!ptr->load()) {
            CW_LOG_W("加载context失败: %s = %s", get_type_name(this->type), info->name.c_str());
            return nullptr;
        }
        CW_LOG_I("新建context: %s = %d = %s", info->name.c_str(), iter->second.size(), ptr->id.c_str());
        iter->second.push_back(ptr);
        CW_LOG_D("返回独占context: %s = %d = %s", info->name.c_str(), iter->second.size(), ptr->id.c_str());
        ptr->usage = true;
        ptr->ref_count++;
        return ptr;
    } else {
        this->condition.wait_for(lock, std::chrono::seconds(this->timeout), [&iter]() {
            return std::any_of(iter->second.begin(), iter->second.end(), [](auto& context) {
                return !context->usage;
            });
        });
        for (auto& context : iter->second) {
            if (context->usage) {
                continue;
            }
            CW_LOG_D("返回独占context: %s = %d = %s", info->name.c_str(), iter->second.size(), context->id.c_str());
            context->usage = true;
            context->ref_count++;
            return context;
        }
        CW_LOG_W("获取独占context超时: %s", info->name.c_str());
        return nullptr;
    }
}

void caiwei::runtime::Runtime::put_context(std::shared_ptr<caiwei::context::Context> context) {
    if (context == nullptr) {
        return;
    }
    CW_LOG_D("回收context: %s", context->id.c_str());
    std::lock_guard<std::mutex> lock(mutex);
    context->usage = false;
    context->ref_count--;
    this->condition.notify_all();
}

size_t caiwei::runtime::Runtime::optimize() {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto map_iter = this->context_map.begin(); map_iter != this->context_map.end();) {
        if (map_iter->second.size() <= this->min_pool) {
            ++map_iter;
            continue;
        }
        for (auto iter = map_iter->second.begin(); iter != map_iter->second.end();) {
            if ((*iter)->usage) {
                ++iter;
                continue;
            }
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - (*iter)->last_run_time);
            if (duration.count() > this->keepalive) {
                CW_LOG_I("优化释放context: %s = %s", map_iter->first.c_str(), (*iter)->id.c_str());
                iter = map_iter->second.erase(iter);
            } else {
                ++iter;
            }
        }
        if (map_iter->second.empty()) {
            map_iter = this->context_map.erase(map_iter);
        } else {
            ++map_iter;
        }
    }
    return this->context_map.size();
}

std::shared_ptr<caiwei::context::ClsContext> caiwei::runtime::Runtime::get_cls_context(const caiwei::context::ContextInfo* info) {
    return nullptr;
}

std::shared_ptr<caiwei::context::DetContext> caiwei::runtime::Runtime::get_det_context(const caiwei::context::ContextInfo* info) {
    return nullptr;
}

std::shared_ptr<caiwei::context::ASRContext> caiwei::runtime::Runtime::get_asr_context(const caiwei::context::ContextInfo* info) {
    return nullptr;
}

std::shared_ptr<caiwei::context::LLMContext> caiwei::runtime::Runtime::get_llm_context(const caiwei::context::ContextInfo* info) {
    return nullptr;
}

std::shared_ptr<caiwei::context::VLMContext> caiwei::runtime::Runtime::get_vlm_context(const caiwei::context::ContextInfo* info) {
    return nullptr;
}

std::shared_ptr<caiwei::context::EmbeddingContext> caiwei::runtime::Runtime::get_embedding_context(const caiwei::context::ContextInfo* info) {
    return nullptr;
}

std::shared_ptr<caiwei::context::RerankingContext> caiwei::runtime::Runtime::get_reranking_context(const caiwei::context::ContextInfo* info) {
    return nullptr;
}

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
static void init_rknn2();
#endif
#ifdef ENABLE_CAIWEI_RUNTIME_RKNN3
static void init_rknn3();
#endif
#ifdef ENABLE_CAIWEI_RUNTIME_LLAMACPP
static void init_llamacpp();
#endif
#ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
static void init_onnxruntime();
#endif

void caiwei::runtime::init() {
    #ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
    init_rknn2();
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_RKNN3
    init_rknn3();
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_LLAMACPP
    init_llamacpp();
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
    init_onnxruntime();
    #endif
}

void caiwei::runtime::stop() {
}

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
static void init_rknn2() {
}
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN3
static void init_rknn3() {
    rknn3_devices devs;
    int ret = rknn3_find_devices(&devs);
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_I("没有找到RKNN3设备");
        return;
    }
    for (int i = 0; i < devs.n_devices; i++) {
        CW_LOG_I("RKNN3设备 %d: 类型=%s, ID=%s", i, devs.devices[i].type, devs.devices[i].id);
    }
}
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_LLAMACPP
static void caiwei_llama_log(enum ggml_log_level level, const char* text, void* /* user_data */) {
    size_t n = std::strlen(text);
    if (n == 0) {
        return;
    }
    n--;
    if (level == GGML_LOG_LEVEL_DEBUG) {
        // CW_LOG_D("%.*s", n, text);
    } else if (level == GGML_LOG_LEVEL_INFO) {
        CW_LOG_I("%.*s", n, text);
    } else if (level == GGML_LOG_LEVEL_WARN) {
        CW_LOG_W("%.*s", n, text);
    } else if (level == GGML_LOG_LEVEL_ERROR) {
        CW_LOG_E("%.*s", n, text);
    } else {
        // -
    }
}

static void init_llamacpp() {
    llama_log_set(caiwei_llama_log, nullptr);
    mtmd_log_set(caiwei_llama_log, nullptr);
    mtmd_helper_log_set(caiwei_llama_log, nullptr);
    ggml_backend_load_all();
    CW_LOG_I("llama.cpp version: %s", ggml_version());
    const size_t backend_count = ggml_backend_reg_count();
    for (size_t i = 0; i < backend_count; i++) {
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
