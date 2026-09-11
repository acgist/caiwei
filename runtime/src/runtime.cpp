#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/runtime.hpp"

#ifdef ENABLE_CAIWEI_RUNTIME_LLAMACPP
#include "llama.h"
#endif
#ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
#include "onnxruntime_cxx_api.h"
#endif

std::mutex caiwei::runtime::runtime_mutex;
std::map<caiwei::runtime::Type, std::shared_ptr<caiwei::runtime::Runtime>> caiwei::runtime::runtime_map;

caiwei::runtime::Runtime::Runtime(int min_pool, int max_pool, int timeout, int keepalive, caiwei::runtime::Type type)
  : id(caiwei::env::id()),
    min_pool(min_pool),
    max_pool(max_pool),
    timeout(timeout),
    keepalive(keepalive), type(type) {
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
        if (context->share) {
            CW_LOG_D("返回共享context: %s = %d = %s", info->name.c_str(), iter->second.size(), context->id.c_str());
            return context;
        }
        if (context->usage) {
            continue;
        }
        context->usage = true;
        CW_LOG_D("返回独占context: %s = %d = %s", info->name.c_str(), iter->second.size(), context->id.c_str());
        return context;
    }
    std::shared_ptr<caiwei::context::Context> ptr{ nullptr };
    if (iter->second.size() < this->max_pool) {
        switch (info->type) {
        case caiwei::context::Type::CLS : ptr = this->get_cls_context (info); break;
        case caiwei::context::Type::DET : ptr = this->get_det_context (info); break;
        case caiwei::context::Type::SEG : ptr = this->get_seg_context (info); break;
        case caiwei::context::Type::POSE: ptr = this->get_pose_context(info); break;
        case caiwei::context::Type::ASR : ptr = this->get_asr_context (info); break;
        case caiwei::context::Type::LLM : ptr = this->get_llm_context (info); break;
        case caiwei::context::Type::VLM : ptr = this->get_vlm_context (info); break;
        case caiwei::context::Type::EMBEDDING: ptr = this->get_embedding_context(info); break;
        case caiwei::context::Type::RERANKING: ptr = this->get_reranking_context(info); break;
        default: break;
        }
        if (ptr == nullptr) {
            return nullptr;
        }
        if (!ptr->load()) {
            CW_LOG_E("加载context %s 失败", info->name.c_str());
            return nullptr;
        }
        iter->second.push_back(ptr);
        CW_LOG_I("新建context: %s = %d = %s", info->name.c_str(), iter->second.size(), ptr->id.c_str());
        ptr->usage = true;
        CW_LOG_D("返回独占context: %s = %d = %s", info->name.c_str(), iter->second.size(), ptr->id.c_str());
        return ptr;
    } else {
        this->cv.wait_for(lock, std::chrono::seconds(this->timeout), [&iter]() {
            return std::any_of(iter->second.begin(), iter->second.end(), [](auto& context) {
                return !context->usage;
            });
        });
        for (auto& context : iter->second) {
            if (context->share) {
                CW_LOG_D("返回共享context: %s = %d = %s", info->name.c_str(), iter->second.size(), context->id.c_str());
                return context;
            }
            if (context->usage) {
                continue;
            }
            context->usage = true;
            CW_LOG_D("返回独占context: %s = %d = %s", info->name.c_str(), iter->second.size(), context->id.c_str());
            return context;
        }
        return nullptr;
    }
}

void caiwei::runtime::Runtime::put_context(std::shared_ptr<caiwei::context::Context> context) {
    if (context == nullptr) {
        return;
    }
    CW_LOG_D("释放context: %s", context->id.c_str());
    std::lock_guard<std::mutex> lock(mutex);
    context->usage = false;
    this->cv.notify_all();
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
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - (*iter)->last_run_time);
            if (duration.count() > this->keepalive) {
                iter = map_iter->second.erase(iter);
                CW_LOG_I("优化释放context: %s = %s", map_iter->first.c_str(), (*iter)->id.c_str());
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

std::shared_ptr<caiwei::context::SegContext> caiwei::runtime::Runtime::get_seg_context(const caiwei::context::ContextInfo* info) {
    return nullptr;
}

std::shared_ptr<caiwei::context::PoseContext> caiwei::runtime::Runtime::get_pose_context(const caiwei::context::ContextInfo* info) {
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
