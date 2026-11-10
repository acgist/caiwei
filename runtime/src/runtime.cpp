#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/runtime.hpp"

#include <mutex>
#include <atomic>
#include <thread>
#include <shared_mutex>
#include <condition_variable>

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
#include "rknn2/rknn_api.h"
#endif
#ifdef ENABLE_CAIWEI_RUNTIME_RKNN3
#include "rknn3/rknn3_api.h"
#endif
#ifdef ENABLE_CAIWEI_RUNTIME_LLAMACPP
#include "mtmd-helper.h"
#endif
#ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
#include "onnxruntime_cxx_api.h"
#endif

std::shared_mutex caiwei::runtime::runtime_mutex;
std::map<caiwei::runtime::Type, std::shared_ptr<caiwei::runtime::Runtime>> caiwei::runtime::runtime_map;

static std::mutex              runtime_optimize_mutex;
static std::jthread            runtime_optimize_thread;
static std::condition_variable runtime_optimize_cv;

static void runtime_optimize_timer();

caiwei::runtime::Runtime::Runtime(
    int min_pool,
    int max_pool,
    int timeout,
    int keepalive,
    bool share,
    caiwei::runtime::Type type
) : id(caiwei::env::id()),
    min_pool(std::clamp(min_pool, 0, 128)),
    max_pool(std::clamp(max_pool, 1, 128)),
    timeout(std::clamp(timeout, 10, 60)),
    keepalive(std::clamp(keepalive, 60, 3600)),
    share(share),
    type(type) {
    CW_LOG_I("创建runtime: %s => %s", get_type_name(this->type), id.c_str());
}

caiwei::runtime::Runtime::~Runtime() {
    CW_LOG_I("释放runtime: %s => %s", get_type_name(this->type), id.c_str());
}

std::shared_ptr<caiwei::context::Context> caiwei::runtime::Runtime::get_context(const caiwei::context::ContextInfo* info) {
    if (info == nullptr) {
        return nullptr;
    }
    std::unique_lock<std::mutex> lock(this->mutex);
    auto& content_list = this->context_map[info->name];
    for (auto& context : content_list) {
        if (this->share) {
            if (context->ref_count >= this->max_pool) {
                continue;
            }
            CW_LOG_D("返回context: %s = %s = %d => %s", get_type_name(this->type), info->name.c_str(), content_list.size(), context->id.c_str());
            context->ref_count++;
            return context;
        } else {
            if (context->ref_count > 0) {
                continue;
            }
            CW_LOG_D("返回context: %s = %s = %d => %s", get_type_name(this->type), info->name.c_str(), content_list.size(), context->id.c_str());
            context->ref_count++;
            return context;
        }
    }
    std::shared_ptr<caiwei::context::Context> ptr{ nullptr };
    if (content_list.size() < static_cast<size_t>(this->max_pool)) {
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
            // 日志记录manager.get_context
            return nullptr;
        }
        try {
            if (ptr->load()) {
                CW_LOG_D("加载context成功: %s = %s", get_type_name(this->type), info->name.c_str());
            } else {
                CW_LOG_W("加载context失败: %s = %s", get_type_name(this->type), info->name.c_str());
                return nullptr;
            }
        } catch (...) {
            CW_LOG_E("加载context异常: %s = %s", get_type_name(this->type), info->name.c_str());
            return nullptr;
        }
        CW_LOG_I("新建context成功: %s = %s = %d = %s", get_type_name(this->type), info->name.c_str(), content_list.size(), ptr->id.c_str());
        content_list.push_back(ptr);
        CW_LOG_D("返回context: %s = %s = %d = %s", get_type_name(this->type), info->name.c_str(), content_list.size(), ptr->id.c_str());
        ptr->ref_count++;
        return ptr;
    } else {
        this->condition.wait_for(lock, std::chrono::seconds(this->timeout), [this, &content_list]() {
            return std::any_of(content_list.begin(), content_list.end(), [this](auto& context) {
                if (this->share) {
                    return context->ref_count < this->max_pool;
                } else {
                    return context->ref_count == 0;
                }
            });
        });
        for (auto& context : content_list) {
            if (this->share) {
                if (context->ref_count >= this->max_pool) {
                    continue;
                }
                CW_LOG_D("返回context: %s = %s = %d => %s", get_type_name(this->type), info->name.c_str(), content_list.size(), context->id.c_str());
                context->ref_count++;
                return context;
            } else {
                if (context->ref_count > 0) {
                    continue;
                }
                CW_LOG_D("返回context: %s = %s = %d => %s", get_type_name(this->type), info->name.c_str(), content_list.size(), context->id.c_str());
                context->ref_count++;
                return context;
            }
        }
        CW_LOG_W("获取context超时: %s = %s", get_type_name(this->type), info->name.c_str());
        return nullptr;
    }
}

void caiwei::runtime::Runtime::put_context(std::shared_ptr<caiwei::context::Context> context) {
    if (context == nullptr) {
        return;
    }
    CW_LOG_D("回收context: %s", context->id.c_str());
    std::lock_guard<std::mutex> lock(this->mutex);
    if (context->ref_count > 0) {
        context->ref_count--;
    } else {
        CW_LOG_W("回收context计数失败: %s = %d", context->id.c_str(), context->ref_count.load());
        context->ref_count = 0;
    }
    this->condition.notify_all();
}

size_t caiwei::runtime::Runtime::optimize() {
    std::unique_lock<std::mutex> lock(this->mutex, std::defer_lock);
    if (lock.try_lock()) {
        CW_LOG_D("优化runtime开始: %s => %d", get_type_name(this->type), this->context_map.size());
        const auto now = std::chrono::steady_clock::now();
        for (auto map_iter = this->context_map.begin(); map_iter != this->context_map.end();) {
            auto& context_name = map_iter->first;
            auto& content_list = map_iter->second;
            CW_LOG_D("优化context信息: %s = %s => %d", get_type_name(this->type), context_name.c_str(), content_list.size());
            if (content_list.size() <= static_cast<size_t>(this->min_pool)) {
                if (content_list.size() <= 0) {
                    map_iter = this->context_map.erase(map_iter);
                } else {
                    ++map_iter;
                }
                continue;
            }
            CW_LOG_D("优化context开始: %s = %s => %d", get_type_name(this->type), context_name.c_str(), content_list.size());
            for (auto context_iter = content_list.begin(); context_iter != content_list.end();) {
                const auto& ptr = *context_iter;
                if (ptr->ref_count > 0) {
                    ++context_iter;
                    continue;
                }
                const auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - ptr->last_run_time).count();
                if (duration > this->keepalive) {
                    CW_LOG_I("优化context实例: %s = %s = %s => %lld", get_type_name(this->type), context_name.c_str(), ptr->id.c_str(), duration);
                    context_iter = content_list.erase(context_iter);
                } else {
                    ++context_iter;
                }
            }
            CW_LOG_D("优化context完成: %s = %s => %d", get_type_name(this->type), context_name.c_str(), content_list.size());
            if (content_list.size() <= 0) {
                map_iter = this->context_map.erase(map_iter);
            } else {
                ++map_iter;
            }
        }
        CW_LOG_D("优化runtime完成: %s => %d", get_type_name(this->type), this->context_map.size());
        return this->context_map.size();
    } else {
        CW_LOG_D("优化runtime跳过: %s", get_type_name(this->type));
        // 加锁失败恒等于一避免淘汰
        return 1;
    }
}

std::shared_ptr<caiwei::context::ClsContext> caiwei::runtime::Runtime::get_cls_context(const caiwei::context::ContextInfo* /* info */) {
    return nullptr;
}

std::shared_ptr<caiwei::context::DetContext> caiwei::runtime::Runtime::get_det_context(const caiwei::context::ContextInfo* /* info */) {
    return nullptr;
}

std::shared_ptr<caiwei::context::ASRContext> caiwei::runtime::Runtime::get_asr_context(const caiwei::context::ContextInfo* /* info */) {
    return nullptr;
}

std::shared_ptr<caiwei::context::LLMContext> caiwei::runtime::Runtime::get_llm_context(const caiwei::context::ContextInfo* /* info */) {
    return nullptr;
}

std::shared_ptr<caiwei::context::VLMContext> caiwei::runtime::Runtime::get_vlm_context(const caiwei::context::ContextInfo* /* info */) {
    return nullptr;
}

std::shared_ptr<caiwei::context::EmbeddingContext> caiwei::runtime::Runtime::get_embedding_context(const caiwei::context::ContextInfo* /* info */) {
    return nullptr;
}

std::shared_ptr<caiwei::context::RerankingContext> caiwei::runtime::Runtime::get_reranking_context(const caiwei::context::ContextInfo* /* info */) {
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
    runtime_optimize_timer();
}

void caiwei::runtime::stop() {
    {
        std::lock_guard<std::shared_mutex> lock(caiwei::runtime::runtime_mutex);
        caiwei::runtime::runtime_map.clear();
    }
    std::jthread thread;
    {
        std::lock_guard<std::mutex> lock(runtime_optimize_mutex);
        if (runtime_optimize_thread.joinable()) {
            runtime_optimize_thread.request_stop();
            runtime_optimize_cv.notify_one();
            thread = std::move(runtime_optimize_thread);
        }
    }
    if (thread.joinable()) {
        thread.join();
    }
}

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
static void init_rknn2() {
    CW_LOG_I("RKNN2设备环境");
}
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN3
static void init_rknn3() {
    rknn3_devices devs{};
    int ret = rknn3_find_devices(&devs);
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("RKNN3设备无效");
        return;
    }
    for (uint32_t i = 0; i < devs.n_devices; ++i) {
        CW_LOG_I("RKNN3设备环境: %u = %s => %s", i, devs.devices[i].type, devs.devices[i].id);
    }
}
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_LLAMACPP
static void caiwei_llama_log(enum ggml_log_level level, const char* text, void* /* user_data */) {
    size_t n = std::strlen(text);
    if (n == 0) {
        return;
    }
    if (text[n - 1] == '\n') {
        if (n > 1 && text[n - 2] == '\r') {
            n--;
        }
        n--;
    }
    if (level == GGML_LOG_LEVEL_DEBUG) {
//      CW_LOG_D("%.*s", static_cast<int>(n), text);
    } else if (level == GGML_LOG_LEVEL_INFO) {
        CW_LOG_I("%.*s", static_cast<int>(n), text);
    } else if (level == GGML_LOG_LEVEL_WARN) {
        CW_LOG_W("%.*s", static_cast<int>(n), text);
    } else if (level == GGML_LOG_LEVEL_ERROR) {
        CW_LOG_E("%.*s", static_cast<int>(n), text);
    } else {
        // -
    }
}

static void init_llamacpp() {
    llama_log_set      (caiwei_llama_log, nullptr);
//  mtmd_log_set       (caiwei_llama_log, nullptr);
    mtmd_helper_log_set(caiwei_llama_log, nullptr);
    ggml_backend_load_all();
    CW_LOG_I("llama.cpp版本信息: %s", ggml_version());
    const size_t backend_count = ggml_backend_reg_count();
    for (size_t i = 0; i < backend_count; ++i) {
        ggml_backend_reg* reg = ggml_backend_reg_get(i);
        if (reg == nullptr) {
            continue;
        }
        CW_LOG_I("llama.cpp支持后端: %s", ggml_backend_reg_name(reg));
    }
}
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
static void init_onnxruntime() {
    CW_LOG_I("ONNXRuntime版本信息: %s", Ort::GetVersionString().c_str());
    CW_LOG_I("ONNXRuntime构建信息: %s", Ort::GetBuildInfoString().c_str());
    for(const auto& provider : Ort::GetAvailableProviders()) {
        CW_LOG_I("ONNXRuntime支持后端: %s", provider.c_str());
    }
}
#endif

static void runtime_optimize_timer() {
    std::lock_guard<std::mutex> lock(runtime_optimize_mutex);
    if (runtime_optimize_thread.joinable()) {
        CW_LOG_D("定时线程已经启动");
        return;
    }
    runtime_optimize_thread = std::jthread([](std::stop_token token) {
        caiwei::env::set_thread_name("rt-optimize");
        CW_LOG_D("定时优化runtime线程启动");
        const int interval = caiwei::env::get_int("CAIWEI_RUNTIME_INTERVAL");
        while (!token.stop_requested()) {
            {
                std::unique_lock<std::mutex> lock(runtime_optimize_mutex);
                runtime_optimize_cv.wait_for(lock, std::chrono::seconds(interval));
                if (token.stop_requested()) {
                    break;
                }
            }
            {
                std::unique_lock<std::shared_mutex> lock(caiwei::runtime::runtime_mutex, std::defer_lock);
                if (lock.try_lock()) {
                    CW_LOG_D("定时优化runtime开始: %zu", caiwei::runtime::runtime_map.size());
                    for (auto iter = caiwei::runtime::runtime_map.begin(); iter != caiwei::runtime::runtime_map.end();) {
                        auto& runtime = iter->second;
                        if (runtime->optimize() == 0) {
                            CW_LOG_I("优化runtime实例: %s", runtime->id.c_str());
                            iter = caiwei::runtime::runtime_map.erase(iter);
                        } else {
                            ++iter;
                        }
                    }
                    CW_LOG_D("定时优化runtime结束: %zu", caiwei::runtime::runtime_map.size());
                } else {
                    CW_LOG_D("定时优化runtime跳过");
                }
            }
        }
        CW_LOG_D("定时优化runtime线程结束");
    });
}
