#ifndef CAIWEI_RUNTIME_MANAGER_HPP
#define CAIWEI_RUNTIME_MANAGER_HPP

#include "caiwei/log.hpp"
#include "caiwei/context.hpp"
#include "caiwei/runtime.hpp"
#ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
#include "caiwei/runtime/rknn2.hpp"
#endif
#ifdef ENABLE_CAIWEI_RUNTIME_RKNN3
#include "caiwei/runtime/rknn3.hpp"
#endif
#ifdef ENABLE_CAIWEI_RUNTIME_LLAMACPP
#include "caiwei/runtime/llamacpp.hpp"
#endif
#ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
#include "caiwei/runtime/onnxruntime.hpp"
#endif

namespace caiwei  {
namespace manager {

void init();
void stop();

template <typename C>
class ContextWrapper {
private:
    using I = typename C::I;
    using O = typename C::O;
    std::shared_ptr<C> context{ nullptr };
    std::shared_ptr<caiwei::runtime::Runtime> runtime{ nullptr };
public:
    C* ptr();
    O  run(const I& input);
public:
    ContextWrapper(std::shared_ptr<C> context, std::shared_ptr<caiwei::runtime::Runtime> runtime);
    ~ContextWrapper();
};

template <typename C>
caiwei::manager::ContextWrapper<C>::ContextWrapper(
    std::shared_ptr<C> context,
    std::shared_ptr<caiwei::runtime::Runtime> runtime
) : context(std::move(context)), runtime(std::move(runtime)) {
}

template <typename C>
caiwei::manager::ContextWrapper<C>::~ContextWrapper() {
    this->runtime->put_context(this->context);
}

template <typename C>
C* caiwei::manager::ContextWrapper<C>::ptr() {
    return this->context.get();
}

template <typename C>
typename C::O caiwei::manager::ContextWrapper<C>::run(const I& input) {
    this->context->last_run_time = std::chrono::steady_clock::now();
    return this->context->run(input);
}

template <typename R>
inline std::shared_ptr<R> get_runtime_impl(caiwei::runtime::Type type) {
    auto iter = caiwei::runtime::runtime_map.find(type);
    if (iter != caiwei::runtime::runtime_map.end()) {
        return std::static_pointer_cast<R>(iter->second);
    }
    std::lock_guard<std::mutex> lock(caiwei::runtime::runtime_mutex);
    iter = caiwei::runtime::runtime_map.find(type);
    if (iter != caiwei::runtime::runtime_map.end()) {
        return std::static_pointer_cast<R>(iter->second);
    }
    auto runtime = caiwei::runtime::get_runtime<R>(type);
    if (runtime) {
        caiwei::runtime::runtime_map.emplace(type, runtime);
        return runtime;
    }
    return nullptr;
}

template <typename R>
inline std::shared_ptr<caiwei::context::Context> get_context_impl(const caiwei::context::ContextInfo* info, std::shared_ptr<R> runtime) {
    if (info == nullptr) {
        return nullptr;
    }
    if (runtime == nullptr) {
        return nullptr;
    }
    return runtime->get_context(info);
};

template <typename C>
std::unique_ptr<ContextWrapper<C>> get_context(const std::string& name, caiwei::runtime::Type runtime_type = caiwei::runtime::Type::NONE) {
    const auto* info = caiwei::context::get_context_info(name);
    if (info == nullptr) {
        CW_LOG_D("没有模型配置信息: %s", name.c_str());
        return nullptr;
    }
    #ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
    if (runtime_type == caiwei::runtime::Type::NONE || runtime_type == caiwei::runtime::Type::RKNN2) {
        auto runtime = get_runtime_impl<caiwei::runtime::RKNN2Runtime>(caiwei::runtime::Type::RKNN2);
        auto context = get_context_impl<caiwei::runtime::RKNN2Runtime>(info, runtime);
        if (context != nullptr) {
            return std::make_unique<ContextWrapper<C>>(std::static_pointer_cast<C>(context), runtime);
        }
    }
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_RKNN3
    if (runtime_type == caiwei::runtime::Type::NONE || runtime_type == caiwei::runtime::Type::RKNN3) {
        auto runtime = get_runtime_impl<caiwei::runtime::RKNN3Runtime>(caiwei::runtime::Type::RKNN3);
        auto context = get_context_impl<caiwei::runtime::RKNN3Runtime>(info, runtime);
        if (context != nullptr) {
            return std::make_unique<ContextWrapper<C>>(std::static_pointer_cast<C>(context), runtime);
        }
    }
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_LLAMACPP
    if (runtime_type == caiwei::runtime::Type::NONE || runtime_type == caiwei::runtime::Type::LLAMACPP) {
        auto runtime = get_runtime_impl<caiwei::runtime::LlamaCPPRuntime>(caiwei::runtime::Type::LLAMACPP);
        auto context = get_context_impl<caiwei::runtime::LlamaCPPRuntime>(info, runtime);
        if (context != nullptr) {
            return std::make_unique<ContextWrapper<C>>(std::static_pointer_cast<C>(context), runtime);
        }
    }
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
    if (runtime_type == caiwei::runtime::Type::NONE || runtime_type == caiwei::runtime::Type::ONNXRUNTIME) {
        auto runtime = get_runtime_impl<caiwei::runtime::ONNXRuntimeRuntime>(caiwei::runtime::Type::ONNXRUNTIME);
        auto context = get_context_impl<caiwei::runtime::ONNXRuntimeRuntime>(info, runtime);
        if (context != nullptr) {
            return std::make_unique<ContextWrapper<C>>(std::static_pointer_cast<C>(context), runtime);
        }
    }
    #endif
    return nullptr;
}

using ClsWrapper = caiwei::manager::ContextWrapper<caiwei::context::ClsContext>;
using DetWrapper = caiwei::manager::ContextWrapper<caiwei::context::DetContext>;
using ASRWrapper = caiwei::manager::ContextWrapper<caiwei::context::ASRContext>;
using LLMWrapper = caiwei::manager::ContextWrapper<caiwei::context::LLMContext>;
using VLMWrapper = caiwei::manager::ContextWrapper<caiwei::context::VLMContext>;
using EmbeddingWrapper = caiwei::manager::ContextWrapper<caiwei::context::EmbeddingContext>;
using RerankingWrapper = caiwei::manager::ContextWrapper<caiwei::context::RerankingContext>;

} // manager
} // caiwei

#endif //CAIWEI_RUNTIME_MANAGER_HPP
