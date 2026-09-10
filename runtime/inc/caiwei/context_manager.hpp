#ifndef CAIWEI_RUNTIME_CONTEXT_MANAGER_HPP
#define CAIWEI_RUNTIME_CONTEXT_MANAGER_HPP

#include "caiwei/context.hpp"
#include "caiwei/runtime.hpp"
#ifdef ENABLE_CAIWEI_RUNTIME_CANN
#include "caiwei/runtime/cann.hpp"
#endif
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
namespace context {

extern void free();

template <typename C, typename I, typename O>
class ContextWrapper {
private:
    std::shared_ptr<C> context{ nullptr };
    std::shared_ptr<caiwei::runtime::Runtime> runtime{ nullptr };
public:
    std::shared_ptr<C> ptr();
    O run(const I& input);
public:
    ContextWrapper(std::shared_ptr<C> context, std::shared_ptr<caiwei::runtime::Runtime> runtime);
    ~ContextWrapper();
};

template <typename C, typename I, typename O>
caiwei::context::ContextWrapper<C, I, O>::ContextWrapper(std::shared_ptr<C> context, std::shared_ptr<caiwei::runtime::Runtime> runtime) : context(std::move(context)), runtime(std::move(runtime)) {
    this->context->ref();
}

template <typename C, typename I, typename O>
caiwei::context::ContextWrapper<C, I, O>::~ContextWrapper() {
    this->context->unref();
    this->runtime->put_context(this->context);
}

template <typename C, typename I, typename O>
std::shared_ptr<C> caiwei::context::ContextWrapper<C, I, O>::ptr() {
    return this->context;
}

template <typename C, typename I, typename O>
O caiwei::context::ContextWrapper<C, I, O>::run(const I& input) {
    this->context->last_run_time = std::chrono::system_clock::now();
    return this->context->run(input);
}

extern std::mutex runtime_mutex;
extern std::map<caiwei::runtime::Type, std::shared_ptr<caiwei::runtime::Runtime>> runtime_map;

template <typename R>
inline std::shared_ptr<R> get_runtime_impl(caiwei::runtime::Type type) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    auto iter = runtime_map.find(type);
    if (iter != runtime_map.end()) {
        return std::dynamic_pointer_cast<R>(iter->second);
    }
    auto runtime = caiwei::runtime::get_runtime<R>(type);
    if (runtime) {
        runtime_map[type] = runtime;
        return runtime;
    }
    return nullptr;
}

template <typename R>
inline std::shared_ptr<Context> get_context_impl(const ContextInfo* info, std::shared_ptr<R> runtime) {
    if (info == nullptr) {
        return nullptr;
    }
    if (runtime == nullptr) {
        return nullptr;
    }
    return runtime->get_context(info);
};

template <typename C, typename I, typename O>
std::unique_ptr<ContextWrapper<C, I, O>> get_context(const std::string& name, caiwei::runtime::Type runtime_type = caiwei::runtime::Type::NONE) {
    const auto* info = get_context_info(name);
    if (info == nullptr) {
        return nullptr;
    }
    #ifdef ENABLE_CAIWEI_RUNTIME_CANN
    if (runtime_type == caiwei::runtime::Type::NONE || runtime_type == caiwei::runtime::Type::CANN) {
        auto runtime = get_runtime_impl<caiwei::runtime::CANNRuntime>(caiwei::runtime::Type::CANN);
        auto context = get_context_impl<caiwei::runtime::CANNRuntime>(info, runtime);
        if (context != nullptr) {
            return std::make_unique<ContextWrapper<C, I, O>>(std::dynamic_pointer_cast<C>(context), runtime);
        }
    }
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
    if (runtime_type == caiwei::runtime::Type::NONE || runtime_type == caiwei::runtime::Type::RKNN2) {
        auto runtime = get_runtime_impl<caiwei::runtime::RKNN2Runtime>(caiwei::runtime::Type::RKNN2);
        auto context = get_context_impl<caiwei::runtime::RKNN2Runtime>(info, runtime);
        if (context != nullptr) {
            return std::make_unique<ContextWrapper<C, I, O>>(std::dynamic_pointer_cast<C>(context), runtime);
        }
    }
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_RKNN3
    if (runtime_type == caiwei::runtime::Type::NONE || runtime_type == caiwei::runtime::Type::RKNN3) {
        auto runtime = get_runtime_impl<caiwei::runtime::RKNN3Runtime>(caiwei::runtime::Type::RKNN3);
        auto context = get_context_impl<caiwei::runtime::RKNN3Runtime>(info, runtime);
        if (context != nullptr) {
            return std::make_unique<ContextWrapper<C, I, O>>(std::dynamic_pointer_cast<C>(context), runtime);
        }
    }
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_LLAMACPP
    if (runtime_type == caiwei::runtime::Type::NONE || runtime_type == caiwei::runtime::Type::LLAMACPP) {
        auto runtime = get_runtime_impl<caiwei::runtime::LlamaCPPRuntime>(caiwei::runtime::Type::LLAMACPP);
        auto context = get_context_impl<caiwei::runtime::LlamaCPPRuntime>(info, runtime);
        if (context != nullptr) {
            return std::make_unique<ContextWrapper<C, I, O>>(std::dynamic_pointer_cast<C>(context), runtime);
        }
    }
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
    if (runtime_type == caiwei::runtime::Type::NONE || runtime_type == caiwei::runtime::Type::ONNXRUNTIME) {
        auto runtime = get_runtime_impl<caiwei::runtime::ONNXRuntimeRuntime>(caiwei::runtime::Type::ONNXRUNTIME);
        auto context = get_context_impl<caiwei::runtime::ONNXRuntimeRuntime>(info, runtime);
        if (context != nullptr) {
            return std::make_unique<ContextWrapper<C, I, O>>(std::dynamic_pointer_cast<C>(context), runtime);
        }
    }
    #endif
    return nullptr;
}

}
}

#endif //CAIWEI_RUNTIME_CONTEXT_MANAGER_HPP
