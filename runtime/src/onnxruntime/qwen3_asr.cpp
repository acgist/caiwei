#include "onnxruntime.hpp"

template <>
std::shared_ptr<caiwei::context::ASRContext> caiwei::context::get_context(caiwei::context::Type type, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) {
    return nullptr;
}
