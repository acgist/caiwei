#include "caiwei/log.hpp"
#include "caiwei/runtime.hpp"

caiwei::runtime::MNNRuntime::MNNRuntime() : Runtime(caiwei::runtime::Type::MNN) {
    LOG_INFO("MNNRuntime init");
}

caiwei::runtime::MNNRuntime::~MNNRuntime() {
    LOG_INFO("MNNRuntime stop");
}

template<>
std::shared_ptr<caiwei::runtime::MNNRuntime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    return std::make_shared<caiwei::runtime::MNNRuntime>();
}
