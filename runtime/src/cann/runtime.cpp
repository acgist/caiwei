#include "caiwei/log.hpp"
#include "caiwei/runtime.hpp"

caiwei::runtime::CANNRuntime::CANNRuntime() : Runtime(caiwei::runtime::Type::CANN) {
    CW_LOG_I("CANNRuntime init");
}

caiwei::runtime::CANNRuntime::~CANNRuntime() {
    CW_LOG_I("CANNRuntime stop");
}

template<>
std::shared_ptr<caiwei::runtime::CANNRuntime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    return std::make_shared<caiwei::runtime::CANNRuntime>();
}
