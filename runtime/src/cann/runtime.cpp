#include "caiwei/runtime.hpp"
#include "caiwei/runtime/cann.hpp"

template<>
std::shared_ptr<caiwei::runtime::CANNRuntime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    int min_pool = caiwei::env::get_int("CAIWEI_CANN_MIN_POOL");
    int max_pool = caiwei::env::get_int("CAIWEI_CANN_MAX_POOL");
    return std::make_shared<caiwei::runtime::CANNRuntime>(min_pool, max_pool);
}

caiwei::runtime::CANNRuntime::CANNRuntime(int min_pool, int max_pool) : Runtime(min_pool, max_pool, caiwei::runtime::Type::CANN) {
    CW_LOG_I("CANNRuntime init");
}

caiwei::runtime::CANNRuntime::~CANNRuntime() {
    CW_LOG_I("CANNRuntime stop");
}
