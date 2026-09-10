#include "caiwei/runtime.hpp"
#include "caiwei/runtime/rknn3.hpp"

template<>
std::shared_ptr<caiwei::runtime::RKNN3Runtime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    int min_pool = caiwei::env::get_int("CAIWEI_RKNN3_MIN_POOL");
    int max_pool = caiwei::env::get_int("CAIWEI_RKNN3_MAX_POOL");
    return std::make_shared<caiwei::runtime::RKNN3Runtime>(min_pool, max_pool);
}

caiwei::runtime::RKNN3Runtime::RKNN3Runtime(int min_pool, int max_pool) : Runtime(min_pool, max_pool, caiwei::runtime::Type::RKNN3) {
    CW_LOG_I("RKNN3Runtime init");
}

caiwei::runtime::RKNN3Runtime::~RKNN3Runtime() {
    CW_LOG_I("RKNN3Runtime stop");
}
