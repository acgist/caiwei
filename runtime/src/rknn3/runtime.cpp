#include "caiwei/runtime/rknn3.hpp"

template<>
std::shared_ptr<caiwei::runtime::RKNN3Runtime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    return std::make_shared<caiwei::runtime::RKNN3Runtime>();
}

caiwei::runtime::RKNN3Runtime::RKNN3Runtime() : Runtime(caiwei::runtime::Type::RKNN3) {
    CW_LOG_I("RKNN3Runtime init");
}

caiwei::runtime::RKNN3Runtime::~RKNN3Runtime() {
    CW_LOG_I("RKNN3Runtime stop");
}
