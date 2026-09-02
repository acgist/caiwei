#include "caiwei/runtime/rknn.hpp"

template<>
std::shared_ptr<caiwei::runtime::RKNN2Runtime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    return std::make_shared<caiwei::runtime::RKNN2Runtime>();
}

template<>
std::shared_ptr<caiwei::runtime::RKNN3Runtime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    return std::make_shared<caiwei::runtime::RKNN3Runtime>();
}

caiwei::runtime::RKNN2Runtime::RKNN2Runtime() : Runtime(caiwei::runtime::Type::RKNN2) {
    CW_LOG_I("RKNN2Runtime init");
}

caiwei::runtime::RKNN2Runtime::~RKNN2Runtime() {
    CW_LOG_I("RKNN2Runtime stop");
}

caiwei::runtime::RKNN3Runtime::RKNN3Runtime() : Runtime(caiwei::runtime::Type::RKNN3) {
    CW_LOG_I("RKNN3Runtime init");
}

caiwei::runtime::RKNN3Runtime::~RKNN3Runtime() {
    CW_LOG_I("RKNN3Runtime stop");
}
