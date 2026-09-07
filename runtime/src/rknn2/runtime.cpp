#include "caiwei/runtime/rknn2.hpp"

template<>
std::shared_ptr<caiwei::runtime::RKNN2Runtime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    return std::make_shared<caiwei::runtime::RKNN2Runtime>();
}

caiwei::runtime::RKNN2Runtime::RKNN2Runtime() : Runtime(caiwei::runtime::Type::RKNN2) {
    CW_LOG_I("RKNN2Runtime init");
}

caiwei::runtime::RKNN2Runtime::~RKNN2Runtime() {
    CW_LOG_I("RKNN2Runtime stop");
}
