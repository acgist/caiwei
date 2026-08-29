#include "caiwei/log.hpp"
#include "caiwei/runtime.hpp"

caiwei::runtime::RKNN2Runtime::RKNN2Runtime() : Runtime(caiwei::runtime::Type::RKNN2) {
    LOG_INFO("RKNN2Runtime init");
}

caiwei::runtime::RKNN2Runtime::~RKNN2Runtime() {
    LOG_INFO("RKNN2Runtime stop");
}

caiwei::runtime::RKNN3Runtime::RKNN3Runtime() : Runtime(caiwei::runtime::Type::RKNN3) {
    LOG_INFO("RKNN3Runtime init");
}

caiwei::runtime::RKNN3Runtime::~RKNN3Runtime() {
    LOG_INFO("RKNN3Runtime stop");
}

template<>
std::shared_ptr<caiwei::runtime::RKNN2Runtime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    return std::make_shared<caiwei::runtime::RKNN2Runtime>();
}

template<>
std::shared_ptr<caiwei::runtime::RKNN3Runtime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    return std::make_shared<caiwei::runtime::RKNN3Runtime>();
}
