#include "rknn.hpp"

template <>
std::shared_ptr<caiwei::context::VLRerankingContext> caiwei::context::get_context(caiwei::context::Type type, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime) {
    return nullptr;
}

template <>
std::shared_ptr<caiwei::context::VLRerankingContext> caiwei::context::get_context(caiwei::context::Type type, std::shared_ptr<caiwei::runtime::RKNN3Runtime> runtime) {
    return nullptr;
}
