#include "rknn.hpp"

template <>
std::shared_ptr<caiwei::context::TextRerankingContext> caiwei::context::get_context(caiwei::context::Type type, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime) {
    return nullptr;
}
