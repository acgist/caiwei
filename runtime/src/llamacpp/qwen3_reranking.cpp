#include "llamacpp.hpp"

template <>
std::shared_ptr<caiwei::context::TextRerankingContext> caiwei::context::get_context(caiwei::context::Type type, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}
