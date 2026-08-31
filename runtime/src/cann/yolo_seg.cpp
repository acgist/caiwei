#include "cann.hpp"

template <>
std::shared_ptr<caiwei::context::SegContext> caiwei::context::get_context(caiwei::context::Type type, std::shared_ptr<caiwei::runtime::CANNRuntime> runtime) {
    return nullptr;
}
