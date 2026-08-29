#include "caiwei/context.hpp"

std::mutex caiwei::context::context_mutex;
std::mutex caiwei::context::runtime_mutex;
std::map<caiwei::context::Type, std::shared_ptr<caiwei::context::Context>> caiwei::context::context_map;
std::map<caiwei::runtime::Type, std::shared_ptr<caiwei::runtime::Runtime>> caiwei::context::runtime_map;

caiwei::context::Context::Context(std::shared_ptr<caiwei::runtime::Runtime> runtime) : runtime(std::move(runtime)) {
    this->runtime->ref();
}

caiwei::context::Context::~Context() {
    this->runtime->unref();
}

uint32_t caiwei::context::Context::ref() {
    auto old = this->ref_count.fetch_add(1);
    return old + 1;
}

uint32_t caiwei::context::Context::unref() {
    auto old = this->ref_count.fetch_sub(1);
    return old - 1;
}

void caiwei::context::init() {
}

void caiwei::context::stop() {
}

caiwei::context::DetContext::DetContext(int w, int h, int class_size, float iou_threshold, float confidence_threshold, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime)
 : w(w),
   h(h),
   class_size(class_size),
   iou_threshold(iou_threshold),
   confidence_threshold(confidence_threshold),
   Context(std::move(runtime)) {
}

caiwei::context::DetContext::~DetContext() {
}
