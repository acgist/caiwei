#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/context.hpp"

std::vector<caiwei::context::ContextInfo> caiwei::context::context_info_list;

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

static void init_context_info_list();

void caiwei::context::init() {
    init_context_info_list();
}

void caiwei::context::stop() {
}

caiwei::context::Type caiwei::context::get_context_type(const std::string& name) {
    for (const auto& info : context_info_list) {
        if (info.name == name) {
            return info.type;
        }
    }
    return caiwei::context::Type::NONE;
}

caiwei::context::ClsContext::ClsContext(int w, int h, int top_k, int class_size, float confidence_threshold, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime)
 : w(w),
   h(h),
   top_k(top_k),
   class_size(class_size),
   confidence_threshold(confidence_threshold),
   Context(std::move(runtime)) {
}

caiwei::context::ClsContext::~ClsContext() {
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

static void init_context_info_list() {
    auto& list = caiwei::context::context_info_list;
    std::string name;
    std::string path;
    {
        name = caiwei::env::get("CAIWEI_CLS_NAME");
        path = caiwei::env::get("CAIWEI_CLS_PATH");
        if (!path.empty() && !name.empty()) {
            CW_LOG_I("YOLO_CLS: %s = %s", name.c_str(), path.c_str());
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::YOLO_CLS, std::move(name), std::move(path) });
        }
    }
    {
        name = caiwei::env::get("CAIWEI_DET_NAME");
        path = caiwei::env::get("CAIWEI_DET_PATH");
        if (!path.empty() && !name.empty()) {
            CW_LOG_I("YOLO_DET: %s = %s", name.c_str(), path.c_str());
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::YOLO_DET, std::move(name), std::move(path) });
        }
    }
}
