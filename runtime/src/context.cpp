#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/context.hpp"
#include "caiwei/runtime.hpp"

#include <sstream>

std::vector<caiwei::context::ContextInfo> caiwei::context::context_info_list;

caiwei::context::Context::Context(caiwei::runtime::Runtime* runtime)
  : id(caiwei::env::id()),
    runtime(runtime),
    last_run_time(std::chrono::steady_clock::now()) {
    CW_LOG_D("创建context: %s", this->id.c_str());
}

caiwei::context::Context::~Context() {
    CW_LOG_D("释放context: %s", this->id.c_str());
}

bool caiwei::context::Context::load() {
    return true;
}

static void init_context_info_list();

void caiwei::context::init() {
    init_context_info_list();
}

void caiwei::context::stop() {
}

const caiwei::context::ContextInfo* caiwei::context::get_context_info(const std::string& name) {
    for (const auto& info : context_info_list) {
        if (info.name == name) {
            return &info;
        }
    }
    return nullptr;
}

caiwei::context::ClsContext::ClsContext(int c, int h, int w, int class_size, int top_k, float confidence_threshold, caiwei::runtime::Runtime* runtime)
 : c(c),
   h(h),
   w(w),
   class_size(class_size),
   top_k(top_k),
   confidence_threshold(confidence_threshold),
   Context(runtime) {
}

caiwei::context::ClsContext::~ClsContext() {
}

caiwei::context::DetContext::DetContext(int c, int h, int w, int class_size, float iou_threshold, float confidence_threshold, caiwei::runtime::Runtime* runtime)
 : c(c),
   h(h),
   w(w),
   class_size(class_size),
   iou_threshold(iou_threshold),
   confidence_threshold(confidence_threshold),
   Context(runtime) {
}

caiwei::context::DetContext::~DetContext() {
}

caiwei::context::ASRContext::ASRContext(caiwei::runtime::Runtime* runtime)
 : Context(runtime) {
}

caiwei::context::ASRContext::~ASRContext() {
}

caiwei::context::LLMContext::LLMContext(caiwei::runtime::Runtime* runtime)
 : Context(runtime) {
}

caiwei::context::LLMContext::~LLMContext() {
}

caiwei::context::VLMContext::VLMContext(caiwei::runtime::Runtime* runtime)
 : Context(runtime) {
}

caiwei::context::VLMContext::~VLMContext() {
}

caiwei::context::EmbeddingContext::EmbeddingContext(caiwei::runtime::Runtime* runtime)
 : Context(runtime) {
}

caiwei::context::EmbeddingContext::~EmbeddingContext() {
}

caiwei::context::RerankingContext::RerankingContext(caiwei::runtime::Runtime* runtime)
 : Context(runtime) {
}

caiwei::context::RerankingContext::~RerankingContext() {
}

static void init_context_info_list() {
    auto& list = caiwei::context::context_info_list;
    std::string line;
    std::string info = caiwei::env::get("CAIWEI_CONTEXT_INFO");
    std::stringstream stream(info);
    while (std::getline(stream, line)) {
        std::stringstream info_stream(line);
        std::string type, vendor, name, path;
        std::getline(info_stream, type,   ',');
        std::getline(info_stream, vendor, ',');
        std::getline(info_stream, name,   ',');
        std::getline(info_stream, path,   ',');
        if (type.empty() || vendor.empty() || name.empty() || path.empty()) {
            continue;
        }
        std::vector<std::string> paths;
        std::stringstream path_stream(path);
        std::string path_item;
        while (std::getline(path_stream, path_item, '|')) {
            paths.push_back(std::move(path_item));
        }
        CW_LOG_I("模型配置: %s %s %s = %s", type.c_str(), vendor.c_str(), name.c_str(), path.c_str());
        if (type == "CLS") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::CLS, std::move(type), std::move(name), std::move(path), std::move(vendor), std::move(paths) });
        } else if (type == "DET") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::DET, std::move(type), std::move(name), std::move(path), std::move(vendor), std::move(paths) });
        } else if (type == "ASR") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::ASR, std::move(type), std::move(name), std::move(path), std::move(vendor), std::move(paths) });
        } else if (type == "LLM") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::LLM, std::move(type), std::move(name), std::move(path), std::move(vendor), std::move(paths) });
        } else if (type == "VLM") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::VLM, std::move(type), std::move(name), std::move(path), std::move(vendor), std::move(paths) });
        } else if (type == "EMBEDDING") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::EMBEDDING, std::move(type), std::move(name), std::move(path), std::move(vendor), std::move(paths) });
        } else if (type == "RERANKING") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::RERANKING, std::move(type), std::move(name), std::move(path), std::move(vendor), std::move(paths) });
        } else {
            // -
        }
    }
}
