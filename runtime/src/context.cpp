#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/context.hpp"
#include "caiwei/runtime.hpp"

#include <sstream>

std::vector<caiwei::context::ContextInfo> caiwei::context::context_info_list;

caiwei::context::Context::Context(std::shared_ptr<caiwei::runtime::Runtime> runtime)
  : runtime(std::move(runtime)),
    last_run_time(std::chrono::system_clock::now()) {
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

const caiwei::context::ContextInfo* caiwei::context::get_context_info(const std::string& name) {
    for (const auto& info : context_info_list) {
        if (info.name == name) {
            return &info;
        }
    }
    return nullptr;
}

caiwei::context::ClsContext::ClsContext(int w, int h, int top_k, int class_size, float confidence_threshold, std::shared_ptr<caiwei::runtime::Runtime> runtime)
 : w(w),
   h(h),
   top_k(top_k),
   class_size(class_size),
   confidence_threshold(confidence_threshold),
   Context(std::move(runtime)) {
}

caiwei::context::ClsContext::~ClsContext() {
}

caiwei::context::DetContext::DetContext(int w, int h, int class_size, float iou_threshold, float confidence_threshold, std::shared_ptr<caiwei::runtime::Runtime> runtime)
 : w(w),
   h(h),
   class_size(class_size),
   iou_threshold(iou_threshold),
   confidence_threshold(confidence_threshold),
   Context(std::move(runtime)) {
}

caiwei::context::DetContext::~DetContext() {
}

caiwei::context::OBBContext::OBBContext(std::shared_ptr<caiwei::runtime::Runtime> runtime)
 : Context(std::move(runtime)) {
}

caiwei::context::OBBContext::~OBBContext() {
}

caiwei::context::SegContext::SegContext(std::shared_ptr<caiwei::runtime::Runtime> runtime)
 : Context(std::move(runtime)) {
}

caiwei::context::SegContext::~SegContext() {
}

caiwei::context::SemContext::SemContext(std::shared_ptr<caiwei::runtime::Runtime> runtime)
 : Context(std::move(runtime)) {
}

caiwei::context::SemContext::~SemContext() {
}

caiwei::context::PoseContext::PoseContext(std::shared_ptr<caiwei::runtime::Runtime> runtime)
 : Context(std::move(runtime)) {
}

caiwei::context::PoseContext::~PoseContext() {
}

caiwei::context::DepthContext::DepthContext(std::shared_ptr<caiwei::runtime::Runtime> runtime)
 : Context(std::move(runtime)) {
}

caiwei::context::DepthContext::~DepthContext() {
}

caiwei::context::ASRContext::ASRContext(std::shared_ptr<caiwei::runtime::Runtime> runtime)
 : Context(std::move(runtime)) {
}

caiwei::context::ASRContext::~ASRContext() {
}

caiwei::context::TTSContext::TTSContext(std::shared_ptr<caiwei::runtime::Runtime> runtime)
 : Context(std::move(runtime)) {
}

caiwei::context::TTSContext::~TTSContext() {
}

caiwei::context::LLMContext::LLMContext(std::shared_ptr<caiwei::runtime::Runtime> runtime)
 : Context(std::move(runtime)) {
}

caiwei::context::LLMContext::~LLMContext() {
}

caiwei::context::VLMContext::VLMContext(std::shared_ptr<caiwei::runtime::Runtime> runtime)
 : Context(std::move(runtime)) {
}

caiwei::context::VLMContext::~VLMContext() {
}

caiwei::context::MLLMContext::MLLMContext(std::shared_ptr<caiwei::runtime::Runtime> runtime)
 : Context(std::move(runtime)) {
}

caiwei::context::MLLMContext::~MLLMContext() {
}

caiwei::context::EmbeddingContext::EmbeddingContext(std::shared_ptr<caiwei::runtime::Runtime> runtime)
 : Context(std::move(runtime)) {
}

caiwei::context::EmbeddingContext::~EmbeddingContext() {
}

caiwei::context::VLEmbeddingContext::VLEmbeddingContext(std::shared_ptr<caiwei::runtime::Runtime> runtime)
 : Context(std::move(runtime)) {
}

caiwei::context::VLEmbeddingContext::~VLEmbeddingContext() {
}

caiwei::context::RerankingContext::RerankingContext(std::shared_ptr<caiwei::runtime::Runtime> runtime)
 : Context(std::move(runtime)) {
}

caiwei::context::RerankingContext::~RerankingContext() {
}

caiwei::context::VLRerankingContext::VLRerankingContext(std::shared_ptr<caiwei::runtime::Runtime> runtime)
 : Context(std::move(runtime)) {
}

caiwei::context::VLRerankingContext::~VLRerankingContext() {
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
        CW_LOG_I("模型配置: %s %s %s = %s", type.c_str(), vendor.c_str(), name.c_str(), path.c_str());
        if (type == "CLS") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::CLS, std::move(name), std::move(path), std::move(vendor) });
        } else if (type == "DET") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::DET, std::move(name), std::move(path), std::move(vendor) });
        } else if (type == "OBB") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::OBB, std::move(name), std::move(path), std::move(vendor) });
        } else if (type == "SEG") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::SEG, std::move(name), std::move(path), std::move(vendor) });
        } else if (type == "SEM") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::SEM, std::move(name), std::move(path), std::move(vendor) });
        } else if (type == "POSE") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::POSE, std::move(name), std::move(path), std::move(vendor) });
        } else if (type == "DEPTH") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::DEPTH, std::move(name), std::move(path), std::move(vendor) });
        } else if (type == "ASR") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::ASR, std::move(name), std::move(path), std::move(vendor) });
        } else if (type == "TTS") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::TTS, std::move(name), std::move(path), std::move(vendor) });
        } else if (type == "LLM") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::LLM, std::move(name), std::move(path), std::move(vendor) });
        } else if (type == "VLM") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::VLM, std::move(name), std::move(path), std::move(vendor) });
        } else if (type == "MLLM") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::MLLM, std::move(name), std::move(path), std::move(vendor) });
        } else if (type == "EMBEDDING") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::EMBEDDING, std::move(name), std::move(path), std::move(vendor) });
        } else if (type == "RERANKING") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::RERANKING, std::move(name), std::move(path), std::move(vendor) });
        } else if (type == "VL_EMBEDDING") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::VL_EMBEDDING, std::move(name), std::move(path), std::move(vendor) });
        } else if (type == "VL_RERANKING") {
            list.push_back(caiwei::context::ContextInfo{ caiwei::context::Type::VL_RERANKING, std::move(name), std::move(path), std::move(vendor) });
        } else {
            // -
        }
    }
}
