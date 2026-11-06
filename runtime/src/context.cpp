#include <sstream>

#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/context.hpp"
#include "caiwei/runtime.hpp"

static std::once_flag context_info_map_once_flag;
static std::unordered_map<std::string, caiwei::context::ContextInfo> context_info_map;

static void init_context_info_map();

inline constexpr const char* empty_char = " \t\r\n";

caiwei::context::Context::Context(
    Type type,
    caiwei::runtime::Runtime* runtime
) : type(type),
    id(caiwei::env::id()),
    ref_count(0),
    last_run_time(std::chrono::steady_clock::now()),
    runtime(runtime) {
    CW_LOG_D("创建context: %s", this->id.c_str());
}

caiwei::context::Context::~Context() {
    CW_LOG_D("释放context: %s", this->id.c_str());
}

caiwei::context::ClsContext::ClsContext(
    int c,
    int h,
    int w,
    int class_size,
    int top_k,
    float confidence_threshold,
    caiwei::runtime::Runtime* runtime
) : c(c),
    h(h),
    w(w),
    class_size(class_size),
    top_k(top_k),
    confidence_threshold(confidence_threshold),
    Context(Type::CLS, runtime) {
}

caiwei::context::ClsContext::~ClsContext() {
}

caiwei::context::DetContext::DetContext(
    int c,
    int h,
    int w,
    int class_size,
    float iou_threshold,
    float confidence_threshold,
    caiwei::runtime::Runtime* runtime
) : c(c),
    h(h),
    w(w),
    class_size(class_size),
    iou_threshold(iou_threshold),
    confidence_threshold(confidence_threshold),
    Context(Type::DET, runtime) {
}

caiwei::context::DetContext::~DetContext() {
}

caiwei::context::ASRContext::ASRContext(
    caiwei::runtime::Runtime* runtime
) : Context(Type::ASR, runtime) {
}

caiwei::context::ASRContext::~ASRContext() {
}

caiwei::context::LLMContext::LLMContext(
    caiwei::runtime::Runtime* runtime
) : Context(Type::LLM, runtime) {
}

caiwei::context::LLMContext::~LLMContext() {
}

caiwei::context::VLMContext::VLMContext(
    caiwei::runtime::Runtime* runtime
) : Context(Type::VLM, runtime) {
}

caiwei::context::VLMContext::~VLMContext() {
}

caiwei::context::EmbeddingContext::EmbeddingContext(
    caiwei::runtime::Runtime* runtime
) : Context(Type::EMBEDDING, runtime) {
}

caiwei::context::EmbeddingContext::~EmbeddingContext() {
}

caiwei::context::RerankingContext::RerankingContext(
    caiwei::runtime::Runtime* runtime
) : Context(Type::RERANKING, runtime) {
}

caiwei::context::RerankingContext::~RerankingContext() {
}

void caiwei::context::init() {
    std::call_once(context_info_map_once_flag, []() {
        init_context_info_map();
    });
}

void caiwei::context::stop() {
}

const caiwei::context::ContextInfo* caiwei::context::get_context_info(const std::string& name) {
    auto iter = context_info_map.find(name);
    if (iter == context_info_map.end()) {
        return nullptr;
    }
    return &iter->second;
}

std::vector<const caiwei::context::ContextInfo*> caiwei::context::get_all_context_info() {
    std::vector<const caiwei::context::ContextInfo*> ret;
    for (const auto& [_, v] : context_info_map) {
        ret.push_back(&v);
    }
    return ret;
}

inline static std::string strip(const std::string& value) {
    auto pos_a = value.find_first_not_of(empty_char);
    if (pos_a == std::string_view::npos) {
        return "";
    }
    auto pos_z = value.find_last_not_of(empty_char);
    return value.substr(pos_a, pos_z - pos_a + 1);
}

static void init_context_info_map() {
    static const std::unordered_map<std::string, caiwei::context::Type> type_map = {
        { "CLS",       caiwei::context::Type::CLS       },
        { "DET",       caiwei::context::Type::DET       },
        { "ASR",       caiwei::context::Type::ASR       },
        { "LLM",       caiwei::context::Type::LLM       },
        { "VLM",       caiwei::context::Type::VLM       },
        { "EMBEDDING", caiwei::context::Type::EMBEDDING },
        { "RERANKING", caiwei::context::Type::RERANKING },
    };
    std::unordered_map<std::string, caiwei::context::ContextInfo> map;
    std::string line;
    std::string info = caiwei::env::get("CAIWEI_CONTEXT_INFO");
    std::stringstream stream(info);
    while (std::getline(stream, line)) {
        line = strip(line);
        if (line.empty()) {
            continue;
        }
        std::string type, vendor, name, path;
        std::stringstream info_stream(line);
        std::getline(info_stream, type,   ',');
        std::getline(info_stream, vendor, ',');
        std::getline(info_stream, name,   ',');
        std::getline(info_stream, path,   ',');
        type   = strip(type);
        vendor = strip(vendor);
        name   = strip(name);
        path   = strip(path);
        if (type.empty() || vendor.empty() || name.empty() || path.empty()) {
            CW_LOG_W("模型配置错误: %s", line.c_str());
            continue;
        }
        const auto iter = type_map.find(type);
        if (iter == type_map.end()) {
            CW_LOG_W("未知模型类型: %s", type.c_str());
            continue;
        }
        std::string path_item;
        std::vector<std::string> path_list;
        std::stringstream path_stream(path);
        while (std::getline(path_stream, path_item, '|')) {
            path_list.push_back(std::move(path_item));
        }
        CW_LOG_I("模型配置: %s %s %s = %s", type.c_str(), vendor.c_str(), name.c_str(), path.c_str());
        map.insert_or_assign(name, caiwei::context::ContextInfo{
            iter->second,
            type,
            name,
            path,
            vendor,
            std::move(path_list)
        });
    }
    context_info_map.swap(map);
}
