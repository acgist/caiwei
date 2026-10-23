#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/runtime.hpp"
#include "caiwei/runtime/llamacpp.hpp"

#include <filesystem>

template<>
std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    if (type != caiwei::runtime::Type::LLAMACPP) {
        CW_LOG_W("类型无效: %d", type);
        return nullptr;
    }
    const int  min_pool  = caiwei::env::get_int("CAIWEI_LLAMACPP_MIN_POOL");
    const int  max_pool  = caiwei::env::get_int("CAIWEI_LLAMACPP_MAX_POOL");
    const int  timeout   = caiwei::env::get_int("CAIWEI_RUNTIME_TIMEOUT");
    const int  keepalive = caiwei::env::get_int("CAIWEI_RUNTIME_KEEPALIVE");
    const bool share     = caiwei::env::get_bool("CAIWEI_LLAMACPP_SHARE");
    return std::make_shared<caiwei::runtime::LlamaCPPRuntime>(min_pool, max_pool, timeout, keepalive, share);
}

caiwei::runtime::LlamaCPPRuntime::LlamaCPPRuntime(int min_pool, int max_pool, int timeout, int keepalive, bool share) : Runtime(min_pool, max_pool, timeout, keepalive, share, caiwei::runtime::Type::LLAMACPP) {
    CW_LOG_I("创建LlamaCPPRuntime");
}

caiwei::runtime::LlamaCPPRuntime::~LlamaCPPRuntime() {
    CW_LOG_I("释放LlamaCPPRuntime");
}

std::shared_ptr<caiwei::context::ASRContext> caiwei::runtime::LlamaCPPRuntime::get_asr_context(const caiwei::context::ContextInfo* info) {
    std::vector<std::string> paths = info->paths;
    if (paths.size() != 2) {
        CW_LOG_W("模型无效: %s", info->path.c_str());
        return nullptr;
    }
    std::string model_path  = paths[0];
    std::string mmproj_path = paths[1];
    if (!std::filesystem::exists(model_path)) {
        CW_LOG_W("模型无效: %s", model_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(mmproj_path)) {
        CW_LOG_W("模型无效: %s", mmproj_path.c_str());
        return nullptr;
    }
    caiwei::text::SpecialToken special_token;
    special_token.load_asr();
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_ASR_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::ASRLlamaCPPContext>(model_path, mmproj_path, max_token_length, special_token, this);
}

std::shared_ptr<caiwei::context::LLMContext> caiwei::runtime::LlamaCPPRuntime::get_llm_context(const caiwei::context::ContextInfo* info) {
    if (!std::filesystem::exists(info->path)) {
        CW_LOG_W("模型无效: %s", info->path.c_str());
        return nullptr;
    }
    caiwei::text::SpecialToken special_token;
    special_token.load_llm();
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_LLM_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::LLMLlamaCPPContext>(info->path, max_token_length, special_token, this);
}

std::shared_ptr<caiwei::context::VLMContext> caiwei::runtime::LlamaCPPRuntime::get_vlm_context(const caiwei::context::ContextInfo* info) {
    std::vector<std::string> paths = info->paths;
    if (paths.size() != 2) {
        CW_LOG_W("模型无效: %s", info->path.c_str());
        return nullptr;
    }
    std::string model_path  = paths[0];
    std::string mmproj_path = paths[1];
    if (!std::filesystem::exists(model_path)) {
        CW_LOG_W("模型无效: %s", model_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(mmproj_path)) {
        CW_LOG_W("模型无效: %s", mmproj_path.c_str());
        return nullptr;
    }
    caiwei::text::SpecialToken special_token;
    special_token.load_vlm();
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_VLM_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::VLMLlamaCPPContext>(model_path, mmproj_path, max_token_length, special_token, this);
}

std::shared_ptr<caiwei::context::EmbeddingContext> caiwei::runtime::LlamaCPPRuntime::get_embedding_context(const caiwei::context::ContextInfo* info) {
    if (!std::filesystem::exists(info->path)) {
        CW_LOG_W("模型无效: %s", info->path.c_str());
        return nullptr;
    }
    caiwei::text::SpecialToken special_token;
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_EMBEDDING_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::EmbeddingLlamaCPPContext>(info->path, max_token_length, special_token, this);
}

std::shared_ptr<caiwei::context::RerankingContext> caiwei::runtime::LlamaCPPRuntime::get_reranking_context(const caiwei::context::ContextInfo* info) {
    if (!std::filesystem::exists(info->path)) {
        CW_LOG_W("模型无效: %s", info->path.c_str());
        return nullptr;
    }
    caiwei::text::SpecialToken special_token;
    special_token.load_reranking();
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_RERANKING_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::RerankingLlamaCPPContext>(info->path, max_token_length, special_token, this);
}
