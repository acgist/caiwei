#include <filesystem>

#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/runtime.hpp"
#include "caiwei/runtime/rknn3.hpp"

template<>
std::shared_ptr<caiwei::runtime::RKNN3Runtime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    if (type != caiwei::runtime::Type::RKNN3) {
        CW_LOG_W("类型无效: %d", type);
        return nullptr;
    }
    const int  min_pool  = caiwei::env::get_int("CAIWEI_RKNN3_MIN_POOL");
    const int  max_pool  = caiwei::env::get_int("CAIWEI_RKNN3_MAX_POOL");
    const int  timeout   = caiwei::env::get_int("CAIWEI_RUNTIME_TIMEOUT");
    const int  keepalive = caiwei::env::get_int("CAIWEI_RUNTIME_KEEPALIVE");
    const bool share     = caiwei::env::get_bool("CAIWEI_RKNN3_SHARE");
    return std::make_shared<caiwei::runtime::RKNN3Runtime>(min_pool, max_pool, timeout, keepalive, share);
}

caiwei::runtime::RKNN3Runtime::RKNN3Runtime(
    int min_pool, int max_pool, int timeout, int keepalive, bool share
) : Runtime(min_pool, max_pool, timeout, keepalive, share, caiwei::runtime::Type::RKNN3) {
    CW_LOG_I("创建RKNN3Runtime");
}

caiwei::runtime::RKNN3Runtime::~RKNN3Runtime() {
    CW_LOG_I("释放RKNN3Runtime");
}

std::shared_ptr<caiwei::context::ASRContext> caiwei::runtime::RKNN3Runtime::get_asr_context(const caiwei::context::ContextInfo* info) {
    std::vector<std::string> paths = info->paths;
    if (paths.size() != 6) {
        CW_LOG_W("模型无效: %s", info->path.c_str());
        return nullptr;
    }
    const std::string model_path        = paths[0];
    const std::string weight_path       = paths[1];
    const std::string embedding_path    = paths[2];
    const std::string tokenizer_path    = paths[3];
    const std::string media_model_path  = paths[4];
    const std::string media_weight_path = paths[5];
    if (!std::filesystem::exists(model_path)) {
        CW_LOG_W("模型无效: %s", model_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(weight_path)) {
        CW_LOG_W("模型无效: %s", weight_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(embedding_path)) {
        CW_LOG_W("模型无效: %s", embedding_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(tokenizer_path)) {
        CW_LOG_W("模型无效: %s", tokenizer_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(media_model_path)) {
        CW_LOG_W("模型无效: %s", media_model_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(media_weight_path)) {
        CW_LOG_W("模型无效: %s", media_weight_path.c_str());
        return nullptr;
    }
    caiwei::text::SpecialToken special_token;
    special_token.load_asr();
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_ASR_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::ASRRKNN3Context>(model_path, weight_path, embedding_path, tokenizer_path, media_model_path, media_weight_path, max_token_length, special_token, this);
}

std::shared_ptr<caiwei::context::LLMContext> caiwei::runtime::RKNN3Runtime::get_llm_context(const caiwei::context::ContextInfo* info) {
    std::vector<std::string> paths = info->paths;
    if (paths.size() != 4) {
        CW_LOG_W("模型无效: %s", info->path.c_str());
        return nullptr;
    }
    const std::string model_path     = paths[0];
    const std::string weight_path    = paths[1];
    const std::string embedding_path = paths[2];
    const std::string tokenizer_path = paths[3];
    if (!std::filesystem::exists(model_path)) {
        CW_LOG_W("模型无效: %s", model_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(weight_path)) {
        CW_LOG_W("模型无效: %s", weight_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(embedding_path)) {
        CW_LOG_W("模型无效: %s", embedding_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(tokenizer_path)) {
        CW_LOG_W("模型无效: %s", tokenizer_path.c_str());
        return nullptr;
    }
    caiwei::text::SpecialToken special_token;
    special_token.load_llm();
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_LLM_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::LLMRKNN3Context>(model_path, weight_path, embedding_path, tokenizer_path, max_token_length, special_token, this);
}


std::shared_ptr<caiwei::context::VLMContext> caiwei::runtime::RKNN3Runtime::get_vlm_context(const caiwei::context::ContextInfo* info) {
    std::vector<std::string> paths = info->paths;
    if (paths.size() != 6) {
        CW_LOG_W("模型无效: %s", info->path.c_str());
        return nullptr;
    }
    const std::string model_path        = paths[0];
    const std::string weight_path       = paths[1];
    const std::string embedding_path    = paths[2];
    const std::string tokenizer_path    = paths[3];
    const std::string media_model_path  = paths[4];
    const std::string media_weight_path = paths[5];
    if (!std::filesystem::exists(model_path)) {
        CW_LOG_W("模型无效: %s", model_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(weight_path)) {
        CW_LOG_W("模型无效: %s", weight_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(embedding_path)) {
        CW_LOG_W("模型无效: %s", embedding_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(tokenizer_path)) {
        CW_LOG_W("模型无效: %s", tokenizer_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(media_model_path)) {
        CW_LOG_W("模型无效: %s", media_model_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(media_weight_path)) {
        CW_LOG_W("模型无效: %s", media_weight_path.c_str());
        return nullptr;
    }
    caiwei::text::SpecialToken special_token;
    special_token.load_vlm();
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_VLM_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::VLMRKNN3Context>(model_path, weight_path, embedding_path, tokenizer_path, media_model_path, media_weight_path, max_token_length, special_token, this);
}

std::shared_ptr<caiwei::context::EmbeddingContext> caiwei::runtime::RKNN3Runtime::get_embedding_context(const caiwei::context::ContextInfo* info) {
    std::vector<std::string> paths = info->paths;
    if (paths.size() != 4) {
        CW_LOG_W("模型无效: %s", info->path.c_str());
        return nullptr;
    }
    const std::string model_path     = paths[0];
    const std::string weight_path    = paths[1];
    const std::string embedding_path = paths[2];
    const std::string tokenizer_path = paths[3];
    if (!std::filesystem::exists(model_path)) {
        CW_LOG_W("模型无效: %s", model_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(weight_path)) {
        CW_LOG_W("模型无效: %s", weight_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(embedding_path)) {
        CW_LOG_W("模型无效: %s", embedding_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(tokenizer_path)) {
        CW_LOG_W("模型无效: %s", tokenizer_path.c_str());
        return nullptr;
    }
    caiwei::text::SpecialToken special_token;
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_EMBEDDING_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::EmbeddingRKNN3Context>(model_path, weight_path, embedding_path, tokenizer_path, max_token_length, special_token, this);
}

std::shared_ptr<caiwei::context::RerankingContext> caiwei::runtime::RKNN3Runtime::get_reranking_context(const caiwei::context::ContextInfo* info) {
    std::vector<std::string> paths = info->paths;
    if (paths.size() != 4) {
        CW_LOG_W("模型无效: %s", info->path.c_str());
        return nullptr;
    }
    const std::string model_path     = paths[0];
    const std::string weight_path    = paths[1];
    const std::string embedding_path = paths[2];
    const std::string tokenizer_path = paths[3];
    if (!std::filesystem::exists(model_path)) {
        CW_LOG_W("模型无效: %s", model_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(weight_path)) {
        CW_LOG_W("模型无效: %s", weight_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(embedding_path)) {
        CW_LOG_W("模型无效: %s", embedding_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(tokenizer_path)) {
        CW_LOG_W("模型无效: %s", tokenizer_path.c_str());
        return nullptr;
    }
    caiwei::text::SpecialToken special_token;
    special_token.load_reranking();
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_RERANKING_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::RerankingRKNN3Context>(model_path, weight_path, embedding_path, tokenizer_path, max_token_length, special_token, this);
}
