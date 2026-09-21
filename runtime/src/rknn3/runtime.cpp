#include "caiwei/runtime.hpp"
#include "caiwei/runtime/rknn3.hpp"

template<>
std::shared_ptr<caiwei::runtime::RKNN3Runtime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    int min_pool  = caiwei::env::get_int("CAIWEI_RKNN3_MIN_POOL");
    int max_pool  = caiwei::env::get_int("CAIWEI_RKNN3_MAX_POOL");
    int timeout   = caiwei::env::get_int("CAIWEI_RUNTIME_TIMEOUT");
    int keepalive = caiwei::env::get_int("CAIWEI_RUNTIME_KEEPALIVE");
    return std::make_shared<caiwei::runtime::RKNN3Runtime>(min_pool, max_pool, timeout, keepalive);
}

caiwei::runtime::RKNN3Runtime::RKNN3Runtime(int min_pool, int max_pool, int timeout, int keepalive) : Runtime(min_pool, max_pool, timeout, keepalive, caiwei::runtime::Type::RKNN3) {
    CW_LOG_I("RKNN3Runtime init");
}

caiwei::runtime::RKNN3Runtime::~RKNN3Runtime() {
    CW_LOG_I("RKNN3Runtime stop");
}

std::shared_ptr<caiwei::context::LLMContext> caiwei::runtime::RKNN3Runtime::get_llm_context(const caiwei::context::ContextInfo* info) {
    std::vector<std::string> paths = info->paths;
    if (paths.size() != 4) {
        CW_LOG_W("RKNN3模型路径配置无效: %s", info->path.c_str());
        return nullptr;
    }
    std::string model_path = paths[0];
    std::string weight_path = paths[1];
    std::string embedding_path = paths[2];
    std::string tokenizer_path = paths[3];
    if (!std::filesystem::exists(model_path)) {
        CW_LOG_W("RKNN3模型无效: %s", model_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(weight_path)) {
        CW_LOG_W("RKNN3模型无效: %s", weight_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(embedding_path)) {
        CW_LOG_W("RKNN3模型无效: %s", embedding_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(tokenizer_path)) {
        CW_LOG_W("RKNN3模型无效: %s", tokenizer_path.c_str());
        return nullptr;
    }
    caiwei::text::SpecialToken special_token;
    special_token.bos = caiwei::env::get("CAIWEI_LLM_TOKEN_BOS");
    special_token.eos = caiwei::env::get("CAIWEI_LLM_TOKEN_EOS");
    special_token.pad = caiwei::env::get("CAIWEI_LLM_TOKEN_PAD");
    special_token.b_thinking = caiwei::env::get("CAIWEI_LLM_TOKEN_BTHINKING");
    special_token.e_thinking = caiwei::env::get("CAIWEI_LLM_TOKEN_ETHINKING");
    special_token.b_toolcall = caiwei::env::get("CAIWEI_LLM_TOKEN_BTOOLCALL");
    special_token.e_toolcall = caiwei::env::get("CAIWEI_LLM_TOKEN_ETOOLCALL");
    special_token.enable_thinking = caiwei::env::get("CAIWEI_LLM_ENABLE_THINKING");
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_LLM_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::LLMRKNN3Context>(model_path, weight_path, embedding_path, tokenizer_path, max_token_length, special_token, this);
}

std::shared_ptr<caiwei::context::EmbeddingContext> caiwei::runtime::RKNN3Runtime::get_embedding_context(const caiwei::context::ContextInfo* info) {
    std::vector<std::string> paths = info->paths;
    if (paths.size() != 4) {
        CW_LOG_W("RKNN3模型路径配置无效: %s", info->path.c_str());
        return nullptr;
    }
    std::string model_path = paths[0];
    std::string weight_path = paths[1];
    std::string embedding_path = paths[2];
    std::string tokenizer_path = paths[3];
    if (!std::filesystem::exists(model_path)) {
        CW_LOG_W("RKNN3模型无效: %s", model_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(weight_path)) {
        CW_LOG_W("RKNN3模型无效: %s", weight_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(embedding_path)) {
        CW_LOG_W("RKNN3模型无效: %s", embedding_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(tokenizer_path)) {
        CW_LOG_W("RKNN3模型无效: %s", tokenizer_path.c_str());
        return nullptr;
    }
    caiwei::text::SpecialToken special_token;
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_EMBEDDING_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::EmbeddingRKNN3Context>(model_path, weight_path, embedding_path, tokenizer_path, max_token_length, special_token, this);
}

std::shared_ptr<caiwei::context::RerankingContext> caiwei::runtime::RKNN3Runtime::get_reranking_context(const caiwei::context::ContextInfo* info) {
    std::vector<std::string> paths = info->paths;
    if (paths.size() != 4) {
        CW_LOG_W("RKNN3模型路径配置无效: %s", info->path.c_str());
        return nullptr;
    }
    std::string model_path = paths[0];
    std::string weight_path = paths[1];
    std::string embedding_path = paths[2];
    std::string tokenizer_path = paths[3];
    if (!std::filesystem::exists(model_path)) {
        CW_LOG_W("RKNN3模型无效: %s", model_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(weight_path)) {
        CW_LOG_W("RKNN3模型无效: %s", weight_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(embedding_path)) {
        CW_LOG_W("RKNN3模型无效: %s", embedding_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(tokenizer_path)) {
        CW_LOG_W("RKNN3模型无效: %s", tokenizer_path.c_str());
        return nullptr;
    }
    std::string system_prompt      = caiwei::env::get("CAIWEI_RERANKING_SYSTEM");
    std::string instruction_prompt = caiwei::env::get("CAIWEI_RERANKING_INSTRUCTION");
    std::string instruction_key    = caiwei::env::get("CAIWEI_RERANKING_INSTRUCTION_KEY");
    std::string query_key          = caiwei::env::get("CAIWEI_RERANKING_QUERY_KEY");
    std::string document_key       = caiwei::env::get("CAIWEI_RERANKING_DOCUMENT_KEY");
    caiwei::text::SpecialToken special_token;
    std::string bos_key = caiwei::env::get("CAIWEI_RERANKING_TOKEN_BOS");
    std::string eos_key = caiwei::env::get("CAIWEI_RERANKING_TOKEN_EOS");
    special_token.bos = bos_key;
    special_token.eos = eos_key;
    special_token.b_thinking = caiwei::env::get("CAIWEI_RERANKING_TOKEN_BTHINKING");
    special_token.e_thinking = caiwei::env::get("CAIWEI_RERANKING_TOKEN_ETHINKING");
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_RERANKING_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::RerankingRKNN3Context>(
        model_path,
        weight_path,
        embedding_path,
        tokenizer_path,
        max_token_length,
        bos_key,
        eos_key,
        system_prompt,
        instruction_prompt,
        instruction_key,
        query_key,
        document_key,
        special_token,
        this
    );
}
