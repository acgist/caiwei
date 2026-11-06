#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/runtime.hpp"
#include "caiwei/runtime/rknn3.hpp"

template<>
std::shared_ptr<caiwei::runtime::RKNN3Runtime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    if (type != caiwei::runtime::Type::RKNN3) {
        CW_LOG_W("类型无效: %s", get_type_name(type));
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
    if (!caiwei::context::check_model_path(info, 6)) {
        CW_LOG_W("模型无效: %s", info->path.c_str());
        return nullptr;
    }
    const auto& path_list = info->path_list;
    caiwei::text::SpecialToken special_token;
    special_token.load_asr();
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_ASR_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::ASRRKNN3Context>(
        path_list[0],
        path_list[1],
        path_list[2],
        path_list[3],
        path_list[4],
        path_list[5],
        max_token_length,
        std::move(special_token),
        this
    );
}

std::shared_ptr<caiwei::context::LLMContext> caiwei::runtime::RKNN3Runtime::get_llm_context(const caiwei::context::ContextInfo* info) {
    if (!caiwei::context::check_model_path(info, 4)) {
        CW_LOG_W("模型无效: %s", info->path.c_str());
        return nullptr;
    }
    const auto& path_list = info->path_list;
    caiwei::text::SpecialToken special_token;
    special_token.load_llm();
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_LLM_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::LLMRKNN3Context>(
        path_list[0],
        path_list[1],
        path_list[2],
        path_list[3],
        max_token_length,
        std::move(special_token),
        this
    );
}


std::shared_ptr<caiwei::context::VLMContext> caiwei::runtime::RKNN3Runtime::get_vlm_context(const caiwei::context::ContextInfo* info) {
    if (!caiwei::context::check_model_path(info, 6)) {
        CW_LOG_W("模型无效: %s", info->path.c_str());
        return nullptr;
    }
    const auto& path_list = info->path_list;
    caiwei::text::SpecialToken special_token;
    special_token.load_vlm();
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_VLM_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::VLMRKNN3Context>(
        path_list[0],
        path_list[1],
        path_list[2],
        path_list[3],
        path_list[4],
        path_list[5],
        max_token_length,
        std::move(special_token),
        this
    );
}

std::shared_ptr<caiwei::context::EmbeddingContext> caiwei::runtime::RKNN3Runtime::get_embedding_context(const caiwei::context::ContextInfo* info) {
    if (!caiwei::context::check_model_path(info, 4)) {
        CW_LOG_W("模型无效: %s", info->path.c_str());
        return nullptr;
    }
    const auto& path_list = info->path_list;
    caiwei::text::SpecialToken special_token;
    special_token.load_embedding();
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_EMBEDDING_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::EmbeddingRKNN3Context>(
        path_list[0],
        path_list[1],
        path_list[2],
        path_list[3],
        max_token_length,
        std::move(special_token),
        this
    );
}

std::shared_ptr<caiwei::context::RerankingContext> caiwei::runtime::RKNN3Runtime::get_reranking_context(const caiwei::context::ContextInfo* info) {
    if (!caiwei::context::check_model_path(info, 4)) {
        CW_LOG_W("模型无效: %s", info->path.c_str());
        return nullptr;
    }
    const auto& path_list = info->path_list;
    caiwei::text::SpecialToken special_token;
    special_token.load_reranking();
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_RERANKING_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::RerankingRKNN3Context>(
        path_list[0],
        path_list[1],
        path_list[2],
        path_list[3],
        max_token_length,
        std::move(special_token),
        this
    );
}
