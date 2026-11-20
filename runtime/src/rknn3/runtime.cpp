#include "caiwei/runtime.hpp"
#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/runtime/rknn3.hpp"

template<>
std::shared_ptr<caiwei::runtime::RKNN3Runtime> caiwei::runtime::get_runtime() {
    return std::make_shared<caiwei::runtime::RKNN3Runtime>(
        caiwei::env::get_int("CAIWEI_RKNN3_MIN_POOL"),
        caiwei::env::get_int("CAIWEI_RKNN3_MAX_POOL"),
        caiwei::env::get_int("CAIWEI_RUNTIME_TIMEOUT"),
        caiwei::env::get_int("CAIWEI_RUNTIME_KEEPALIVE"),
        caiwei::env::get_bool("CAIWEI_RKNN3_SHARE")
    );
}

caiwei::runtime::RKNN3Runtime::RKNN3Runtime(int min_pool, int max_pool, int timeout, int keepalive, bool share) :
    Runtime(min_pool, max_pool, timeout, keepalive, share, caiwei::runtime::Type::RKNN3) {
    CW_LOG_I("创建RKNN3Runtime");
}

caiwei::runtime::RKNN3Runtime::~RKNN3Runtime() {
    CW_LOG_I("释放RKNN3Runtime");
}

std::shared_ptr<caiwei::context::ASRContext>
caiwei::runtime::RKNN3Runtime::get_asr_context(const caiwei::context::ContextInfo* info) {
    if (!caiwei::context::check_model_path(info, 6)) {
        return nullptr;
    }
    const auto& path_list = info->path_list;
    caiwei::text::SpecialToken special_token;
    special_token.load_asr();
    return std::make_shared<caiwei::context::ASRRKNN3Context>(
        path_list[0],
        path_list[1],
        path_list[2],
        path_list[3],
        path_list[4],
        path_list[5],
        caiwei::env::get_int("CAIWEI_ASR_MAX_TOKEN_LENGTH"),
        std::move(special_token),
        this
    );
}

std::shared_ptr<caiwei::context::LLMContext>
caiwei::runtime::RKNN3Runtime::get_llm_context(const caiwei::context::ContextInfo* info) {
    if (!caiwei::context::check_model_path(info, 4)) {
        return nullptr;
    }
    const auto& path_list = info->path_list;
    caiwei::text::SpecialToken special_token;
    special_token.load_llm();
    return std::make_shared<caiwei::context::LLMRKNN3Context>(
        path_list[0],
        path_list[1],
        path_list[2],
        path_list[3],
        caiwei::env::get_int("CAIWEI_LLM_MAX_TOKEN_LENGTH"),
        std::move(special_token),
        this
    );
}

std::shared_ptr<caiwei::context::VLMContext>
caiwei::runtime::RKNN3Runtime::get_vlm_context(const caiwei::context::ContextInfo* info) {
    if (!caiwei::context::check_model_path(info, 6)) {
        return nullptr;
    }
    const auto& path_list = info->path_list;
    caiwei::text::SpecialToken special_token;
    special_token.load_vlm();
    return std::make_shared<caiwei::context::VLMRKNN3Context>(
        path_list[0],
        path_list[1],
        path_list[2],
        path_list[3],
        path_list[4],
        path_list[5],
        caiwei::env::get_int("CAIWEI_VLM_MAX_TOKEN_LENGTH"),
        std::move(special_token),
        this
    );
}

std::shared_ptr<caiwei::context::EmbeddingContext>
caiwei::runtime::RKNN3Runtime::get_embedding_context(const caiwei::context::ContextInfo* info) {
    if (!caiwei::context::check_model_path(info, 4)) {
        return nullptr;
    }
    const auto& path_list = info->path_list;
    caiwei::text::SpecialToken special_token;
    special_token.load_embedding();
    return std::make_shared<caiwei::context::EmbeddingRKNN3Context>(
        path_list[0],
        path_list[1],
        path_list[2],
        path_list[3],
        caiwei::env::get_int("CAIWEI_EMBEDDING_MAX_TOKEN_LENGTH"),
        std::move(special_token),
        this
    );
}

std::shared_ptr<caiwei::context::RerankingContext>
caiwei::runtime::RKNN3Runtime::get_reranking_context(const caiwei::context::ContextInfo* info) {
    if (!caiwei::context::check_model_path(info, 4)) {
        return nullptr;
    }
    const auto& path_list = info->path_list;
    caiwei::text::SpecialToken special_token;
    special_token.load_reranking();
    return std::make_shared<caiwei::context::RerankingRKNN3Context>(
        path_list[0],
        path_list[1],
        path_list[2],
        path_list[3],
        caiwei::env::get_int("CAIWEI_RERANKING_MAX_TOKEN_LENGTH"),
        std::move(special_token),
        this
    );
}
