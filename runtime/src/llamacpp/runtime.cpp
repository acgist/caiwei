#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/runtime.hpp"
#include "caiwei/runtime/llamacpp.hpp"

template<>
std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> caiwei::runtime::get_runtime() {
    const int  min_pool  = caiwei::env::get_int("CAIWEI_LLAMACPP_MIN_POOL");
    const int  max_pool  = caiwei::env::get_int("CAIWEI_LLAMACPP_MAX_POOL");
    const int  timeout   = caiwei::env::get_int("CAIWEI_RUNTIME_TIMEOUT");
    const int  keepalive = caiwei::env::get_int("CAIWEI_RUNTIME_KEEPALIVE");
    const bool share     = caiwei::env::get_bool("CAIWEI_LLAMACPP_SHARE");
    return std::make_shared<caiwei::runtime::LlamaCPPRuntime>(min_pool, max_pool, timeout, keepalive, share);
}

caiwei::runtime::LlamaCPPRuntime::LlamaCPPRuntime(
    int min_pool, int max_pool, int timeout, int keepalive, bool share
) : Runtime(min_pool, max_pool, timeout, keepalive, share, caiwei::runtime::Type::LLAMACPP) {
    CW_LOG_I("创建LlamaCPPRuntime");
}

caiwei::runtime::LlamaCPPRuntime::~LlamaCPPRuntime() {
    CW_LOG_I("释放LlamaCPPRuntime");
}

std::shared_ptr<caiwei::context::ASRContext> caiwei::runtime::LlamaCPPRuntime::get_asr_context(const caiwei::context::ContextInfo* info) {
    if (!caiwei::context::check_model_path(info, 2)) {
        return nullptr;
    }
    const auto& path_list = info->path_list;
    caiwei::text::SpecialToken special_token;
    special_token.load_asr();
    return std::make_shared<caiwei::context::ASRLlamaCPPContext>(
        path_list[0],
        path_list[1],
        caiwei::env::get_int("CAIWEI_ASR_MAX_TOKEN_LENGTH"),
        std::move(special_token),
        this
    );
}

std::shared_ptr<caiwei::context::LLMContext> caiwei::runtime::LlamaCPPRuntime::get_llm_context(const caiwei::context::ContextInfo* info) {
    if (!caiwei::context::check_model_path(info)) {
        return nullptr;
    }
    caiwei::text::SpecialToken special_token;
    special_token.load_llm();
    return std::make_shared<caiwei::context::LLMLlamaCPPContext>(
        info->path,
        caiwei::env::get_int("CAIWEI_LLM_MAX_TOKEN_LENGTH"),
        std::move(special_token),
        this
    );
}

std::shared_ptr<caiwei::context::VLMContext> caiwei::runtime::LlamaCPPRuntime::get_vlm_context(const caiwei::context::ContextInfo* info) {
    if (!caiwei::context::check_model_path(info, 2)) {
        return nullptr;
    }
    const auto& path_list = info->path_list;
    caiwei::text::SpecialToken special_token;
    special_token.load_vlm();
    return std::make_shared<caiwei::context::VLMLlamaCPPContext>(
        path_list[0],
        path_list[1],
        caiwei::env::get_int("CAIWEI_VLM_MAX_TOKEN_LENGTH"),
        std::move(special_token),
        this
    );
}

std::shared_ptr<caiwei::context::EmbeddingContext> caiwei::runtime::LlamaCPPRuntime::get_embedding_context(const caiwei::context::ContextInfo* info) {
    if (!caiwei::context::check_model_path(info)) {
        return nullptr;
    }
    caiwei::text::SpecialToken special_token;
    special_token.load_embedding();
    return std::make_shared<caiwei::context::EmbeddingLlamaCPPContext>(
        info->path,
        caiwei::env::get_int("CAIWEI_EMBEDDING_MAX_TOKEN_LENGTH"),
        std::move(special_token),
        this
    );
}

std::shared_ptr<caiwei::context::RerankingContext> caiwei::runtime::LlamaCPPRuntime::get_reranking_context(const caiwei::context::ContextInfo* info) {
    if (!caiwei::context::check_model_path(info)) {
        return nullptr;
    }
    caiwei::text::SpecialToken special_token;
    special_token.load_reranking();
    return std::make_shared<caiwei::context::RerankingLlamaCPPContext>(
        info->path,
        caiwei::env::get_int("CAIWEI_RERANKING_MAX_TOKEN_LENGTH"),
        std::move(special_token),
        this
    );
}
