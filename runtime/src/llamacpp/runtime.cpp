#include "caiwei/runtime.hpp"
#include "caiwei/runtime/llamacpp.hpp"

template<>
std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    int min_pool = caiwei::env::get_int("CAIWEI_LLAMACPP_MIN_POOL");
    int max_pool = caiwei::env::get_int("CAIWEI_LLAMACPP_MAX_POOL");
    return std::make_shared<caiwei::runtime::LlamaCPPRuntime>(min_pool, max_pool);
}

caiwei::runtime::LlamaCPPRuntime::LlamaCPPRuntime(int min_pool, int max_pool) : Runtime(min_pool, max_pool, caiwei::runtime::Type::LLAMACPP) {
    CW_LOG_I("LlamaCPPRuntime init");
}

caiwei::runtime::LlamaCPPRuntime::~LlamaCPPRuntime() {
    CW_LOG_I("LlamaCPPRuntime stop");
}


std::shared_ptr<caiwei::context::LLMContext> caiwei::runtime::LlamaCPPRuntime::get_llm_context(const caiwei::context::ContextInfo* info) {
    if (!std::filesystem::exists(info->path)) {
        CW_LOG_W("LlamaCPP模型无效: %s", info->path.c_str());
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
    return std::make_shared<caiwei::context::LLMLlamaCPPContext>(info->path, max_token_length, special_token, this);
}

std::shared_ptr<caiwei::context::VLMContext> caiwei::runtime::LlamaCPPRuntime::get_vlm_context(const caiwei::context::ContextInfo* info) {
    caiwei::text::SpecialToken special_token;
    special_token.bos = caiwei::env::get("CAIWEI_VLM_TOKEN_BOS");
    special_token.eos = caiwei::env::get("CAIWEI_VLM_TOKEN_EOS");
    special_token.pad = caiwei::env::get("CAIWEI_VLM_TOKEN_PAD");
    special_token.b_image = caiwei::env::get("CAIWEI_VLM_TOKEN_BIMAGE");
    special_token.c_image = caiwei::env::get("CAIWEI_VLM_TOKEN_CIMAGE");
    special_token.e_image = caiwei::env::get("CAIWEI_VLM_TOKEN_EIMAGE");
    special_token.b_video = caiwei::env::get("CAIWEI_VLM_TOKEN_BVIDEO");
    special_token.c_video = caiwei::env::get("CAIWEI_VLM_TOKEN_CVIDEO");
    special_token.e_video = caiwei::env::get("CAIWEI_VLM_TOKEN_EVIDEO");
    special_token.b_thinking = caiwei::env::get("CAIWEI_VLM_TOKEN_BTHINKING");
    special_token.e_thinking = caiwei::env::get("CAIWEI_VLM_TOKEN_ETHINKING");
    special_token.b_toolcall = caiwei::env::get("CAIWEI_VLM_TOKEN_BTOOLCALL");
    special_token.e_toolcall = caiwei::env::get("CAIWEI_VLM_TOKEN_ETOOLCALL");
    special_token.enable_thinking = caiwei::env::get("CAIWEI_VLM_ENABLE_THINKING");
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_VLM_MAX_TOKEN_LENGTH");
    return nullptr;
}
