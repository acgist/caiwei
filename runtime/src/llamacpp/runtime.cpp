#include "caiwei/runtime.hpp"
#include "caiwei/runtime/llamacpp.hpp"

template<>
std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> caiwei::runtime::get_runtime(caiwei::runtime::Type type) {
    int min_pool  = caiwei::env::get_int("CAIWEI_LLAMACPP_MIN_POOL");
    int max_pool  = caiwei::env::get_int("CAIWEI_LLAMACPP_MAX_POOL");
    int timeout   = caiwei::env::get_int("CAIWEI_RUNTIME_TIMEOUT");
    int keepalive = caiwei::env::get_int("CAIWEI_RUNTIME_KEEPALIVE");
    return std::make_shared<caiwei::runtime::LlamaCPPRuntime>(min_pool, max_pool, timeout, keepalive);
}

caiwei::runtime::LlamaCPPRuntime::LlamaCPPRuntime(int min_pool, int max_pool, int timeout, int keepalive) : Runtime(min_pool, max_pool, timeout, keepalive, caiwei::runtime::Type::LLAMACPP) {
    CW_LOG_I("LlamaCPPRuntime init");
}

caiwei::runtime::LlamaCPPRuntime::~LlamaCPPRuntime() {
    CW_LOG_I("LlamaCPPRuntime stop");
}

std::shared_ptr<caiwei::context::ASRContext> caiwei::runtime::LlamaCPPRuntime::get_asr_context(const caiwei::context::ContextInfo* info) {
    std::vector<std::string> paths = info->paths;
    if (paths.size() != 2) {
        CW_LOG_W("LlamaCPP模型路径配置无效: %s", info->path.c_str());
        return nullptr;
    }
    std::string model_path = paths[0];
    std::string mmproj_path = paths[1];
    if (!std::filesystem::exists(model_path)) {
        CW_LOG_W("LlamaCPP模型无效: %s", model_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(mmproj_path)) {
        CW_LOG_W("LlamaCPP模型无效: %s", mmproj_path.c_str());
        return nullptr;
    }
    caiwei::text::SpecialToken special_token;
    special_token.bos = caiwei::env::get("CAIWEI_ASR_TOKEN_BOS");
    special_token.eos = caiwei::env::get("CAIWEI_ASR_TOKEN_EOS");
    special_token.pad = caiwei::env::get("CAIWEI_ASR_TOKEN_PAD");
    special_token.b_audio = caiwei::env::get("CAIWEI_ASR_TOKEN_BAUDIO");
    special_token.c_audio = caiwei::env::get("CAIWEI_ASR_TOKEN_CAUDIO");
    special_token.e_audio = caiwei::env::get("CAIWEI_ASR_TOKEN_EAUDIO");
    special_token.b_thinking = caiwei::env::get("CAIWEI_ASR_TOKEN_BTHINKING");
    special_token.e_thinking = caiwei::env::get("CAIWEI_ASR_TOKEN_ETHINKING");
    special_token.b_toolcall = caiwei::env::get("CAIWEI_ASR_TOKEN_BTOOLCALL");
    special_token.e_toolcall = caiwei::env::get("CAIWEI_ASR_TOKEN_ETOOLCALL");
    special_token.enable_thinking = caiwei::env::get("CAIWEI_ASR_ENABLE_THINKING");
    std::string media_marker = caiwei::env::get("CAIWEI_ASR_MEDIA_MARKER");
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_ASR_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::ASRLlamaCPPContext>(model_path, mmproj_path, media_marker, max_token_length, special_token, this);
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
    std::vector<std::string> paths = info->paths;
    if (paths.size() != 2) {
        CW_LOG_W("LlamaCPP模型路径配置无效: %s", info->path.c_str());
        return nullptr;
    }
    std::string model_path = paths[0];
    std::string mmproj_path = paths[1];
    if (!std::filesystem::exists(model_path)) {
        CW_LOG_W("LlamaCPP模型无效: %s", model_path.c_str());
        return nullptr;
    }
    if (!std::filesystem::exists(mmproj_path)) {
        CW_LOG_W("LlamaCPP模型无效: %s", mmproj_path.c_str());
        return nullptr;
    }
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
    std::string media_marker = caiwei::env::get("CAIWEI_ASR_MEDIA_MARKER");
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_VLM_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::VLMLlamaCPPContext>(model_path, mmproj_path, media_marker, max_token_length, special_token, this);
}

std::shared_ptr<caiwei::context::EmbeddingContext> caiwei::runtime::LlamaCPPRuntime::get_embedding_context(const caiwei::context::ContextInfo* info) {
    if (!std::filesystem::exists(info->path)) {
        CW_LOG_W("LlamaCPP模型无效: %s", info->path.c_str());
        return nullptr;
    }
    caiwei::text::SpecialToken special_token;
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_EMBEDDING_MAX_TOKEN_LENGTH");
    return std::make_shared<caiwei::context::EmbeddingLlamaCPPContext>(info->path, max_token_length, special_token, this);
}

std::shared_ptr<caiwei::context::RerankingContext> caiwei::runtime::LlamaCPPRuntime::get_reranking_context(const caiwei::context::ContextInfo* info) {
    if (!std::filesystem::exists(info->path)) {
        CW_LOG_W("LlamaCPP模型无效: %s", info->path.c_str());
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
    return std::make_shared<caiwei::context::RerankingLlamaCPPContext>(
        info->path,
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
