#include "caiwei/runtime/llamacpp.hpp"

#include <algorithm>

caiwei::context::LlamaCPPContext::LlamaCPPContext(std::string path, int32_t max_token_length, caiwei::text::SpecialToken special_token)
  : path(std::move(path)), special_token(std::move(special_token)) {
    CW_LOG_I("创建LlamaCPPContext: %s", this->path.c_str());
    llama_model_params params = llama_model_default_params();
    this->model = llama_model_load_from_file(this->path.c_str(), params);
    if (this->model == nullptr) {
        CW_LOG_W("加载LlamaCPPContext模型失败: %s", this->path.c_str());
        return;
    }
    this->vocab = llama_model_get_vocab(this->model);
    this->max_token_length = std::min(max_token_length, llama_model_n_ctx_train(this->model));
    this->special_token.bos = token_to_string(this->vocab, llama_vocab_bos(this->vocab), this->special_token.bos);
    this->special_token.eos = token_to_string(this->vocab, llama_vocab_eos(this->vocab), this->special_token.eos);
    this->special_token.pad = token_to_string(this->vocab, llama_vocab_pad(this->vocab), this->special_token.pad);
    this->chat_template.set_template(llama_model_chat_template(this->model, nullptr), this->special_token.bos, this->special_token.eos);
}

caiwei::context::LlamaCPPContext::~LlamaCPPContext() {
    if (this->model) {
        CW_LOG_D("释放LlamaCPPRuntime: %s", this->path.c_str());
        llama_free_model(this->model);
        this->model = nullptr;
    }
}

llama_context* caiwei::context::LlamaCPPContext::get_context(const caiwei::text::CompletionsRequest& request) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx   = this->max_token_length;
    params.n_batch = this->max_token_length;
#if CAIWEI_DEBUG
    params.no_perf = true;
#else
    params.no_perf = false;
#endif
    return llama_init_from_model(this->model, params);
}

llama_sampler* caiwei::context::LlamaCPPContext::get_sampler(const caiwei::text::CompletionsRequest& request) {
    llama_sampler_chain_params params = llama_sampler_chain_default_params();
#if CAIWEI_DEBUG
    params.no_perf = true;
#else
    params.no_perf = false;
#endif
    llama_sampler* sampler = llama_sampler_chain_init(params);
    if (sampler == nullptr) {
        return nullptr;
    }
    if (request.seed.has_value()) {
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(request.seed.value()));
    }
    if (request.presence_penalty.has_value() || request.frequency_penalty.has_value()) {
        float pres = std::clamp(request.presence_penalty.value_or(0.0F), -2.0F, 2.0F);
        float freq = std::clamp(request.frequency_penalty.value_or(0.0F), -2.0F, 2.0F);
        if (std::abs(pres) > 1e-6F || std::abs(freq) > 1e-6F) {
            int n_vocab = llama_vocab_n_tokens(this->vocab);
            llama_sampler_chain_add(sampler, llama_sampler_init_penalties(n_vocab, -1, 1.0F, pres, freq));
        }
    }
    if (request.temperature.has_value() && request.temperature.value() <= 1e-6F) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
        float temp  = std::clamp(request.temperature.value_or(1.0F), 1e-6F, 10.0F);
        float top_p = std::clamp(request.top_p.value_or(0.95F), 0.0F, 1.0F);
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
        if (!request.seed.has_value()) {
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        }
    }
    return sampler;
}

std::generator<caiwei::text::Result> caiwei::context::LlamaCPPContext::generate(
    llama_context* context,
    llama_sampler* sampler,
    uint32_t max_tokens,
    const std::string& prompt
) {
    const uint32_t n_ctx = llama_n_ctx(context);
    const int n_prompt_tokens = -llama_tokenize(this->vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (n_prompt_tokens > n_ctx) {
        CW_LOG_W("提示词超长 %d > %u", n_prompt_tokens, n_ctx);
        co_yield caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_STOP, static_cast<uint32_t>(n_prompt_tokens), 0 };
        co_return;
    }
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(this->vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        CW_LOG_W("提示词分词失败: %s", prompt.c_str());
        co_yield caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_STOP, static_cast<uint32_t>(n_prompt_tokens), 0 };
        co_return;
    }
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_model_has_encoder(this->model)) {
        CW_LOG_W("不支持的编码模型: %s", this->path.c_str());
        co_yield caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_STOP, static_cast<uint32_t>(n_prompt_tokens), 0 };
        co_return;
    }
    uint32_t generated_tokens = 0;
    llama_token token_id;
    std::string buffer;
    buffer.resize(1024);
    llama_token b_thinking = string_to_token(this->vocab, this->special_token.b_thinking);
    llama_token e_thinking = string_to_token(this->vocab, this->special_token.e_thinking);
    llama_token b_toolcall = string_to_token(this->vocab, this->special_token.b_toolcall);
    llama_token e_toolcall = string_to_token(this->vocab, this->special_token.e_toolcall);
    bool thinking = false;
    bool toolcall = false;
    caiwei::text::ResultToolcall result_toolcall;
    while (max_tokens == 0 || generated_tokens < max_tokens) {
        llama_pos n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(context), 0);
        if (n_ctx_used < 0) {
            n_ctx_used = 0;
        } else {
            n_ctx_used += 1;
        }
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            CW_LOG_W("上下文长度超过最大长度: %d", n_ctx);
            co_yield caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_LENGTH, static_cast<uint32_t>(n_prompt_tokens), generated_tokens };
            co_return;
        }
        int ret = llama_decode(context, batch);
        if (ret != 0) {
            CW_LOG_W("解码失败: ret = %d", ret);
            co_yield caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_STOP, static_cast<uint32_t>(n_prompt_tokens), generated_tokens };
            co_return;
        }
        token_id = llama_sampler_sample(sampler, context, -1);
        if (token_id == LLAMA_TOKEN_NULL) {
            CW_LOG_W("采样返回NULL");
            co_yield caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_STOP, static_cast<uint32_t>(n_prompt_tokens), generated_tokens };
            co_return;
        }
        if (llama_vocab_is_eog(this->vocab, token_id)) {
            if (toolcall) {
                co_yield caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_TOOL_CALLS, static_cast<uint32_t>(n_prompt_tokens), generated_tokens };
            } else {
                co_yield caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_STOP, static_cast<uint32_t>(n_prompt_tokens), generated_tokens };
            }
            co_return;
        }
        ++generated_tokens;
        decode:
        int32_t buffer_length = llama_token_to_piece(this->vocab, token_id, buffer.data(), buffer.size(), 0, true);
        if (buffer_length < 0) {
            CW_LOG_W("解码失败: %d", token_id);
            co_yield caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_STOP, static_cast<uint32_t>(n_prompt_tokens), generated_tokens };
            co_return;
        }
        if (buffer_length > buffer.size()) {
            buffer.resize(buffer.size() + 1024);
            goto decode;
        }
        if (token_id == b_thinking) {
            thinking = true;
        } else if (token_id == e_thinking) {
            thinking = false;
        } else if (token_id == b_toolcall) {
            toolcall = true;
            result_toolcall.increment();
        } else if (token_id == e_toolcall) {
            // TOOLCALL不要修改状态
            result_toolcall.finish();
            co_yield caiwei::text::Result{ thinking, toolcall, &result_toolcall };
        } else {
            std::string token(buffer.begin(), buffer.begin() + buffer_length);
            #if CAIWEI_DEBUG
            std::printf("%s", token.c_str());
            std::fflush(stdout);
            #endif
            if (toolcall) {
                result_toolcall.put_token(std::move(token));
                co_yield caiwei::text::Result{ thinking, toolcall, &result_toolcall };
            } else {
                co_yield caiwei::text::Result{ thinking, toolcall, token };
            }
        }
        batch = llama_batch_get_one(&token_id, 1);
    }
}

std::shared_ptr<caiwei::context::ClsContext> caiwei::context::get_cls_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::DetContext> caiwei::context::get_det_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::OBBContext> caiwei::context::get_obb_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::SegContext> caiwei::context::get_seg_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::SemContext> caiwei::context::get_sem_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::PoseContext> caiwei::context::get_pose_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::DepthContext> caiwei::context::get_depth_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::ASRContext> caiwei::context::get_asr_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::TTSContext> caiwei::context::get_tts_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::LLMContext> caiwei::context::get_llm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
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
    return std::make_shared<LLMLlamaCPPContext>(info->path, max_token_length, special_token, runtime);
}

std::shared_ptr<caiwei::context::VLMContext> caiwei::context::get_vlm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
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

std::shared_ptr<caiwei::context::MLLMContext> caiwei::context::get_mllm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    caiwei::text::SpecialToken special_token;
    special_token.bos = caiwei::env::get("CAIWEI_MLLM_TOKEN_BOS");
    special_token.eos = caiwei::env::get("CAIWEI_MLLM_TOKEN_EOS");
    special_token.pad = caiwei::env::get("CAIWEI_MLLM_TOKEN_PAD");
    special_token.b_audio = caiwei::env::get("CAIWEI_MLLM_TOKEN_BAUDIO");
    special_token.c_audio = caiwei::env::get("CAIWEI_MLLM_TOKEN_CAUDIO");
    special_token.e_audio = caiwei::env::get("CAIWEI_MLLM_TOKEN_EAUDIO");
    special_token.b_image = caiwei::env::get("CAIWEI_MLLM_TOKEN_BIMAGE");
    special_token.c_image = caiwei::env::get("CAIWEI_MLLM_TOKEN_CIMAGE");
    special_token.e_image = caiwei::env::get("CAIWEI_MLLM_TOKEN_EIMAGE");
    special_token.b_video = caiwei::env::get("CAIWEI_MLLM_TOKEN_BVIDEO");
    special_token.c_video = caiwei::env::get("CAIWEI_MLLM_TOKEN_CVIDEO");
    special_token.e_video = caiwei::env::get("CAIWEI_MLLM_TOKEN_EVIDEO");
    special_token.b_thinking = caiwei::env::get("CAIWEI_MLLM_TOKEN_BTHINKING");
    special_token.e_thinking = caiwei::env::get("CAIWEI_MLLM_TOKEN_ETHINKING");
    special_token.b_toolcall = caiwei::env::get("CAIWEI_MLLM_TOKEN_BTOOLCALL");
    special_token.e_toolcall = caiwei::env::get("CAIWEI_MLLM_TOKEN_ETOOLCALL");
    special_token.enable_thinking = caiwei::env::get("CAIWEI_MLLM_ENABLE_THINKING");
    uint32_t max_token_length = caiwei::env::get_int("CAIWEI_MLLM_MAX_TOKEN_LENGTH");
    return nullptr;
}

std::shared_ptr<caiwei::context::EmbeddingContext> caiwei::context::get_embedding_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::RerankingContext> caiwei::context::get_reranking_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::VLEmbeddingContext> caiwei::context::get_vl_embedding_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::VLRerankingContext> caiwei::context::get_vl_reranking_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::string caiwei::context::token_to_string(const llama_vocab* vocab, llama_token token, std::string default_token) {
    if (token == LLAMA_TOKEN_NULL) {
        return std::move(default_token);
    }
    static const int ret_length = 64;
    std::string ret;
    ret.resize(ret_length);
    const int length = llama_token_to_piece(vocab, token, ret.data(), ret.size(), 0, true);
    if (length < 0) {
        return std::move(default_token);
    } else if (length >= ret_length) {
        return std::move(default_token);
    } else {
        ret.resize(length);
        return ret;
    }
}

llama_token caiwei::context::string_to_token(const llama_vocab* vocab, const std::string& token) {
    std::vector<llama_token> prompt_tokens(1);
    if (llama_tokenize(vocab, token.c_str(), token.size(), prompt_tokens.data(), 1, true, true) < 0) {
        return LLAMA_TOKEN_NULL;
    }
    return prompt_tokens[0];
}