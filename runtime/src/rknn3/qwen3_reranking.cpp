#include "caiwei/runtime/rknn3.hpp"

// 1.0.0不能正常使用

caiwei::context::RerankingRKNN3Context::RerankingRKNN3Context(
    std::string model_path,
    std::string weight_path,
    std::string embedding_path,
    std::string tokenizer_path,
    int32_t max_token_length,
    std::string bos_key,
    std::string eos_key,
    std::string system_prompt,
    std::string instruction_prompt,
    std::string instruction_key,
    std::string query_key,
    std::string document_key,
    caiwei::text::SpecialToken special_token,
    caiwei::runtime::Runtime* runtime
) : RerankingContext(runtime),
    RKNN3Context(std::move(model_path), std::move(weight_path), std::move(embedding_path), std::move(tokenizer_path), max_token_length, std::move(special_token)),
    bos_key(std::move(bos_key)),
    eos_key(std::move(eos_key)),
    system_prompt(std::move(system_prompt)),
    instruction_prompt(std::move(instruction_prompt)),
    instruction_key(std::move(instruction_key)),
    query_key(std::move(query_key)),
    document_key(std::move(document_key)) {
}

caiwei::context::RerankingRKNN3Context::~RerankingRKNN3Context() {
}

bool caiwei::context::RerankingRKNN3Context::load() {
    return this->load_model();
}

static void euclidean(const float* embd, float* out, int size) {
    if (size <= 1) {
        return;
    }
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += embd[i] * embd[i];
    }
    sum = std::sqrt(sum);
    const float norm = sum > 0.0 ? 1.0 / sum : 0.0F;
    for (int i = 0; i < size; i++) {
        out[i] = embd[i] * norm;
    }
}

caiwei::text::RerankingResult caiwei::context::RerankingRKNN3Context::run(caiwei::text::RerankingsRequest& request) {
    // TODO 变成生成模型了
    ContextSession context_session;
    context_session.context = this->context;
    context_session.tokenizer = this->tokenizer;
    context_session.embedding_dim = this->embedding_dim;
    context_session.embedding_data = this->embedding_data;
    context_session.init_output_tensors(1);
    rknn3_session_ptr session{ this->get_session({}, &context_session) };
    if (!session) {
        CW_LOG_W("获取RKNN3会话失败");
        return {};
    }
    caiwei::text::RerankingResult result;
    for (const auto& document : request.documents) {
        std::string prompt;
        prompt
            .append(this->bos_key)
            .append("system\n")
            .append(this->system_prompt)
            .append(this->eos_key)
            .append("\n")
            .append(this->bos_key)
            .append("user\n")
            .append(this->instruction_key)
            .append(request.instruct.value_or(this->instruction_prompt))
            .append("\n")
            .append(this->query_key)
            .append(request.query)
            .append("\n")
            .append(this->document_key)
            .append(document)
            .append("\n")
            .append(this->eos_key)
            .append("\n")
            .append(this->bos_key)
            .append("assistant\n")
            .append(this->special_token.b_thinking)
            .append("\n\n")
            .append(this->special_token.e_thinking)
            .append("\n\n");
        CW_LOG_D("reranking prompt:\n%s", prompt.c_str());
        std::vector<rknn3_llm_input> inputs(1);
        rknn3_llm_tensor tensor{};
        tensor.name     = "input_embeds";
        // TODO
        tensor.prompt   = prompt.c_str();
        tensor.embed    = NULL;
        tensor.tokens   = NULL;
        tensor.n_tokens = 0;
        tensor.enable_thinking = false;
        inputs[0].input_type = RKNN3_LLM_INPUT_PROMPT;
        inputs[0].llm_input  = tensor;
        rknn3_llm_infer_param llm_infer_param;
        llm_infer_param.keep_history = 0;
        llm_infer_param.max_new_tokens = this->max_token_length;
        context_session.llm_begin_time = std::chrono::system_clock::now();
        context_session.first = true;
        int ret = rknn3_session_run(session.get(), inputs.data(), inputs.size(), &llm_infer_param);
        context_session.llm_end_time = std::chrono::system_clock::now();
        if (ret < 0) {
            CW_LOG_W("RKNN3会话运行失败: %d", ret);
        } else {
            euclidean(context_session.model_output[0].data(), context_session.model_output[0].data(), context_session.model_output[0].size());
            result.result.push_back(std::move(context_session.model_output[0]));
            CW_LOG_I("RKNN3会话完成: %d = %d = %d", ret, context_session.n_decode_tokens, context_session.n_prefill_tokens);
            #ifdef CAIWEI_DEBUG
            RKLLMRunState state{};
            ret = rknn3_session_query_state(session.get(), &state);
            if (ret < 0) {
                CW_LOG_W("RKNN3会话查询状态失败: %d", ret);
            } else {
                CW_LOG_I("RKNN3会话完成: %d = %d = %d", ret, state.n_decode_tokens, state.n_prefill_tokens);
                // context_session.n_decode_tokens  = state.n_decode_tokens;
                // context_session.n_prefill_tokens = state.n_prefill_tokens;
                printf_session_perf(&context_session);
            }
            #endif
        }
    }
    result.prompt_tokens = context_session.n_prefill_tokens;
    result.total_tokens  = context_session.n_prefill_tokens;
    return result;
}
