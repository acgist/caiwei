#include "caiwei/log.hpp"
#include "caiwei/runtime/rknn3.hpp"

// 1.0.0不能正常使用

caiwei::context::RerankingRKNN3Context::RerankingRKNN3Context(
    std::string model_path,
    std::string weight_path,
    std::string embedding_path,
    std::string tokenizer_path,
    int32_t max_token_length,
    caiwei::text::SpecialToken special_token,
    caiwei::runtime::Runtime* runtime
) : RerankingContext(runtime),
    RKNN3Context(std::move(model_path), std::move(weight_path), std::move(embedding_path), std::move(tokenizer_path), max_token_length, std::move(special_token)) {
}

caiwei::context::RerankingRKNN3Context::~RerankingRKNN3Context() {
}

bool caiwei::context::RerankingRKNN3Context::load() {
    return this->load_model() && this->init_output_tensors(1);
}

caiwei::text::RerankingResult caiwei::context::RerankingRKNN3Context::run(const caiwei::text::RerankingsRequest& request) {
    ContextSession context_session;
    context_session.tokenizer = this->tokenizer;
    context_session.embedding_dim = this->embedding_dim;
    context_session.embedding_data = this->embedding_data;
    context_session.model_output.resize(this->output_tensors.size());
    for (int i = 0; i < this->output_tensors.size(); ++i) {
        context_session.model_output[i].resize(this->output_tensors[i].attr->n_elems);
    }
    rknn3_session_ptr session{ this->get_session(nullptr, &context_session) };
    if (!session) {
        CW_LOG_W("获取RKNN3会话失败");
        return {};
    }
    caiwei::text::RerankingResult result;
    for (const auto& document : request.documents) {
        std::string prompt = caiwei::text::get_reranking_prompt(document, request, this->special_token);
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
        auto begin_time = std::chrono::steady_clock::now();
        int ret = rknn3_session_run(session.get(), inputs.data(), inputs.size(), &llm_infer_param);
        auto end_time = std::chrono::steady_clock::now();
        context_session.prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count();
        if (ret < 0) {
            CW_LOG_W("RKNN3会话运行失败: %d", ret);
        } else {
            caiwei::context::euclidean(context_session.model_output[0].data(), context_session.model_output[0].data(), context_session.model_output[0].size());
            result.result.push_back(std::move(context_session.model_output[0]));
            CW_LOG_I("RKNN3会话完成: %d = %d = %d", ret, context_session.n_decode_tokens, context_session.n_prefill_tokens);
            #ifdef CAIWEI_DEBUG
            RKLLMRunState state{};
            ret = rknn3_session_query_state(session.get(), &state);
            if (ret < 0) {
                CW_LOG_W("RKNN3会话查询状态失败: %d", ret);
            } else {
                CW_LOG_I("RKNN3会话完成: %d = %d = %d", ret, state.n_decode_tokens, state.n_prefill_tokens);
                context_session.n_decode_tokens  = state.n_decode_tokens;
                context_session.n_prefill_tokens = state.n_prefill_tokens;
                printf_session_perf(&context_session);
            }
            #endif
        }
    }
    result.prompt_tokens = context_session.n_prefill_tokens;
    result.total_tokens  = context_session.n_prefill_tokens;
    return result;
}
