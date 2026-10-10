#include "caiwei/log.hpp"
#include "caiwei/runtime/rknn3.hpp"

caiwei::context::LLMRKNN3Context::LLMRKNN3Context(std::string model_path, std::string weight_path, std::string embedding_path, std::string tokenizer_path, int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime)
  : LLMContext(runtime),
    RKNN3Context(std::move(model_path), std::move(weight_path), std::move(embedding_path), std::move(tokenizer_path), max_token_length, std::move(special_token)) {
}
    
caiwei::context::LLMRKNN3Context::~LLMRKNN3Context() {
}

bool caiwei::context::LLMRKNN3Context::load() {
    return this->load_model();
}

std::generator<caiwei::text::Result> caiwei::context::LLMRKNN3Context::run(caiwei::text::CompletionsRequest& request) {
    return this->generate(request);
}

std::vector<rknn3_llm_input> caiwei::context::LLMRKNN3Context::get_inputs(rknn3_session* session, ContextSession* context_session, caiwei::text::CompletionsRequest& request) {
    bool enable_thinking = false;
    if (request.extra_body.has_value()) {
        enable_thinking = request.extra_body.value().enable_thinking.value_or(false);
    }
    this->prompt = this->chat_template.apply(this->special_token, request);
    CW_LOG_I("prompt: %s", this->prompt.c_str());
    rknn3_llm_tensor tensor{};
    tensor.name     = "input_embeds";
    tensor.prompt   = this->prompt.c_str();
    tensor.embed    = nullptr;
    tensor.tokens   = nullptr;
    tensor.n_tokens = 0;
    tensor.enable_thinking = enable_thinking;
    std::vector<rknn3_llm_input> inputs(1);
    inputs[0].input_type = RKNN3_LLM_INPUT_PROMPT;
    inputs[0].llm_input  = tensor;
    return inputs;
}
