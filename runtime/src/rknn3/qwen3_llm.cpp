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
    rknn3_llm_tensor tensor{};
    tensor.name     = "input_embeds";
    // TODO
    tensor.prompt   = "你好，解释一下碧螺萧萧";
    tensor.embed    = NULL;
    tensor.tokens   = NULL;
    tensor.n_tokens = 0;
    tensor.enable_thinking = false;
    std::vector<rknn3_llm_input> inputs(1);
    inputs[0].input_type = RKNN3_LLM_INPUT_PROMPT;
    inputs[0].llm_input  = tensor;
    return inputs;
}
