#include "caiwei/runtime/rknn3.hpp"

caiwei::context::LLMRKNN3Context::LLMRKNN3Context(std::string model_path, std::string weight_path, std::string embedding_path, std::string tokenizer_path, int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime)
  : LLMContext(runtime),
    RKNN3Context(std::move(model_path), std::move(weight_path), std::move(embedding_path), std::move(tokenizer_path),
    max_token_length, std::move(special_token)) {
}
    
caiwei::context::LLMRKNN3Context::~LLMRKNN3Context() {
}

bool caiwei::context::LLMRKNN3Context::load() {
    return this->load_model();
}

std::generator<caiwei::text::Result> caiwei::context::LLMRKNN3Context::run(caiwei::text::CompletionsRequest& request) {
    return this->generate(request);
}
