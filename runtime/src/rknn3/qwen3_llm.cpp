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

std::generator<std::string> caiwei::context::LLMRKNN3Context::run(const caiwei::text::CompletionsRequest& request) {
  std::string content;
  std::string thinking;
  std::string toolcall;
  std::string finish_reason;
  for (const auto& result : this->generate(request)) {
      if (request.stream) {
          std::string ret = caiwei::text::chunk_choice(request, result);
          if (ret.empty()) {
              continue;
          }
          co_yield ret;
      } else {
        if (result.thinking) {
          thinking += result.token;
        } else if (result.toolcall) {
          toolcall += result.token;
        } else {
          content += result.token;
        }
        if (!result.finish_reason.empty()) {
            finish_reason = result.finish_reason;
        }
      }
  }
  if (!request.stream) {
    co_yield caiwei::text::response_choice(request, finish_reason, std::move(content), std::move(thinking), std::move(toolcall));
  }
}
