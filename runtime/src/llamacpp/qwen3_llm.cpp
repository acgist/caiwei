#include "caiwei/runtime/llamacpp.hpp"

caiwei::context::LLMLlamaCPPContext::LLMLlamaCPPContext(std::string path, int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime)
  : LLMContext(runtime),
    LlamaCPPContext(std::move(path), max_token_length, std::move(special_token)) {
}
    
caiwei::context::LLMLlamaCPPContext::~LLMLlamaCPPContext() {
}

bool caiwei::context::LLMLlamaCPPContext::load() {
    return this->load_model();
}

std::generator<std::string> caiwei::context::LLMLlamaCPPContext::run(const caiwei::text::CompletionsRequest& request) {
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
