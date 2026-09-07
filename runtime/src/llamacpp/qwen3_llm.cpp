#include "caiwei/runtime/llamacpp.hpp"

caiwei::context::LLMLlamaCPPContext::LLMLlamaCPPContext(std::string path, int32_t max_token_length, caiwei::text::SpecialToken special_token, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime)
  : LLMContext(std::move(runtime)),
    LlamaCPPContext(std::move(path), max_token_length, std::move(special_token)) {
}
    
caiwei::context::LLMLlamaCPPContext::~LLMLlamaCPPContext() {
}
    
std::generator<std::string> caiwei::context::LLMLlamaCPPContext::run(const caiwei::text::CompletionsRequest& request) {
    llama_context_ptr context{ get_context(request) };
    llama_sampler_ptr sampler{ get_sampler(request) };
    std::string prompt = this->chat_template.apply(this->special_token, request);
    std::string content;
    std::string thinking;
    std::string toolcall;
    std::string finish_reason;
    for (const auto& result : this->generate(context.get(), sampler.get(), request.max_tokens.value_or(0), prompt)) {
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
    #if CAIWEI_DEBUG
    llama_perf_context_print(context.get());
    llama_perf_sampler_print(sampler.get());
    #endif
}
