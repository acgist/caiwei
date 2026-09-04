#include "caiwei/runtime/llamacpp.hpp"

caiwei::context::LLMLlamaCPPContext::LLMLlamaCPPContext(std::string path, int32_t max_token_length, caiwei::text::SpecialToken special_token, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime)
  : LLMContext(std::move(runtime)),
    LlamaCPPContext(std::move(path), max_token_length, std::move(special_token)) {
}
    
caiwei::context::LLMLlamaCPPContext::~LLMLlamaCPPContext() {
}
    
std::vector<std::string> caiwei::context::LLMLlamaCPPContext::run(const caiwei::text::CompletionsRequest& request) {
    llama_context_ptr context{ get_context(request) };
    llama_sampler_ptr sampler{ get_sampler(request) };
    std::string prompt = this->chat_template.apply(this->special_token, request);
    this->generate(context.get(), sampler.get(), request.max_tokens.value_or(0), prompt);
    #if CAIWEI_DEBUG
    llama_perf_context_print(context.get());
    llama_perf_sampler_print(sampler.get());
    #endif
    return {};
}
