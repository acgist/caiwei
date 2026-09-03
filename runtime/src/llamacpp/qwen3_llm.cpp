#include "caiwei/runtime/llamacpp.hpp"

caiwei::context::LLMLlamaCPPContext::LLMLlamaCPPContext(std::string path, int32_t max_token_length, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime)
  : LLMContext(std::move(runtime)),
    LlamaCPPContext(std::move(path), max_token_length) {
}
    
caiwei::context::LLMLlamaCPPContext::~LLMLlamaCPPContext() {
}
    
std::vector<std::string> caiwei::context::LLMLlamaCPPContext::run(const caiwei::text::CompletionsRequest& request) {
    llama_context* context = get_context(request);
    llama_sampler* sampler = get_sampler(request);
    // if (context == nullptr) {
    //     CW_LOG_E("LLMLlamaCPPContext上下文无效");
    //     goto release;
    // }
    // if (sampler == nullptr) {
    //     CW_LOG_E("LLMLlamaCPPContext采样器无效");
    //     goto release;
    // }
    // TODO THINK START END
    // TODO TOOLS START END
    std::string prompt = this->chat_template.apply(request);
    // llama_sampler_chain_add(sampler, llama_sampler_init_temp());
    this->generate(context, sampler, request.max_generate_tokens(), prompt, request.stop_tokens());
    #if CAIWEI_DEBUG
    llama_perf_context_print(context);
    llama_perf_sampler_print(sampler);
    #endif
    release:
    release_chat(&context, &sampler);
    return {};
}
