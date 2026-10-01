#include "caiwei/log.hpp"
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

std::generator<caiwei::text::Result> caiwei::context::LLMLlamaCPPContext::run(caiwei::text::CompletionsRequest& request) {
    return this->generate(request);
}
