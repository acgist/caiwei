#include "caiwei/log.hpp"
#include "caiwei/runtime/llamacpp.hpp"

caiwei::context::RerankingLlamaCPPContext::RerankingLlamaCPPContext(
    std::string path,
    int32_t max_token_length,
    caiwei::text::SpecialToken special_token,
    caiwei::runtime::Runtime* runtime
) : RerankingContext(runtime),
    LlamaCPPContext(std::move(path), max_token_length, std::move(special_token)) {
}

caiwei::context::RerankingLlamaCPPContext::~RerankingLlamaCPPContext() {
}

bool caiwei::context::RerankingLlamaCPPContext::load() {
    return this->load_model();
}

caiwei::text::RerankingResult caiwei::context::RerankingLlamaCPPContext::run(const caiwei::text::RerankingsRequest& request) {
    uint32_t prompt_tokens = 0;
    std::vector<std::vector<llama_token>> inputs;
    for (const auto& document : request.documents) {
        std::string prompt = caiwei::text::get_reranking_prompt(document, request, this->special_token);
        inputs.push_back(caiwei::context::tokenize(this->vocab, prompt, false, true));
        prompt_tokens += inputs.back().size();
    }
    caiwei::text::RerankingResult result;
    result.result = this->pooling(inputs);
    result.prompt_tokens = prompt_tokens;
    result.total_tokens  = prompt_tokens;
    return result;
}
