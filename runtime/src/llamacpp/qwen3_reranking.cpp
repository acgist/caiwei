#include "caiwei/log.hpp"
#include "caiwei/runtime/llamacpp.hpp"
#include "caiwei/tokenizer.hpp"

caiwei::context::RerankingLlamaCPPContext::RerankingLlamaCPPContext(
    std::string path, int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime
) :
    RerankingContext(runtime),
    LlamaCPPContext(std::move(path), max_token_length, std::move(special_token)) {
}

caiwei::context::RerankingLlamaCPPContext::~RerankingLlamaCPPContext() {
}

bool caiwei::context::RerankingLlamaCPPContext::load() {
    this->release();
    if (!this->load_model()) {
        this->release();
        return false;
    }
    return true;
}

caiwei::text::RerankingResult caiwei::context::RerankingLlamaCPPContext::run(const caiwei::text::RerankingsRequest& request) {
    uint32_t prompt_tokens = 0;
    std::vector<std::vector<llama_token>> inputs;
    for (const auto& document : request.documents) {
        std::string prompt = caiwei::text::get_reranking_prompt(document, request, this->special_token);
        auto tokens = caiwei::context::tokenize(this->vocab, prompt, false, true);
        if (tokens.size() > static_cast<size_t>(this->max_token_length)) {
            CW_LOG_W("词元超量截断: %zu => %d", tokens.size(), this->max_token_length);
            tokens.resize(this->max_token_length);
        }
        prompt_tokens += tokens.size();
        inputs.push_back(std::move(tokens));
    }
    caiwei::text::RerankingResult result {};
    result.result = this->pooling(inputs);
    // clang-format off
    result.prompt_tokens = prompt_tokens;
    result.total_tokens  = prompt_tokens;
    // clang-format on
    return result;
}
