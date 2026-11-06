#include "caiwei/log.hpp"
#include "caiwei/tokenizer.hpp"
#include "caiwei/runtime/llamacpp.hpp"

caiwei::context::EmbeddingLlamaCPPContext::EmbeddingLlamaCPPContext(
    std::string path,
    int32_t max_token_length,
    caiwei::text::SpecialToken special_token,
    caiwei::runtime::Runtime* runtime
) : EmbeddingContext(runtime),
    LlamaCPPContext(std::move(path), max_token_length, std::move(special_token)) {
}

caiwei::context::EmbeddingLlamaCPPContext::~EmbeddingLlamaCPPContext() {
}

bool caiwei::context::EmbeddingLlamaCPPContext::load() {
    return this->load_model();
}

caiwei::text::EmbeddingResult caiwei::context::EmbeddingLlamaCPPContext::run(const caiwei::text::EmbeddingsRequest& request) {
    uint32_t prompt_tokens = 0;
    std::vector<std::vector<llama_token>> inputs;
    if (std::holds_alternative<std::string>(request.input)) {
        const auto& item = std::get<std::string>(request.input);
        inputs.push_back(caiwei::context::tokenize(this->vocab, item, true, true));
        prompt_tokens += inputs.back().size();
    } else {
        const auto& items = std::get<std::vector<std::string>>(request.input);
        for (const auto& item : items) {
            inputs.push_back(caiwei::context::tokenize(this->vocab, item, true, true));
            prompt_tokens += inputs.back().size();
        }
    }
    caiwei::text::EmbeddingResult result;
    result.result = this->pooling(inputs);
    result.prompt_tokens = prompt_tokens;
    result.total_tokens  = prompt_tokens;
    return result;
}
