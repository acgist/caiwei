#include "caiwei/log.hpp"
#include "caiwei/runtime/rknn3.hpp"

caiwei::context::RerankingRKNN3Context::RerankingRKNN3Context(
    std::string model_path,
    std::string weight_path,
    std::string embedding_path,
    std::string tokenizer_path,
    int32_t max_token_length,
    caiwei::text::SpecialToken special_token,
    caiwei::runtime::Runtime* runtime
) : RerankingContext(runtime),
    RKNN3Context(
        std::move(model_path),
        std::move(weight_path),
        std::move(embedding_path),
        std::move(tokenizer_path),
        max_token_length,
        std::move(special_token)
    ) {
}

caiwei::context::RerankingRKNN3Context::~RerankingRKNN3Context() {
}

bool caiwei::context::RerankingRKNN3Context::load() {
    return this->load_model() && this->init_output_tensors();
}

caiwei::text::RerankingResult caiwei::context::RerankingRKNN3Context::run(const caiwei::text::RerankingsRequest& request) {
    std::vector<std::string> document_list;
    for (const auto& document : request.documents) {
        std::string prompt = caiwei::text::get_reranking_prompt(document, request, this->special_token);
        document_list.emplace_back(std::move(prompt));
        // 不能直接取址
    }
    std::vector<const std::string*> prompt_list;
    for (const auto& document: document_list) {
        prompt_list.emplace_back(&document);
    }
    uint32_t prompt_tokens = 0;
    caiwei::text::RerankingResult result;
    result.result = this->pooling(prompt_tokens, prompt_list);
    result.prompt_tokens = prompt_tokens;
    result.total_tokens  = prompt_tokens;
    return result;
}
