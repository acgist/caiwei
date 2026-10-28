#include "caiwei/log.hpp"
#include "caiwei/runtime/rknn3.hpp"

caiwei::context::EmbeddingRKNN3Context::EmbeddingRKNN3Context(
    std::string model_path,
    std::string weight_path,
    std::string embedding_path,
    std::string tokenizer_path,
    int32_t max_token_length,
    caiwei::text::SpecialToken special_token,
    caiwei::runtime::Runtime* runtime
) : EmbeddingContext(runtime),
    RKNN3Context(
        std::move(model_path),
        std::move(weight_path),
        std::move(embedding_path),
        std::move(tokenizer_path),
        max_token_length,
        std::move(special_token)
    ) {
}

caiwei::context::EmbeddingRKNN3Context::~EmbeddingRKNN3Context() {
}

bool caiwei::context::EmbeddingRKNN3Context::load() {
    return this->load_model() && this->init_output_tensors();
}

caiwei::text::EmbeddingResult caiwei::context::EmbeddingRKNN3Context::run(const caiwei::text::EmbeddingsRequest& request) {
    std::vector<const std::string*> prompt_list;
    if (std::holds_alternative<std::string>(request.input)) {
        const auto& item = std::get<std::string>(request.input);
        prompt_list.push_back(&item);
    } else if (std::holds_alternative<std::vector<std::string>>(request.input)) {
        const auto& items = std::get<std::vector<std::string>>(request.input);
        for (const auto& item : items) {
            prompt_list.emplace_back(&item);
        }
    } else {
        CW_LOG_W("不支持的输入类型");
        return {};
    }
    uint32_t prompt_tokens = 0;
    caiwei::text::EmbeddingResult result;
    result.result = this->pooling(prompt_tokens, prompt_list);
    result.prompt_tokens = prompt_tokens;
    result.total_tokens  = prompt_tokens;
    return result;
}
