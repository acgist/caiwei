#include "caiwei/runtime/rknn3.hpp"

caiwei::context::ASRRKNN3Context::ASRRKNN3Context(
    std::string model_path, std::string weight_path, std::string embedding_path, std::string tokenizer_path,
    std::string media_model_path, std::string media_weight_path,
    int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime
) : ASRContext(runtime),
    RKNN3Context(
    std::move(model_path), std::move(weight_path), std::move(embedding_path), std::move(tokenizer_path),
    std::move(media_model_path), std::move(media_weight_path), max_token_length, special_token) {
}

caiwei::context::ASRRKNN3Context::~ASRRKNN3Context() {
}

bool caiwei::context::ASRRKNN3Context::load() {
    return this->load_model() && this->load_media_model();
}

std::generator<caiwei::text::Result> caiwei::context::ASRRKNN3Context::run(caiwei::text::CompletionsRequest& request) {
    return this->generate(request);
}

std::vector<rknn3_llm_input> caiwei::context::ASRRKNN3Context::get_inputs(rknn3_session* session, ContextSession* context_session, caiwei::text::CompletionsRequest& request) {
    return {};
}
