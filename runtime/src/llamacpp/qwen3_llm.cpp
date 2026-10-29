#include "caiwei/log.hpp"
#include "caiwei/tokenizer.hpp"
#include "caiwei/runtime/llamacpp.hpp"

caiwei::context::LLMLlamaCPPContext::LLMLlamaCPPContext(
    std::string path,
    int32_t max_token_length,
    caiwei::text::SpecialToken special_token,
    caiwei::runtime::Runtime* runtime
) : LLMContext(runtime),
    LlamaCPPContext(std::move(path), max_token_length, std::move(special_token)) {
}
    
caiwei::context::LLMLlamaCPPContext::~LLMLlamaCPPContext() {
}

bool caiwei::context::LLMLlamaCPPContext::load() {
    return this->load_model();
}

std::generator<caiwei::text::CompletionResult> caiwei::context::LLMLlamaCPPContext::run(const caiwei::text::CompletionsRequest& request) {
    return this->generate(request);
}

caiwei::context::caiwei_llama_batch caiwei::context::LLMLlamaCPPContext::get_inputs(std::string& prompt, llama_context* context, const caiwei::text::CompletionsRequest& request) const {
    caiwei_llama_batch caiwei_batch;
    prompt = this->chat_template.apply(this->special_token, request);
    if (prompt.empty()) {
        CW_LOG_W("提示词为空");
        caiwei_batch.success = false;
        caiwei_batch.finish_reason = caiwei::text::FINISH_REASON_STOP;
        return caiwei_batch;
    }
    std::vector<llama_token> prompt_tokens = caiwei::context::tokenize(this->vocab, prompt);
    if (prompt_tokens.empty()) {
        CW_LOG_W("提示词为空");
        caiwei_batch.success = false;
        caiwei_batch.finish_reason = caiwei::text::FINISH_REASON_STOP;
        return caiwei_batch;
    }
    caiwei_batch.n_prompt_tokens = prompt_tokens.size();
    const uint32_t n_batch = llama_n_batch(context);
    if (caiwei_batch.n_prompt_tokens > n_batch) {
        CW_LOG_W("提示词超长: %d > %d", caiwei_batch.n_prompt_tokens, n_batch);
        caiwei_batch.success = false;
        caiwei_batch.finish_reason = caiwei::text::FINISH_REASON_LENGTH;
        return caiwei_batch;
    }
    caiwei_batch.batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    const int32_t ret = llama_decode(context, caiwei_batch.batch);
    if (ret == 1) {
        CW_LOG_W("解码失败: %d", ret);
        caiwei_batch.success = false;
        caiwei_batch.finish_reason = caiwei::text::FINISH_REASON_LENGTH;
        return caiwei_batch;
    } else if (ret != 0) {
        CW_LOG_W("解码失败: %d", ret);
        caiwei_batch.success = false;
        caiwei_batch.finish_reason = caiwei::text::FINISH_REASON_STOP;
        return caiwei_batch;
    } else {
        // -
    }
    return caiwei_batch;
}
