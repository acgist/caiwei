#include "caiwei/runtime/llamacpp.hpp"

caiwei::context::LLMLlamaCPPContext::LLMLlamaCPPContext(std::string path, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime)
  : LLMContext(std::move(runtime)),
    LlamaCPPContext(std::move(path)) {
}
    
caiwei::context::LLMLlamaCPPContext::~LLMLlamaCPPContext() {
}
    
std::vector<std::string> caiwei::context::LLMLlamaCPPContext::run(const caiwei::text::CompletionsRequest& request) {
    llama_context_params context_params = llama_context_default_params();
    llama_context* context = llama_init_from_model(this->model, context_params);
    llama_sampler* sampler = nullptr;
    std::vector<llama_chat_message> messages;
    std::vector<char> formatted(llama_n_ctx(context));
    int new_len = 0;
    int prev_len = 0;
    std::string ret;
    auto generate = [&, this](const std::string & prompt) {
        std::string response;

        const bool is_first = llama_memory_seq_pos_max(llama_get_memory(context), 0) == -1;

        // tokenize the prompt
        const int n_prompt_tokens = -llama_tokenize(this->vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);
        if (llama_tokenize(this->vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
            GGML_ABORT("failed to tokenize the prompt\n");
        }

        // prepare a batch for the prompt
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        llama_token new_token_id;
        while (true) {
            // check if we have enough space in the context to evaluate this batch
            int n_ctx = llama_n_ctx(context);
            int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(context), 0) + 1;
            if (n_ctx_used + batch.n_tokens > n_ctx) {
                printf("\033[0m\n");
                fprintf(stderr, "context size exceeded\n");
                exit(0);
            }

            int ret = llama_decode(context, batch);
            if (ret != 0) {
                GGML_ABORT("failed to decode, ret = %d\n", ret);
            }

            // sample the next token
            new_token_id = llama_sampler_sample(sampler, context, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(this->vocab, new_token_id)) {
                break;
            }

            // convert the token to a string, print it and add it to the response
            char buf[256];
            int n = llama_token_to_piece(this->vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                GGML_ABORT("failed to convert token to piece\n");
            }
            std::string piece(buf, n);
            printf("%s", piece.c_str());
            fflush(stdout);
            response += piece;

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);
        }

        return response;
    };
    if (context == nullptr) {
        CW_LOG_E("LLMLlamaCPPContext上下文无效");
        goto release;
    }
    sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (sampler == nullptr) {
        CW_LOG_E("LLMLlamaCPPContext采样器无效");
        goto release;
    }
    messages.push_back({"user", "request"});
    new_len = llama_chat_apply_template(this->model_chat_template, messages.data(), messages.size(), true, formatted.data(), formatted.size());
    if (new_len > (int) formatted.size()) {
        formatted.resize(new_len);
        new_len = llama_chat_apply_template(this->model_chat_template, messages.data(), messages.size(), true, formatted.data(), formatted.size());
    }
    if (new_len < 0) {
        return {};
    }
    ret = generate(std::string(formatted.begin() + prev_len, formatted.begin() + new_len));
    prev_len = llama_chat_apply_template(this->model_chat_template, messages.data(), messages.size(), false, nullptr, 0);
    if (prev_len < 0) {
        fprintf(stderr, "failed to apply the chat template\n");
        return {};
    }
    // llama_sampler_chain_add(sampler, llama_sampler_init_temp());
    release:
    if (context != nullptr) {
        llama_free(context);
        context = nullptr;
    }
    if (sampler != nullptr) {
        llama_sampler_free(sampler);
        sampler = nullptr;
    }
    return {};
}
