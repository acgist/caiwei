#include "caiwei/log.hpp"
#include "caiwei/runtime/llamacpp.hpp"

caiwei::context::EmbeddingLlamaCPPContext::EmbeddingLlamaCPPContext(std::string path, int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime)
  : EmbeddingContext(runtime),
    LlamaCPPContext(std::move(path), max_token_length, std::move(special_token)) {
}

caiwei::context::EmbeddingLlamaCPPContext::~EmbeddingLlamaCPPContext() {
}

bool caiwei::context::EmbeddingLlamaCPPContext::load() {
    return this->load_model();
}

caiwei::text::EmbeddingResult caiwei::context::EmbeddingLlamaCPPContext::run(caiwei::text::EmbeddingsRequest& request) {
    llama_context_ptr context{ get_context(true) };
    const enum llama_pooling_type pooling_type = llama_pooling_type(context.get());
    if (llama_model_has_encoder(this->model)) {
        CW_LOG_W("不支持的编码模型: %s", this->path.c_str());
        return {};
    }
    uint32_t prompt_tokens = 0;
    const int n_batch = llama_n_batch(context.get());
    std::vector<std::vector<llama_token>> inputs;
    if (std::holds_alternative<std::string>(request.input)) {
        const auto& item = std::get<std::string>(request.input);
        inputs.push_back(this->tokenize(item, context.get(), false, false));
        prompt_tokens += inputs.back().size();
    } else if (std::holds_alternative<std::vector<std::string>>(request.input)) {
        const auto& items = std::get<std::vector<std::string>>(request.input);
        for (const auto& item : items) {
            inputs.push_back(this->tokenize(item, context.get(), false, false));
            prompt_tokens += inputs.back().size();
        }
    } else {
        CW_LOG_W("不支持的输入类型");
    }
    if (inputs.size() > n_batch) {
        CW_LOG_W("不支持的输入数量: %d", inputs.size());
        return {};
    }
    const int n_prompts = inputs.size();
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    int n_embd_count = 0;
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        for (int k = 0; k < n_prompts; k++) {
            n_embd_count += inputs[k].size();
        }
    } else {
        n_embd_count = n_prompts;
    }
    const int n_embd_out = llama_model_n_embd_out(this->model);
    std::vector<float> embeddings(n_embd_count * n_embd_out, 0);
    float* emb = embeddings.data();
    const int n_seq_max = llama_max_parallel_sequences();
    int emb_pos = 0;
    int seq_pos = 0;
    for (int k = 0; k < n_prompts; ++k) {
        auto& input = inputs[k];
        const size_t n_tokens = input.size();
        if (batch.n_tokens + n_tokens > n_batch || seq_pos >= n_seq_max) {
            float* out = emb + emb_pos * n_embd_out;
            caiwei::context::batch_decode(context.get(), batch, out, seq_pos, n_embd_out);
            emb_pos += pooling_type == LLAMA_POOLING_TYPE_NONE ? batch.n_tokens : seq_pos;
            seq_pos = 0;
            batch.n_tokens = 0;
        }
        for (size_t i = 0; i < n_tokens; ++i) {
            caiwei::context::batch_add(batch, input[i], i, { seq_pos }, true);
        }
        seq_pos += 1;
    }
    float* out = emb + emb_pos * n_embd_out;
    caiwei::context::batch_decode(context.get(), batch, out, seq_pos, n_embd_out);
#if CAIWEI_DEBUG
    llama_perf_context_print(context.get());
#endif
    llama_batch_free(batch);
    caiwei::text::EmbeddingResult result;
    for (int i = 0; i < n_embd_count; ++i) {
        std::vector<float> item;
        item.assign(embeddings.begin() + i * n_embd_out, embeddings.begin() + (i + 1) * n_embd_out);
        result.result.push_back(std::move(item));
    }
    result.prompt_tokens = prompt_tokens;
    result.total_tokens  = prompt_tokens;
    return result;
}
