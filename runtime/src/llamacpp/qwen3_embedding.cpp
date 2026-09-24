#include "caiwei/runtime/llamacpp.hpp"

/**
 * 参考代码
 * deps\llama.cpp\examples\embedding\embedding.cpp
 */

caiwei::context::EmbeddingLlamaCPPContext::EmbeddingLlamaCPPContext(std::string path, int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime)
  : EmbeddingContext(runtime),
    LlamaCPPContext(std::move(path), max_token_length, std::move(special_token)) {
}

caiwei::context::EmbeddingLlamaCPPContext::~EmbeddingLlamaCPPContext() {
}

bool caiwei::context::EmbeddingLlamaCPPContext::load() {
    return this->load_model();
}

static void batch_add(struct llama_batch& batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id>& seq_id, bool logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_id.size();
    for (size_t i = 0; i < seq_id.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_id[i];
    }
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

static void euclidean(const float* embd, float* out, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += embd[i] * embd[i];
    }
    sum = std::sqrt(sum);
    const float norm = sum > 0.0 ? 1.0 / sum : 0.0F;
    for (int i = 0; i < size; i++) {
        out[i] = embd[i] * norm;
    }
}

static void batch_decode(llama_context* ctx, llama_batch& batch, float* output, int n_seq, int n_embd_out) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    llama_memory_clear(llama_get_memory(ctx), true);
    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        CW_LOG_W("解码失败: %d", ret);
    }
    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }
        const float* embd = nullptr;
        int embd_pos = 0;
        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            embd = llama_get_embeddings_ith(ctx, i);
            embd_pos = i;
        } else {
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            embd_pos = batch.seq_id[i][0];
        }
        if (embd == nullptr) {
            CW_LOG_W("获取嵌入失败: %d", i);
            continue;
        }
        float* out = output + embd_pos * n_embd_out;
        euclidean(embd, out, n_embd_out);
    }
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
            batch_decode(context.get(), batch, out, seq_pos, n_embd_out);
            emb_pos += pooling_type == LLAMA_POOLING_TYPE_NONE ? batch.n_tokens : seq_pos;
            seq_pos = 0;
            batch.n_tokens = 0;
        }
        for (size_t i = 0; i < n_tokens; ++i) {
            batch_add(batch, input[i], i, { seq_pos }, true);
        }
        seq_pos += 1;
    }
    float* out = emb + emb_pos * n_embd_out;
    batch_decode(context.get(), batch, out, seq_pos, n_embd_out);
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
