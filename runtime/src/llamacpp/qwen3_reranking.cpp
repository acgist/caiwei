#include "caiwei/runtime/llamacpp.hpp"

caiwei::context::RerankingLlamaCPPContext::RerankingLlamaCPPContext(
    std::string path,
    int32_t max_token_length,
    std::string bos_key,
    std::string eos_key,
    std::string system_prompt,
    std::string instruction_prompt,
    std::string instruction_key,
    std::string query_key,
    std::string document_key,
    caiwei::text::SpecialToken special_token,
    caiwei::runtime::Runtime* runtime
) : RerankingContext(runtime),
    LlamaCPPContext(std::move(path), max_token_length, std::move(special_token)),
    bos_key(std::move(bos_key)),
    eos_key(std::move(eos_key)),
    system_prompt(std::move(system_prompt)),
    instruction_prompt(std::move(instruction_prompt)),
    instruction_key(std::move(instruction_key)),
    query_key(std::move(query_key)),
    document_key(std::move(document_key)) {
}

caiwei::context::RerankingLlamaCPPContext::~RerankingLlamaCPPContext() {
}

bool caiwei::context::RerankingLlamaCPPContext::load() {
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
    // for (int i = 0; i < size; i++) {
    //     out[i] = embd[i];
    // }
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

caiwei::text::RerankingResult caiwei::context::RerankingLlamaCPPContext::run(const caiwei::text::RerankingsRequest& request) {
    llama_context_ptr context{ get_context(true) };
    const enum llama_pooling_type pooling_type = llama_pooling_type(context.get());
    if (llama_model_has_encoder(this->model)) {
        CW_LOG_W("不支持的编码模型: %s", this->path.c_str());
        return {};
    }
    if (pooling_type != LLAMA_POOLING_TYPE_RANK) {
        CW_LOG_W("不支持的池化类型: %d", pooling_type);
        return {};
    }
    uint32_t prompt_tokens = 0;
    const int n_batch = llama_n_batch(context.get());
    std::vector<std::vector<llama_token>> inputs;
    for (const auto& document : request.documents) {
        std::string prompt;
        prompt
            .append(this->bos_key)
            .append("system\n")
            .append(this->system_prompt)
            .append(this->eos_key)
            .append("\n")
            .append(this->bos_key)
            .append("user\n")
            .append(this->instruction_key)
            .append(request.instruct.value_or(this->instruction_prompt))
            .append("\n")
            .append(this->query_key)
            .append(request.query)
            .append("\n")
            .append(this->document_key)
            .append(document)
            .append("\n")
            .append(this->eos_key)
            .append("\n")
            .append(this->bos_key)
            .append("assistant\n")
            .append(this->special_token.b_thinking)
            .append("\n\n")
            .append(this->special_token.e_thinking)
            .append("\n\n");
        CW_LOG_D("reranking prompt: %s", prompt.c_str());
        inputs.push_back(this->tokenize(prompt, context.get(), false, true));
        prompt_tokens += inputs.back().size();
    }
    if (inputs.size() > n_batch) {
        CW_LOG_W("不支持的输入数量: %d", inputs.size());
        return {};
    }
    const int n_prompts = inputs.size();
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    int n_embd_count = n_prompts;
    const int n_embd_out = llama_model_n_cls_out(this->model);
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
            emb_pos += seq_pos;
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
    caiwei::text::RerankingResult result;
    for (int i = 0; i < n_embd_count; ++i) {
        std::vector<float> item;
        item.assign(embeddings.begin() + i * n_embd_out, embeddings.begin() + (i + 1) * n_embd_out);
        result.result.push_back(std::move(item));
    }
    result.prompt_tokens = prompt_tokens;
    result.total_tokens  = prompt_tokens;
    return result;
}
