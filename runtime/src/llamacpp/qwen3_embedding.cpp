#include "caiwei/runtime/llamacpp.hpp"

// caiwei::context::EmbeddingLlamaCPPContext::EmbeddingLlamaCPPContext() {
// }

// caiwei::context::EmbeddingLlamaCPPContext::~EmbeddingLlamaCPPContext() {
// }

// bool caiwei::context::EmbeddingLlamaCPPContext::load() {
//     return this->load_model();
// }

// void common_batch_add(
//     struct llama_batch & batch,
//            llama_token   id,
//              llama_pos   pos,
//             const std::vector<llama_seq_id> & seq_ids,
//                   bool   logits) {
//     GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

//     batch.token   [batch.n_tokens] = id;
//     batch.pos     [batch.n_tokens] = pos;
//     batch.n_seq_id[batch.n_tokens] = seq_ids.size();
//     for (size_t i = 0; i < seq_ids.size(); ++i) {
//     batch.seq_id[batch.n_tokens][i] = seq_ids[i];
//     }
//     batch.logits  [batch.n_tokens] = logits;

//     batch.n_tokens++;
// }

// static void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, llama_seq_id seq_id) {
//     size_t n_tokens = tokens.size();
//     for (size_t i = 0; i < n_tokens; i++) {
//         common_batch_add(batch, tokens[i], i, { seq_id }, true);
//     }
// }

// static void euclidean(const float * inp, float * out, int n) {
//     double sum = 0.0;
//     for (int i = 0; i < n; i++) {
//         sum += inp[i] * inp[i];
//     }
//     sum = std::sqrt(sum);
// }

// static void batch_decode(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd_out) {
//     const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

//     // clear previous kv_cache values (irrelevant for embeddings)
//     llama_memory_clear(llama_get_memory(ctx), true);

//     // run model
//     if (llama_decode(ctx, batch) < 0) {
//     }

//     for (int i = 0; i < batch.n_tokens; i++) {
//         if (!batch.logits[i]) {
//             continue;
//         }

//         const float * embd = nullptr;
//         int embd_pos = 0;

//         if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
//             // try to get token embeddings
//             embd = llama_get_embeddings_ith(ctx, i);
//             embd_pos = i;
//             GGML_ASSERT(embd != NULL && "failed to get token embeddings");
//         } else {
//             // try to get sequence embeddings - supported only when pooling_type is not NONE
//             embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
//             embd_pos = batch.seq_id[i][0];
//             GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");
//         }

//         float * out = output + embd_pos * n_embd_out;
        
//         euclidean(embd, out, n_embd_out);
//     }
// }

// std::string caiwei::context::EmbeddingLlamaCPPContext::run(const caiwei::text::EmbeddingRequest& request) {
//     llama_context_ptr context{ get_context() };
//     const enum llama_pooling_type pooling_type = llama_pooling_type(context.get());
//     if (pooling_type != LLAMA_POOLING_TYPE_MEAN) {
//         return "-";
//     }
//     if (llama_model_has_encoder(this->model) || llama_model_has_decoder(this->model)) {
//         CW_LOG_W("不支持的编码模型: %s", this->path.c_str());
//         return "-";
//     }
//     const int n_batch = llama_n_batch(context.get());
//     std::vector<std::vector<llama_token>> inputs;
//     if (std::holds_alternative<caiwei::text::EmbeddingRequestInputItem>(request.input)) {
//         const auto& item = std::get<caiwei::text::EmbeddingRequestInputItem>(request.input);
//         if (std::holds_alternative<std::string>(item)) {
//             inputs.push_back(this->tokenize(std::get<std::string>(item), context.get()));
//         } else if (std::holds_alternative<caiwei::text::EmbeddingRequestInputContent>(item)) {
//             const auto& x = std::get<caiwei::text::EmbeddingRequestInputContent>(item);
//             inputs.push_back(this->tokenize(x.text.value_or(""), context.get()));
//         } else {
//             // =
//         }
//     } else if (std::holds_alternative<std::vector<caiwei::text::EmbeddingRequestInputItem>>(request.input)) {
//         const auto& items = std::get<std::vector<caiwei::text::EmbeddingRequestInputItem>>(request.input);
//         for (const auto& item : items) {
//             if (std::holds_alternative<std::string>(item)) {
//                 inputs.push_back(this->tokenize(std::get<std::string>(item), context.get()));
//             } else if (std::holds_alternative<caiwei::text::EmbeddingRequestInputContent>(item)) {
//                 const auto& x = std::get<caiwei::text::EmbeddingRequestInputContent>(item);
//                 inputs.push_back(this->tokenize(x.text.value_or(""), context.get()));
//             } else {
//                 // =
//             }
//         }
//     } else {
//         // =
//     }
//     if (inputs.size() > n_batch) {
//         return "-";
//     }
//     const int n_prompts = inputs.size();
//     struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
//     int n_embd_count = 0;
//     if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
//         for (int k = 0; k < n_prompts; k++) {
//             n_embd_count += inputs[k].size();
//         }
//     } else {
//         n_embd_count = n_prompts;
//     }
//     const int n_embd_out = llama_model_n_embd_out(model);
//     std::vector<float> embeddings(n_embd_count * n_embd_out, 0);
//     float * emb = embeddings.data();
//     const int n_seq_max = llama_max_parallel_sequences();
//     // break into batches
//     int e = 0; // number of embeddings already stored
//     int s = 0; // number of prompts in current batch
//     for (int k = 0; k < n_prompts; k++) {
//         // clamp to n_batch tokens
//         auto & inp = inputs[k];

//         const uint64_t n_toks = inp.size();

//         // encode if at capacity
//         if (batch.n_tokens + n_toks > n_batch || s >= n_seq_max) {
//             float * out = emb + e * n_embd_out;
//             batch_decode(context.get(), batch, out, s, n_embd_out );
//             e += pooling_type == LLAMA_POOLING_TYPE_NONE ? batch.n_tokens : s;
//             s = 0;
//             batch.n_tokens = 0;
//         }
//         // add to batch
//         batch_add_seq(batch, inp, s);
//         s += 1;
//     }

//     // final batch
//     float * out = emb + e * n_embd_out;
//     batch_decode(context.get(), batch, out, s, n_embd_out);
//     llama_perf_context_print(context.get());
//     llama_batch_free(batch);
// }
