
#include <algorithm>

#include "llama-cpp.h"
#include "mtmd-helper.h"

#include "caiwei/log.hpp"
#include "caiwei/tokenizer.hpp"
#include "caiwei/runtime/llamacpp.hpp"

static void batch_add(llama_batch& batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id>& seq_id, bool logits);
static void batch_decode_pooling(llama_context* ctx, llama_batch& batch, const enum llama_pooling_type& pooling_type, float* output, int n_seq, int n_embd_out);

caiwei::context::caiwei_llama_batch::~caiwei_llama_batch() {
    if (this->mtmd) {
        llama_batch_free(this->batch);
    }
}

caiwei::context::LlamaCPPContext::LlamaCPPContext(
    std::string path,
    int32_t max_token_length,
    caiwei::text::SpecialToken special_token
) : path(std::move(path)),
    max_token_length(max_token_length),
    special_token(std::move(special_token)) {
    CW_LOG_I("创建LlamaCPPContext: %s", this->path.c_str());
}

caiwei::context::LlamaCPPContext::LlamaCPPContext(
    std::string path,
    std::string mtmd_path,
    int32_t max_token_length,
    caiwei::text::SpecialToken special_token
) : path(std::move(path)),
    mtmd_path(std::move(mtmd_path)),
    max_token_length(max_token_length),
    special_token(std::move(special_token)) {
    CW_LOG_I("创建LlamaCPPContext: %s", this->path.c_str());
}

caiwei::context::LlamaCPPContext::~LlamaCPPContext() {
    CW_LOG_I("释放LlamaCPPContext: %s", this->path.c_str());
    if (this->model) {
        CW_LOG_D("释放LlamaCPPContext model");
        llama_model_free(this->model);
        this->model = nullptr;
    }
    if (this->mtmd_model) {
        CW_LOG_D("释放LlamaCPPContext mtmd_model");
        mtmd_free(this->mtmd_model);
        this->mtmd_model = nullptr;
    }
}

bool caiwei::context::LlamaCPPContext::load_model() {
    if (this->model) {
        return true;
    }
    llama_model_params params = llama_model_default_params();
    this->model = llama_model_load_from_file(this->path.c_str(), params);
    if (this->model == nullptr) {
        CW_LOG_W("模型加载失败: %s", this->path.c_str());
        return false;
    }
    this->vocab = llama_model_get_vocab(this->model);
    if (this->vocab == nullptr) {
        CW_LOG_W("模型加载失败: %s", this->path.c_str());
        return false;
    }
    CW_LOG_I("模型加载成功: %s", this->path.c_str());
    this->chat_template.set_template(llama_model_chat_template(this->model, nullptr), this->special_token.bos, this->special_token.eos);
    const int32_t n_ctx_train = llama_model_n_ctx_train(this->model);
    const int32_t min_ctx_len = std::min(this->max_token_length, n_ctx_train);
    CW_LOG_I("模型最大输入长度: min(%d/%d) = %d", this->max_token_length, n_ctx_train, min_ctx_len);
    this->max_token_length = min_ctx_len;
    return true;
}

bool caiwei::context::LlamaCPPContext::load_mtmd_model() {
    if (this->mtmd_model) {
        return true;
    }
    if (this->special_token.media_marker.empty()) {
        this->special_token.media_marker = mtmd_default_marker();
        CW_LOG_W("设置默认媒体标记: %s", this->special_token.media_marker.c_str());
    }
    mtmd_context_params params = mtmd_context_params_default();
    params.media_marker = this->special_token.media_marker.c_str();
    this->mtmd_model = mtmd_init_from_file(this->mtmd_path.c_str(), this->model, params);
    if (this->mtmd_model == nullptr) {
        CW_LOG_W("模型加载失败: %s", this->mtmd_path.c_str());
        return false;
    }
    CW_LOG_I("模型加载成功: %s", this->mtmd_path.c_str());
    return true;
}

llama_context* caiwei::context::LlamaCPPContext::get_context(bool embeddings) const {
    llama_context_params params = llama_context_default_params();
    params.n_ctx      = this->max_token_length;
    params.n_batch    = this->max_token_length;
    params.embeddings = embeddings;
    if (embeddings) {
        params.n_seq_max  = llama_max_parallel_sequences();
        params.kv_unified = true;
    }
    #if ENABLE_CAIWEI_PERF
    params.no_perf = false;
    #else
    params.no_perf = true;
    #endif
    return llama_init_from_model(this->model, params);
}

llama_sampler* caiwei::context::LlamaCPPContext::get_sampler(const caiwei::text::CompletionsRequest& request) const {
    llama_sampler_chain_params params = llama_sampler_chain_default_params();
    #if ENABLE_CAIWEI_PERF
    params.no_perf = false;
    #else
    params.no_perf = true;
    #endif
    llama_sampler* sampler = llama_sampler_chain_init(params);
    if (sampler == nullptr) {
        return nullptr;
    }
    if (request.presence_penalty.has_value() || request.frequency_penalty.has_value() || request.repeat_penalty.has_value()) {
        const float presence_penalty  = std::clamp(request.presence_penalty .value_or(0.0F), -2.0F, 2.0F);
        const float frequency_penalty = std::clamp(request.frequency_penalty.value_or(0.0F), -2.0F, 2.0F);
        const float repeat_penalty    = std::clamp(request.repeat_penalty   .value_or(1.04F), 0.1F, 2.0F);    
        if (std::abs(presence_penalty) > 1e-6F || std::abs(frequency_penalty) > 1e-6F || std::abs(repeat_penalty - 1.0F) > 1e-6F) {
            llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
                llama_vocab_n_tokens(this->vocab),
                std::clamp<int32_t>(this->max_token_length, 1, 256),
                repeat_penalty,
                frequency_penalty,
                presence_penalty
            ));
        }
    }
    if (request.temperature.has_value() && request.temperature.value() <= 1e-6F) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
        const int   top_k       = std::clamp(request.top_k.value_or(10), 0, 100);
        const float top_p       = std::clamp(request.top_p.value_or(0.95F), 0.0F, 1.0F);
        const float temperature = std::clamp(request.temperature.value_or(1.0F), 1e-6F, 1.0F);
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
        if (request.seed.has_value()) {
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(request.seed.value()));
        } else {
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        }
    }
    return sampler;
}

caiwei::context::caiwei_llama_batch caiwei::context::LlamaCPPContext::get_inputs(std::string& /* prompt */, llama_context* /* context */, const caiwei::text::CompletionsRequest& /* request */) const {
    return {};
}

bool caiwei::context::LlamaCPPContext::get_mtmd_inputs(caiwei_llama_batch& caiwei_batch, mtmd::bitmaps& bitmaps, const std::string& prompt, llama_context* context, const caiwei::text::CompletionsRequest& /* request */) const {
    mtmd_input_text text;
    text.text          = prompt.data();
    text.text_len      = prompt.size();
    text.add_special   = false;
    text.parse_special = true;
    auto bitmaps_ptr   = bitmaps.c_ptr();
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    int32_t ret = mtmd_tokenize(this->mtmd_model, chunks.ptr.get(), &text, bitmaps_ptr.data(), bitmaps_ptr.size());
    if (ret != 0) {
        CW_LOG_W("分词失败: %d = %s", ret, prompt.c_str());
        return false;
    }
    const size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
    if (n_chunks == 0) {
        CW_LOG_W("分词失败: %zu", n_chunks);
        return false;
    }
    int32_t   n_batch    = this->max_token_length;
    llama_pos n_pos      = 0;
    llama_pos n_pos_chunk = 0;
    mtmd::batch_ptr mtmd_batch{ nullptr };
    for (size_t i = 0; i < n_chunks; i++) {
        auto* chunk      = mtmd_input_chunks_get(chunks.ptr.get(), i);
        auto  chunk_type = mtmd_input_chunk_get_type(chunk);
        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            CW_LOG_D("文本嵌入: %zu / %zu", i + 1, n_chunks);
            ret = mtmd_helper_eval_chunk_single(this->mtmd_model, context, chunk, n_pos, 0, n_batch, i == n_chunks - 1, &n_pos_chunk);
            if (ret != 0) {
                CW_LOG_W("文本嵌入失败: %zu = %d", i, ret);
                caiwei_batch.n_prompt_tokens = n_pos;
                return false;
            }
            n_pos = n_pos_chunk;
        } else {
            float* embd = nullptr;
            if (mtmd_batch) {
                embd = mtmd_batch_get_output_embd(mtmd_batch.get(), chunk);
            }
            if (embd == nullptr) {
                mtmd_batch.reset(mtmd_batch_init(this->mtmd_model));
                ret = mtmd_batch_add_chunk(mtmd_batch.get(), chunk);
                if (ret != 0) {
                    CW_LOG_W("添加媒体失败: %zu = %d", i, ret);
                    caiwei_batch.n_prompt_tokens = n_pos;
                    return false;
                }
                int n_batch_chunk = 1;
                for (size_t j = i + 1; j < n_chunks; j++) {
                    auto* next_chunk      = mtmd_input_chunks_get(chunks.ptr.get(), j);
                    auto  next_chunk_type = mtmd_input_chunk_get_type(next_chunk);
                    if (next_chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                        break;
                    }
                    ret = mtmd_batch_add_chunk(mtmd_batch.get(), next_chunk);
                    if (ret != 0) {
                        CW_LOG_W("添加媒体失败: %zu = %d", j, ret);
                        caiwei_batch.n_prompt_tokens = n_pos;
                        return false;
                    }
                    n_batch_chunk++;
                }
                CW_LOG_D("媒体编码: %zu / %zu = %d", i + 1, n_chunks, n_batch_chunk);
                ret = mtmd_batch_encode(mtmd_batch.get());
                if (ret != 0) {
                    CW_LOG_W("媒体编码失败: %zu = %d", i, ret);
                    caiwei_batch.n_prompt_tokens = n_pos;
                    return false;
                }
                embd = mtmd_batch_get_output_embd(mtmd_batch.get(), chunk);
            }
            CW_LOG_D("媒体嵌入: %zu / %zu", i + 1, n_chunks);
            ret = mtmd_helper_decode_image_chunk(this->mtmd_model, context, chunk, embd, n_pos, 0, n_batch, &n_pos_chunk, nullptr, nullptr);
            if (ret != 0) {
                CW_LOG_W("媒体嵌入失败: %zu = %d", i, ret);
                caiwei_batch.n_prompt_tokens = n_pos;
                return false;
            }
            n_pos = n_pos_chunk;
        }
    }
    caiwei_batch.n_pos = n_pos;
    caiwei_batch.batch = llama_batch_init(1, 0, 1);
    caiwei_batch.mtmd  = true;
    caiwei_batch.n_prompt_tokens = n_pos;
    return true;
}

std::vector<std::vector<float>> caiwei::context::LlamaCPPContext::pooling(const std::vector<std::vector<llama_token>>& inputs) const {
    llama_context_ptr context{ this->get_context(true) };
    const int n_batch = llama_n_batch(context.get());
    const int n_prompts = inputs.size();
    if (n_prompts > n_batch) {
        CW_LOG_W("不支持的输入数量: %d > %d", n_prompts, n_batch);
        return {};
    }
    const enum llama_pooling_type pooling_type = llama_pooling_type(context.get());
    int n_embd_out   = 0;
    int n_embd_count = 0;
    if (pooling_type == LLAMA_POOLING_TYPE_LAST) {
        n_embd_out   = llama_model_n_embd_out(this->model);
        n_embd_count = n_prompts;
    } else if (pooling_type == LLAMA_POOLING_TYPE_RANK) {
        n_embd_out   = llama_model_n_cls_out(this->model);
        n_embd_count = n_prompts;
    } else if (pooling_type == LLAMA_POOLING_TYPE_MEAN) {
        n_embd_out   = llama_model_n_embd_out(this->model);
        n_embd_count = n_prompts;
    } else if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        n_embd_out = llama_model_n_embd_out(this->model);
        for (int k = 0; k < n_prompts; k++) {
            n_embd_count += inputs[k].size();
        }
    } else {
        CW_LOG_W("不支持的池化类型: %d", pooling_type);
        return {};
    }
    int emb_pos = 0;
    int seq_pos = 0;
    const int seq_max = llama_max_parallel_sequences();
    std::vector<float> embeddings(n_embd_count * n_embd_out, 0);
    float* emb = embeddings.data();
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (int k = 0; k < n_prompts; ++k) {
        auto& input = inputs[k];
        const size_t n_tokens = input.size();
        if (n_tokens > static_cast<size_t>(n_batch)) {
            CW_LOG_W("单个嵌入词元超长: %zu > %d", n_tokens, n_batch);
            llama_batch_free(batch);
            return {};
        }
        if (batch.n_tokens + n_tokens > static_cast<size_t>(n_batch) || seq_pos >= seq_max) {
            float* out = emb + emb_pos * n_embd_out;
            batch_decode_pooling(context.get(), batch, pooling_type, out, seq_pos, n_embd_out);
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
    batch_decode_pooling(context.get(), batch, pooling_type, out, seq_pos, n_embd_out);
    llama_batch_free(batch);
    #if ENABLE_CAIWEI_PERF
    llama_perf_context_print(context.get());
    #endif
    std::vector<std::vector<float>> result;
    for (int i = 0; i < n_embd_count; ++i) {
        std::vector<float> item;
        item.assign(embeddings.begin() + i * n_embd_out, embeddings.begin() + (i + 1) * n_embd_out);
        result.push_back(std::move(item));
    }
    return result;
}

std::generator<caiwei::text::CompletionResult> caiwei::context::LlamaCPPContext::generate(const caiwei::text::CompletionsRequest& request) const {
    llama_context_ptr context{ this->get_context()        };
    llama_sampler_ptr sampler{ this->get_sampler(request) };
    if (context == nullptr || sampler == nullptr) {
        CW_LOG_W("创建上下文或采样器失败");
        co_yield caiwei::text::CompletionResult{ caiwei::text::FINISH_REASON_STOP, 0, 0 };
        co_return;
    }
    std::string prompt;
    caiwei_llama_batch caiwei_batch = this->get_inputs(prompt, context.get(), request);
    const uint32_t n_prompt_tokens = caiwei_batch.n_prompt_tokens;
    if (!caiwei_batch.success) {
        co_yield caiwei::text::CompletionResult{ caiwei_batch.finish_reason, n_prompt_tokens, 0 };
        co_return;
    }
    CW_LOG_D("提示词: %s", prompt.c_str());
    const uint32_t n_ctx = llama_n_ctx(context.get());
    if (n_prompt_tokens > n_ctx) {
        CW_LOG_W("提示词超长: %d > %d", n_prompt_tokens, n_ctx);
        co_yield caiwei::text::CompletionResult{ caiwei::text::FINISH_REASON_LENGTH, n_prompt_tokens, 0 };
        co_return;
    }
    llama_batch& batch = caiwei_batch.batch;
    TokenBuffer token_buffer;
    llama_token token_id;
    bool thinking = false;
    bool toolcall = false;
    const llama_token b_thinking = caiwei::context::piece_to_token(this->vocab, this->special_token.b_thinking);
    const llama_token e_thinking = caiwei::context::piece_to_token(this->vocab, this->special_token.e_thinking);
    const llama_token b_toolcall = caiwei::context::piece_to_token(this->vocab, this->special_token.b_toolcall);
    const llama_token e_toolcall = caiwei::context::piece_to_token(this->vocab, this->special_token.e_toolcall);
    caiwei::text::ToolcallSession toolcall_session;
    uint32_t generated_tokens = 0;
    const uint32_t max_completion_tokens = request.max_completion_tokens.value_or(this->max_token_length);
    auto raii_guard = caiwei::context::RaiiGuard([&]() {
        #if ENABLE_CAIWEI_PERF
        llama_perf_sampler_print(sampler.get());
        llama_perf_context_print(context.get());
        #endif
    });
    while (true) {
        if (generated_tokens > max_completion_tokens) {
            CW_LOG_W("生成内容超过最大长度: %d", max_completion_tokens);
            co_yield caiwei::text::CompletionResult{ caiwei::text::FINISH_REASON_LENGTH, n_prompt_tokens, generated_tokens };
            break;
        }
        token_id = llama_sampler_sample(sampler.get(), context.get(), -1);
        if (token_id == LLAMA_TOKEN_NULL) {
            CW_LOG_W("采样失败: %d", token_id);
            co_yield caiwei::text::CompletionResult{ caiwei::text::FINISH_REASON_STOP, n_prompt_tokens, generated_tokens };
            break;
        }
        ++generated_tokens;
        if (llama_vocab_is_eog(this->vocab, token_id)) {
            #if ENABLE_CAIWEI_PERF
            std::printf("\n");
            std::fflush(stdout);
            #endif
            CW_LOG_D("采样结束: %d", token_id);
            co_yield caiwei::text::CompletionResult{ thinking, toolcall, n_prompt_tokens, generated_tokens };
            break;
        }
        if (token_id == b_thinking) {
            thinking = true;
        } else if (token_id == e_thinking) {
            thinking = false;
        } else if (token_id == b_toolcall) {
            toolcall = true;
            toolcall_session.reset();
        } else if (token_id == e_toolcall) {
            toolcall_session.finish();
            co_yield caiwei::text::CompletionResult{ thinking, toolcall, "", toolcall_session };
        } else {
            const int32_t buffer_length = caiwei::context::token_to_piece(this->vocab, token_id, token_buffer.get_buffer(), token_buffer.get_length());
            if (buffer_length < 0) {
                CW_LOG_W("解码失败: %d", buffer_length);
                co_yield caiwei::text::CompletionResult{ caiwei::text::FINISH_REASON_STOP, n_prompt_tokens, generated_tokens };
                break;
            }
            std::string token = token_buffer.get_token(buffer_length);
            #if ENABLE_CAIWEI_PERF
            std::printf("%s", token.c_str());
            std::fflush(stdout);
            #endif
            co_yield caiwei::text::CompletionResult{ thinking, toolcall, std::move(token), toolcall_session };
        }
        if (caiwei_batch.mtmd) {
            batch.n_tokens = 0;
            batch_add(batch, token_id, caiwei_batch.n_pos++, { 0 }, true);
        } else {
            batch.n_tokens = 1;
            batch.token = &token_id;
        }
        const int32_t ret = llama_decode(context.get(), batch);
        if (ret == 1) {
            CW_LOG_W("解码失败: %d", ret);
            co_yield caiwei::text::CompletionResult{ caiwei::text::FINISH_REASON_LENGTH, n_prompt_tokens, generated_tokens };
            break;
        } else if (ret != 0) {
            CW_LOG_W("解码失败: %d", ret);
            co_yield caiwei::text::CompletionResult{ caiwei::text::FINISH_REASON_STOP, n_prompt_tokens, generated_tokens };
            break;
        } else {
            // -
        }
    }
}

static void batch_add(llama_batch& batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id>& seq_id, bool logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_id.size();
    for (size_t i = 0; i < seq_id.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_id[i];
    }
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

static void batch_decode_pooling(llama_context* ctx, llama_batch& batch, const enum llama_pooling_type& pooling_type, float* output, int /* n_seq */, int n_embd_out) {
    llama_memory_clear(llama_get_memory(ctx), true);
    int32_t ret = llama_decode(ctx, batch);
    if (ret != 0) {
        CW_LOG_W("解码失败: %d", ret);
        return;
    }
    for (int32_t i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }
        const float* embd = nullptr;
        int32_t embd_pos = 0;
        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            embd = llama_get_embeddings_ith(ctx, i);
            embd_pos = i;
        } else {
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            embd_pos = batch.seq_id[i][0];
        }
        if (embd == nullptr) {
            CW_LOG_W("嵌入失败: %d", i);
            continue;
        }
        float* out = output + embd_pos * n_embd_out;
        caiwei::context::euclidean(embd, out, n_embd_out);
    }
}
