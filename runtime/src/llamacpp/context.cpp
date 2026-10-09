#include "caiwei/log.hpp"
#include "caiwei/tokenizer.hpp"
#include "caiwei/runtime/llamacpp.hpp"

#include <algorithm>

#include "mtmd-helper.h"

caiwei::context::LlamaCPPContext::LlamaCPPContext(std::string path, int32_t max_token_length, caiwei::text::SpecialToken special_token)
  : path(std::move(path)), max_token_length(max_token_length), special_token(std::move(special_token)) {
    CW_LOG_I("创建LlamaCPPContext: %s", this->path.c_str());
}

caiwei::context::LlamaCPPContext::LlamaCPPContext(std::string path, std::string mtmd_path, std::string media_marker, int32_t max_token_length, caiwei::text::SpecialToken special_token)
  : path(std::move(path)), mtmd_path(std::move(mtmd_path)),
  media_marker(std::move(media_marker)),
  max_token_length(max_token_length), special_token(std::move(special_token)) {
    CW_LOG_I("创建LlamaCPPContext: %s", this->path.c_str());
}

caiwei::context::LlamaCPPContext::~LlamaCPPContext() {
    if (this->model) {
        CW_LOG_D("释放LlamaCPPRuntime: %s", this->path.c_str());
        llama_free_model(this->model);
        this->model = nullptr;
    }
}

bool caiwei::context::LlamaCPPContext::load_model() {
    llama_model_params params = llama_model_default_params();
    this->model = llama_model_load_from_file(this->path.c_str(), params);
    if (this->model == nullptr) {
        CW_LOG_W("加载LlamaCPPContext模型失败: %s", this->path.c_str());
        return false;
    }
    this->vocab = llama_model_get_vocab(this->model);
    CW_LOG_I("模型输入最大输入长度: %d/%d", llama_model_n_ctx_train(this->model), this->max_token_length);
    this->max_token_length = std::min(this->max_token_length, llama_model_n_ctx_train(this->model));
    this->special_token.bos = token_to_piece(this->vocab, llama_vocab_bos(this->vocab), this->special_token.bos);
    this->special_token.eos = token_to_piece(this->vocab, llama_vocab_eos(this->vocab), this->special_token.eos);
    this->special_token.pad = token_to_piece(this->vocab, llama_vocab_pad(this->vocab), this->special_token.pad);
    this->chat_template.set_template(llama_model_chat_template(this->model, nullptr), this->special_token.bos, this->special_token.eos);
    return true;
}

bool caiwei::context::LlamaCPPContext::load_mtmd() {
    if (this->media_marker.empty()) {
        this->media_marker = mtmd_default_marker();
    }
    mtmd_context_params params = mtmd_context_params_default();
    params.media_marker = this->media_marker.c_str();
    this->mtmd_context.reset(mtmd_init_from_file(this->mtmd_path.c_str(), this->model, params));
    return true;
}

llama_context* caiwei::context::LlamaCPPContext::get_context(bool embeddings) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx      = this->max_token_length;
    params.n_batch    = this->max_token_length;
    params.embeddings = embeddings;
    if (embeddings) {
        params.kv_unified = true;
        params.n_seq_max  = llama_max_parallel_sequences();
    }
    #if CAIWEI_DEBUG
    params.no_perf = false;
    #else
    params.no_perf = true;
    #endif
    return llama_init_from_model(this->model, params);
}

llama_sampler* caiwei::context::LlamaCPPContext::get_sampler(const caiwei::text::CompletionsRequest& request) {
    llama_sampler_chain_params params = llama_sampler_chain_default_params();
    #if CAIWEI_DEBUG
    params.no_perf = false;
    #else
    params.no_perf = true;
    #endif
    llama_sampler* sampler = llama_sampler_chain_init(params);
    if (sampler == nullptr) {
        return nullptr;
    }
    if (request.seed.has_value()) {
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(request.seed.value()));
    }
    if (request.presence_penalty.has_value() || request.frequency_penalty.has_value()) {
        float pres = std::clamp(request.presence_penalty.value_or(0.0F), -2.0F, 2.0F);
        float freq = std::clamp(request.frequency_penalty.value_or(0.0F), -2.0F, 2.0F);
        if (std::abs(pres) > 1e-6F || std::abs(freq) > 1e-6F) {
            int n_vocab = llama_vocab_n_tokens(this->vocab);
            llama_sampler_chain_add(sampler, llama_sampler_init_penalties(n_vocab, -1, 1.0F, pres, freq));
        }
    }
    if (request.temperature.has_value() && request.temperature.value() <= 1e-6F) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
        float temp  = std::clamp(request.temperature.value_or(1.0F), 1e-6F, 10.0F);
        float top_p = std::clamp(request.top_p.value_or(0.95F), 0.0F, 1.0F);
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
        // llama_sampler_init_top_k
        if (!request.seed.has_value()) {
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        }
    }
    return sampler;
}

std::generator<caiwei::text::Result> caiwei::context::LlamaCPPContext::generate(caiwei::text::CompletionsRequest& request) {
    llama_context_ptr context{ get_context() };
    llama_sampler_ptr sampler{ get_sampler(request) };
    if (!context || !sampler) {
        co_return;
    }
    // TODO 多模态输入数据多态实现
    std::string prompt = this->chat_template.apply(this->special_token, request);
    CW_LOG_D("提示词: %s", prompt.c_str());
    std::vector<llama_token> prompt_tokens = caiwei::context::tokenize(this->vocab, prompt);
    if (prompt_tokens.empty()) {
        co_yield caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_LENGTH, static_cast<uint32_t>(0), 0 };
        co_return;
    }
    const int n_prompt_tokens = prompt_tokens.size();
    const uint32_t n_ctx = llama_n_ctx(context.get());
    if (n_prompt_tokens > n_ctx) {
        co_yield caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_LENGTH, static_cast<uint32_t>(0), 0 };
        co_return;
    }
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_model_has_encoder(this->model)) {
        CW_LOG_W("不支持的编码模型: %s", this->path.c_str());
        co_yield caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_STOP, static_cast<uint32_t>(n_prompt_tokens), 0 };
        co_return;
    }
    uint32_t generated_tokens = 0;
    llama_token token_id;
    std::string buffer;
    buffer.resize(1024);
    llama_token b_thinking = piece_to_token(this->vocab, this->special_token.b_thinking);
    llama_token e_thinking = piece_to_token(this->vocab, this->special_token.e_thinking);
    llama_token b_toolcall = piece_to_token(this->vocab, this->special_token.b_toolcall);
    llama_token e_toolcall = piece_to_token(this->vocab, this->special_token.e_toolcall);
    bool thinking = false;
    bool toolcall = false;
    caiwei::text::ResultToolcall result_toolcall;
    uint32_t max_completion_tokens = request.max_completion_tokens.value_or(this->max_token_length);
    while (true) {
        llama_pos n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(context.get()), 0);
        if (n_ctx_used < 0) {
            n_ctx_used = 0;
        } else {
            n_ctx_used += 1;
        }
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            CW_LOG_W("上下文长度超过最大长度: %d", n_ctx);
            co_yield caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_LENGTH, static_cast<uint32_t>(n_prompt_tokens), generated_tokens };
            break;
        }
        if (generated_tokens > max_completion_tokens) {
            CW_LOG_W("生成内容超过最大长度: %d", max_completion_tokens);
            co_yield caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_LENGTH, static_cast<uint32_t>(n_prompt_tokens), generated_tokens };
            break;
        }
        int ret = llama_decode(context.get(), batch);
        if (ret != 0) {
            CW_LOG_W("解码失败: ret = %d", ret);
            co_yield caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_STOP, static_cast<uint32_t>(n_prompt_tokens), generated_tokens };
            break;
        }
        token_id = llama_sampler_sample(sampler.get(), context.get(), -1);
        if (token_id == LLAMA_TOKEN_NULL) {
            CW_LOG_W("采样返回NULL");
            co_yield caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_STOP, static_cast<uint32_t>(n_prompt_tokens), generated_tokens };
            break;
        }
        if (llama_vocab_is_eog(this->vocab, token_id)) {
            if (toolcall) {
                co_yield caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_TOOL_CALLS, static_cast<uint32_t>(n_prompt_tokens), generated_tokens };
            } else {
                co_yield caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_STOP, static_cast<uint32_t>(n_prompt_tokens), generated_tokens };
            }
            break;
        }
        ++generated_tokens;
        decode:
        int32_t buffer_length = llama_token_to_piece(this->vocab, token_id, buffer.data(), buffer.size(), 0, true);
        if (buffer_length < 0) {
            CW_LOG_W("解码失败: %d", token_id);
            co_yield caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_STOP, static_cast<uint32_t>(n_prompt_tokens), generated_tokens };
            break;
        }
        if (buffer_length > buffer.size()) {
            buffer.resize(buffer.size() + 1024);
            goto decode;
        }
        if (token_id == b_thinking) {
            thinking = true;
        } else if (token_id == e_thinking) {
            thinking = false;
        } else if (token_id == b_toolcall) {
            toolcall = true;
            result_toolcall.reset();
        } else if (token_id == e_toolcall) {
            // TOOLCALL不要修改状态
            result_toolcall.finish();
            co_yield caiwei::text::Result{ thinking, toolcall, &result_toolcall };
        } else {
            std::string token(buffer.begin(), buffer.begin() + buffer_length);
            #if CAIWEI_DEBUG
            std::printf("%s", token.c_str());
            std::fflush(stdout);
            #endif
            if (toolcall) {
                result_toolcall.put_token(std::move(token));
                co_yield caiwei::text::Result{ thinking, toolcall, &result_toolcall };
            } else {
                co_yield caiwei::text::Result{ thinking, toolcall, token };
            }
        }
        batch = llama_batch_get_one(&token_id, 1);
    }
    #if CAIWEI_DEBUG
    llama_perf_sampler_print(sampler.get());
    llama_perf_context_print(context.get());
    #endif
}

std::generator<caiwei::text::Result> caiwei::context::LlamaCPPContext::generate_mtmd(caiwei::text::CompletionsRequest& request) {
    mtmd::bitmaps bitmaps;
    this->build_bitmaps(request, bitmaps);
    llama_context_ptr context{ get_context() };
    llama_sampler_ptr sampler{ get_sampler(request) };
    if (!context || !sampler) {
        co_return;
    }
    std::string prompt = this->chat_template.apply(this->special_token, request);
//     std::string prompt = R"(<|im_start|>user
// 不要分析内容，告诉我有几张图片和几个视频。<__media__><__media__><__media__><|im_end|>
// <|im_start|>assistant)";
    CW_LOG_D("提示词: %s", prompt.c_str());
    mtmd_input_text text;
    text.text          = prompt.data();
    text.text_len      = prompt.size();
    text.add_special   = false;
    text.parse_special = true;
    llama_pos n_past = 0;
    mtmd::batch_ptr mbatch{ nullptr };
    int n_batch = 2048; // TODO
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(this->mtmd_context.get(), chunks.ptr.get(), &text, bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
    // TODO check res
    size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
    if (n_chunks == 0) {
        co_return;
    }
    for (size_t i = 0; i < n_chunks; i++) {
        auto chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
        auto chunk_type = mtmd_input_chunk_get_type(chunk);
        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            llama_pos new_n_past = n_past;
            res = mtmd_helper_eval_chunk_single(this->mtmd_context.get(), context.get(), chunk, n_past, 0, n_batch, i == n_chunks - 1, &new_n_past);
            if (res != 0) {
                CW_LOG_W("Unable to eval text chunk %zu\n", i);
                co_return;
            }
            n_past = new_n_past;
        } else {
            float* embd = nullptr;
            if (mbatch) {
                embd = mtmd_batch_get_output_embd(mbatch.get(), chunk);
                if (embd) {
                    CW_LOG_D("found embd for media chunk %zu in existing batch\n", i);
                } else {
                    CW_LOG_W("media chunk %zu not found in existing batch, creating new batch\n", i);
                }
            }
            if (!embd) {
                mbatch.reset(mtmd_batch_init(this->mtmd_context.get()));
                res = mtmd_batch_add_chunk(mbatch.get(), chunk);
                int n_added = 1;
                for (size_t j = i + 1; j < n_chunks; j++) {
                    auto next_chunk = mtmd_input_chunks_get(chunks.ptr.get(), j);
                    auto next_type = mtmd_input_chunk_get_type(next_chunk);
                    if (next_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                        break;
                    }
                    res = mtmd_batch_add_chunk(mbatch.get(), next_chunk);
                    if (res != 0) {
                        break;
                    }
                    n_added++;
                }
                int64_t time_start = ggml_time_ms();
                CW_LOG_I("encoding mtmd batch, n_chunks = %d (done = %zu, total = %zu)\n", n_added, i, n_chunks);
                res = mtmd_batch_encode(mbatch.get());
                if (res != 0) {
                    CW_LOG_E("Failed to encode mtmd batch, res = %d\n", res);
                    co_return;
                }
                CW_LOG_I("mtmd batch encoding done in %d ms\n", (int)(ggml_time_ms() - time_start));

                embd = mtmd_batch_get_output_embd(mbatch.get(), chunk);
            }
            llama_pos new_n_past = n_past;
            res = mtmd_helper_decode_image_chunk(this->mtmd_context.get(), context.get(), chunk, embd, n_past, 0, n_batch, &new_n_past, nullptr, nullptr);
            if (res != 0) {
                CW_LOG_W("Unable to decode media chunk %zu\n", i);
                co_return;
            }
            n_past = new_n_past;
        }
    }
    llama_batch batch = llama_batch_init(1, 0, 1);
    uint32_t max_completion_tokens = request.max_completion_tokens.value_or(this->max_token_length);
    llama_token token_id;
    for (int i = 0; i < max_completion_tokens; i++) {
        if (i > max_completion_tokens) {
            // TODO
            break;
        }
        token_id = llama_sampler_sample(sampler.get(), context.get(), -1);
        if (llama_vocab_is_eog(this->vocab, token_id)) {
            break;
        }
        std::string token = caiwei::context::token_to_piece(this->vocab, token_id);
        #if CAIWEI_DEBUG
        std::printf("%s", token.c_str());
        std::fflush(stdout);
        #endif
        batch.n_tokens = 0;
        caiwei::context::batch_add(batch, token_id, n_past++, {0}, true);
        if (llama_decode(context.get(), batch)) {
            CW_LOG_E("failed to decode token\n");
            co_return;
        }
    }
    llama_batch_free(batch);
    #if CAIWEI_DEBUG
    llama_perf_sampler_print(sampler.get());
    llama_perf_context_print(context.get());
    #endif
}

void caiwei::context::LlamaCPPContext::build_bitmaps(caiwei::text::CompletionsRequest& request, mtmd::bitmaps& bitmaps) {
}

void caiwei::context::batch_add(llama_batch& batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id>& seq_id, bool logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_id.size();
    for (size_t i = 0; i < seq_id.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_id[i];
    }
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

void caiwei::context::batch_decode(llama_context* ctx, llama_batch& batch, float* output, int n_seq, int n_embd_out) {
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
        caiwei::context::euclidean(embd, out, n_embd_out);
    }
}
