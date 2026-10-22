#include <algorithm>

#include "caiwei/log.hpp"
#include "caiwei/tokenizer.hpp"
#include "caiwei/runtime/llamacpp.hpp"

#include "llama-cpp.h"

#include "mtmd.h"
#include "mtmd-helper.h"

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
        CW_LOG_D("释放LlamaCPPContext模型: %s", this->path.c_str());
        llama_free_model(this->model);
        this->model = nullptr;
    }
    if (this->mtmd_model) {
        CW_LOG_D("释放LlamaCPPContext模型: %s", this->mtmd_path.c_str());
        mtmd_free(this->mtmd_model);
        this->mtmd_model = nullptr;
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
    if (this->vocab == nullptr) {
        CW_LOG_W("获取LlamaCPPContext模型词汇表失败: %s", this->path.c_str());
        return false;
    }
    CW_LOG_D("加载LlamaCPPContext模型成功: %s", this->path.c_str());
    const int32_t n_ctx_train = llama_model_n_ctx_train(this->model);
    CW_LOG_I("模型最大输入长度: %d/%d", this->max_token_length, n_ctx_train);
    this->max_token_length = std::min(this->max_token_length, n_ctx_train);
    this->chat_template.set_template(llama_model_chat_template(this->model, nullptr), this->special_token.bos, this->special_token.eos);
    return true;
}

bool caiwei::context::LlamaCPPContext::load_mtmd_model() {
    if (this->special_token.media_marker.empty()) {
        this->special_token.media_marker = mtmd_default_marker();
        CW_LOG_W("设置默认媒体标记: %s", this->special_token.media_marker.c_str());
    }
    mtmd_context_params params = mtmd_context_params_default();
    params.media_marker = this->special_token.media_marker.c_str();
    this->mtmd_model = mtmd_init_from_file(this->mtmd_path.c_str(), this->model, params);
    if (this->mtmd_model == nullptr) {
        CW_LOG_W("加载LlamaCPPContext模型失败: %s", this->mtmd_path.c_str());
        return false;
    }
    CW_LOG_D("加载LlamaCPPContext模型成功: %s", this->mtmd_path.c_str());
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
        const float repeat_penalty    = std::clamp(request.repeat_penalty   .value_or(1.04F), 0.0F, 1.2F);    
        if (std::abs(presence_penalty) > 1e-6F || std::abs(frequency_penalty) > 1e-6F || std::abs(repeat_penalty) > 1e-6F) {
            llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
                llama_vocab_n_tokens(this->vocab),
                -1,
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

std::generator<caiwei::text::CompletionResult> caiwei::context::LlamaCPPContext::generate(const caiwei::text::CompletionsRequest& request) const {
    if (llama_model_has_encoder(this->model)) {
        CW_LOG_W("不支持的模型: %s", this->path.c_str());
        co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_STOP, 0, 0 };
        co_return;
    }
    llama_context_ptr context{ this->get_context()        };
    llama_sampler_ptr sampler{ this->get_sampler(request) };
    if (context == nullptr || sampler == nullptr) {
        CW_LOG_W("创建上下文或采样器失败");
        co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_STOP, 0, 0 };
        co_return;
    }
    std::string prompt = this->chat_template.apply(this->special_token, request);
    CW_LOG_D("提示词: %s", prompt.c_str());
    std::vector<llama_token> prompt_tokens = caiwei::context::tokenize(this->vocab, prompt);
    if (prompt_tokens.empty()) {
        CW_LOG_W("提示词为空");
        co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_LENGTH, 0, 0 };
        co_return;
    }
    const uint32_t n_ctx = llama_n_ctx(context.get());
    const uint32_t n_prompt_tokens = prompt_tokens.size();
    if (n_prompt_tokens > n_ctx) {
        CW_LOG_W("提示词超长: %d > %d", n_prompt_tokens, n_ctx);
        co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_LENGTH, n_prompt_tokens, 0 };
        co_return;
    }
    std::string buffer;
    buffer.resize(1024);
    llama_token token_id;
    bool thinking = false;
    bool toolcall = false;
    const llama_token b_thinking = caiwei::context::piece_to_token(this->vocab, this->special_token.b_thinking);
    const llama_token e_thinking = caiwei::context::piece_to_token(this->vocab, this->special_token.e_thinking);
    const llama_token b_toolcall = caiwei::context::piece_to_token(this->vocab, this->special_token.b_toolcall);
    const llama_token e_toolcall = caiwei::context::piece_to_token(this->vocab, this->special_token.e_toolcall);
    caiwei::text::ToolcallSession toolcall_session;
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    uint32_t generated_tokens = 0;
    const uint32_t max_completion_tokens = request.max_completion_tokens.value_or(this->max_token_length);
    while (true) {
        llama_pos n_ctx_pos = llama_memory_seq_pos_max(llama_get_memory(context.get()), 0);
        if (n_ctx_pos < 0) {
            n_ctx_pos = 0;
        } else {
            n_ctx_pos += 1;
        }
        if (n_ctx_pos + batch.n_tokens > n_ctx) {
            CW_LOG_W("上下文超过最大长度: %d", n_ctx);
            co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_LENGTH, n_prompt_tokens, generated_tokens };
            break;
        }
        if (generated_tokens > max_completion_tokens) {
            CW_LOG_W("生成内容超过最大长度: %d", max_completion_tokens);
            co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_LENGTH, n_prompt_tokens, generated_tokens };
            break;
        }
        int32_t ret = llama_decode(context.get(), batch);
        if (ret != 0) {
            CW_LOG_W("解码失败: %d", ret);
            co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_STOP, n_prompt_tokens, generated_tokens };
            break;
        }
        token_id = llama_sampler_sample(sampler.get(), context.get(), -1);
        if (token_id == LLAMA_TOKEN_NULL) {
            CW_LOG_W("采样失败: %d", token_id);
            co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_STOP, n_prompt_tokens, generated_tokens };
            break;
        }
        if (llama_vocab_is_eog(this->vocab, token_id)) {
            #if ENABLE_CAIWEI_PERF
            std::printf("\n");
            std::fflush(stdout);
            #endif
            CW_LOG_D("采样结束: %d", token_id);
            if (toolcall) {
                co_yield caiwei::text::CompletionResult{ thinking, toolcall, caiwei::text::FINISH_REASON_TOOL_CALLS, n_prompt_tokens, generated_tokens };
            } else {
                co_yield caiwei::text::CompletionResult{ thinking, toolcall, caiwei::text::FINISH_REASON_STOP, n_prompt_tokens, generated_tokens };
            }
            break;
        }
        ++generated_tokens;
        int32_t buffer_length = llama_token_to_piece(this->vocab, token_id, buffer.data(), buffer.size(), 0, true);
        if (buffer_length < 0) {
            // 理论上不可能超过这个缓冲直接失败
            CW_LOG_W("解码失败: %d", buffer_length);
            co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_STOP, n_prompt_tokens, generated_tokens };
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
            co_yield caiwei::text::CompletionResult{ thinking, toolcall, toolcall_session };
        } else {
            std::string token(buffer.begin(), buffer.begin() + buffer_length);
            #if ENABLE_CAIWEI_PERF
            std::printf("%s", token.c_str());
            std::fflush(stdout);
            #endif
            if (toolcall) {
                toolcall_session.put_token(std::move(token));
                co_yield caiwei::text::CompletionResult{ thinking, toolcall, toolcall_session };
            } else {
                co_yield caiwei::text::CompletionResult{ thinking, toolcall, token };
            }
        }
        batch = llama_batch_get_one(&token_id, 1);
    }
    #if ENABLE_CAIWEI_PERF
    llama_perf_sampler_print(sampler.get());
    llama_perf_context_print(context.get());
    #endif
}

std::generator<caiwei::text::CompletionResult> caiwei::context::LlamaCPPContext::generate_mtmd(const caiwei::text::CompletionsRequest& request) const {
    if (llama_model_has_encoder(this->model)) {
        CW_LOG_W("不支持的模型: %s", this->path.c_str());
        co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_STOP, 0, 0 };
        co_return;
    }
    llama_context_ptr context{ this->get_context()        };
    llama_sampler_ptr sampler{ this->get_sampler(request) };
    if (context == nullptr || sampler == nullptr) {
        CW_LOG_W("创建上下文或采样器失败");
        co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_STOP, 0, 0 };
        co_return;
    }
    std::string prompt;
    mtmd::bitmaps bitmaps;
    if (!this->build_mtmd(prompt, bitmaps, request)) {
        CW_LOG_W("多模态输入内容失败");
        co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_STOP, 0, 0 };
        co_return;
    }
    CW_LOG_D("提示词: %s", prompt.c_str());
    mtmd_input_text text;
    text.text          = prompt.data();
    text.text_len      = prompt.size();
    text.add_special   = false;
    text.parse_special = true;
    auto bitmaps_ptr = bitmaps.c_ptr();
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    int32_t ret = mtmd_tokenize(this->mtmd_model, chunks.ptr.get(), &text, bitmaps_ptr.data(), bitmaps_ptr.size());
    if (ret != 0) {
        CW_LOG_W("分词失败: %d", ret);
        co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_STOP, 0, 0 };
        co_return;
    }
    const size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
    if (n_chunks == 0) {
        CW_LOG_W("分词失败: %" PRId64, n_chunks);
        co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_STOP, 0, 0 };
        co_return;
    }
    int32_t   n_batch    = this->max_token_length;
    llama_pos n_pos      = 0;
    llama_pos n_pos_next = 0;
    mtmd::batch_ptr mtmd_batch{ nullptr };
    for (size_t i = 0; i < n_chunks; i++) {
        auto* chunk      = mtmd_input_chunks_get(chunks.ptr.get(), i);
        auto  chunk_type = mtmd_input_chunk_get_type(chunk);
        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            ret = mtmd_helper_eval_chunk_single(this->mtmd_model, context.get(), chunk, n_pos, 0, n_batch, i == n_chunks - 1, &n_pos_next);
            if (ret != 0) {
                CW_LOG_W("文本嵌入失败: %" PRId64 " = %d", i, ret);
                co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_STOP, static_cast<uint32_t>(n_pos), 0 };
                co_return;
            }
            CW_LOG_D("文本嵌入: %" PRId64 " / %" PRId64, i + 1, n_chunks);
            n_pos = n_pos_next;
        } else {
            float* embd = nullptr;
            if (mtmd_batch) {
                embd = mtmd_batch_get_output_embd(mtmd_batch.get(), chunk);
            }
            if (embd == nullptr) {
                mtmd_batch.reset(mtmd_batch_init(this->mtmd_model));
                ret = mtmd_batch_add_chunk(mtmd_batch.get(), chunk);
                if (ret != 0) {
                    CW_LOG_W("添加媒体失败: %" PRId64 " = %d", i, ret);
                    co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_STOP, static_cast<uint32_t>(n_pos), 0 };
                    co_return;
                }
                int n_add_chunk = 1;
                for (size_t j = i + 1; j < n_chunks; j++) {
                    auto* next_chunk      = mtmd_input_chunks_get(chunks.ptr.get(), j);
                    auto  next_chunk_type = mtmd_input_chunk_get_type(next_chunk);
                    if (next_chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                        break;
                    }
                    ret = mtmd_batch_add_chunk(mtmd_batch.get(), next_chunk);
                    if (ret != 0) {
                        CW_LOG_W("添加媒体失败: %" PRId64 " = %d", j, ret);
                        co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_STOP, static_cast<uint32_t>(n_pos), 0 };
                        co_return;
                    }
                    n_add_chunk++;
                }
                ret = mtmd_batch_encode(mtmd_batch.get());
                if (ret != 0) {
                    CW_LOG_W("媒体编码失败: %" PRId64 " = %d", i, ret);
                    co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_STOP, static_cast<uint32_t>(n_pos), 0 };
                    co_return;
                }
                CW_LOG_D("媒体编码: %" PRId64 " / %" PRId64 " = %d", i + 1, n_chunks, n_add_chunk);
                embd = mtmd_batch_get_output_embd(mtmd_batch.get(), chunk);
            }
            ret = mtmd_helper_decode_image_chunk(this->mtmd_model, context.get(), chunk, embd, n_pos, 0, n_batch, &n_pos_next, nullptr, nullptr);
            if (ret != 0) {
                CW_LOG_W("媒体嵌入失败: %" PRId64 " = %d", i, ret);
                co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_STOP, static_cast<uint32_t>(n_pos), 0 };
                co_return;
            }
            CW_LOG_D("媒体嵌入: %" PRId64 " / %" PRId64, i + 1, n_chunks);
            n_pos = n_pos_next;
        }
    }
    const uint32_t n_ctx = llama_n_ctx(context.get());
    const uint32_t n_prompt_tokens = n_pos;
    if (n_prompt_tokens > n_ctx) {
        CW_LOG_W("提示词超长: %d > %d", n_prompt_tokens, n_ctx);
        co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_LENGTH, n_prompt_tokens, 0 };
        co_return;
    }
    std::string buffer;
    buffer.resize(1024);
    llama_token token_id;
    bool thinking = false;
    bool toolcall = false;
    const llama_token b_thinking = piece_to_token(this->vocab, this->special_token.b_thinking);
    const llama_token e_thinking = piece_to_token(this->vocab, this->special_token.e_thinking);
    const llama_token b_toolcall = piece_to_token(this->vocab, this->special_token.b_toolcall);
    const llama_token e_toolcall = piece_to_token(this->vocab, this->special_token.e_toolcall);
    caiwei::text::ToolcallSession toolcall_session;
    llama_batch batch = llama_batch_init(1, 0, 1);
    uint32_t generated_tokens = 0;
    uint32_t max_completion_tokens = request.max_completion_tokens.value_or(this->max_token_length);
    while (true) {
        llama_pos n_ctx_pos = llama_memory_seq_pos_max(llama_get_memory(context.get()), 0);
        if (n_ctx_pos < 0) {
            n_ctx_pos = 0;
        } else {
            n_ctx_pos += 1;
        }
        if (n_ctx_pos + batch.n_tokens > n_ctx) {
            CW_LOG_W("上下文超过最大长度: %d", n_ctx);
            co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_LENGTH, n_prompt_tokens, generated_tokens };
            break;
        }
        if (generated_tokens > max_completion_tokens) {
            CW_LOG_W("生成内容超过最大长度: %d", max_completion_tokens);
            co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_LENGTH, n_prompt_tokens, generated_tokens };
            break;
        }
        token_id = llama_sampler_sample(sampler.get(), context.get(), -1);
        if (token_id == LLAMA_TOKEN_NULL) {
            CW_LOG_W("采样失败: %d", token_id);
            co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_STOP, n_prompt_tokens, generated_tokens };
            break;
        }
        if (llama_vocab_is_eog(this->vocab, token_id)) {
            #if ENABLE_CAIWEI_PERF
            std::printf("\n");
            std::fflush(stdout);
            #endif
            CW_LOG_D("采样结束: %d", token_id);
            if (toolcall) {
                co_yield caiwei::text::CompletionResult{ thinking, toolcall, caiwei::text::FINISH_REASON_TOOL_CALLS, n_prompt_tokens, generated_tokens };
            } else {
                co_yield caiwei::text::CompletionResult{ thinking, toolcall, caiwei::text::FINISH_REASON_STOP, n_prompt_tokens, generated_tokens };
            }
            break;
        }
        ++generated_tokens;
        int32_t buffer_length = llama_token_to_piece(this->vocab, token_id, buffer.data(), buffer.size(), 0, true);
        if (buffer_length < 0) {
            // 理论上不可能超过这个缓冲直接失败
            CW_LOG_W("解码失败: %d", buffer_length);
            co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_STOP, n_prompt_tokens, generated_tokens };
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
            co_yield caiwei::text::CompletionResult{ thinking, toolcall, toolcall_session };
        } else {
            std::string token(buffer.begin(), buffer.begin() + buffer_length);
            #if ENABLE_CAIWEI_PERF
            std::printf("%s", token.c_str());
            std::fflush(stdout);
            #endif
            if (toolcall) {
                toolcall_session.put_token(std::move(token));
                co_yield caiwei::text::CompletionResult{ thinking, toolcall, toolcall_session };
            } else {
                co_yield caiwei::text::CompletionResult{ thinking, toolcall, token };
            }
        }
        batch.n_tokens = 0;
        caiwei::context::batch_add(batch, token_id, n_pos++, { 0 }, true);
        ret = llama_decode(context.get(), batch);
        if (ret != 0) {
            CW_LOG_W("解码失败: %d", ret);
            co_yield caiwei::text::CompletionResult{ false, false, caiwei::text::FINISH_REASON_STOP, n_prompt_tokens, generated_tokens };
            break;
        }
    }
    llama_batch_free(batch);
    #if ENABLE_CAIWEI_PERF
    llama_perf_sampler_print(sampler.get());
    llama_perf_context_print(context.get());
    #endif
}

bool caiwei::context::LlamaCPPContext::build_mtmd(std::string& prompt, mtmd::bitmaps& bitmaps, const caiwei::text::CompletionsRequest& request) const {
    return false;
}

std::vector<llama_token> caiwei::context::tokenize(const llama_vocab* vocab, const std::string& text, bool add_special, bool parse_special) {
    const int n_tokens = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_special, parse_special);
    std::vector<llama_token> tokens(n_tokens);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_special, parse_special) < 0) {
        CW_LOG_W("编码失败: %s", text.c_str());
        return {};
    }
    return tokens;
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

void caiwei::context::batch_decode_pooling(llama_context* ctx, llama_batch& batch, float* output, int n_seq, int n_embd_out) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
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
