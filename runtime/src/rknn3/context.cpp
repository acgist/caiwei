#include "caiwei/runtime/rknn3.hpp"

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>

#include "llama-cpp.h"

caiwei::context::RKNN3Context::RKNN3Context(
    std::string model_path, std::string weight_path, std::string embedding_path, std::string tokenizer_path,
    int32_t max_token_length, caiwei::text::SpecialToken special_token
) : model_path(std::move(model_path)), weight_path(std::move(weight_path)),
    tokenizer_path(std::move(tokenizer_path)), embedding_path(std::move(embedding_path)),
    max_token_length(max_token_length), special_token(std::move(special_token)) {
}

caiwei::context::RKNN3Context::~RKNN3Context() {
    if (this->context != 0) {
        rknn3_destroy(this->context);
        this->context = 0;
    }
    if (this->tokenizer) {
        delete this->tokenizer;
        this->tokenizer = nullptr;
    }
    if (this->embedding_fd != -1) {
        if (this->embedding_data != MAP_FAILED && this->embedding_data != NULL) {
            munmap((void*) this->embedding_data, this->emb_st.st_size);
            this->embedding_data = NULL;
        }
        close(this->embedding_fd);
        this->embedding_fd = -1;
    }
}

bool caiwei::context::RKNN3Context::load_model() {
    rknn3_config config;
    config.run_core_mask = 0xFF;
    // config.run_core_mask = RKNN3_NPU_CORE_ALL;
    int ret = rknn3_init(&this->context, NULL);
    if (ret < 0) {
        CW_LOG_W("加载RKNN3上下文失败: %d", ret);
        return false;
    }
    ret = rknn3_load_model_from_path(this->context, this->model_path.c_str(), this->weight_path.c_str());
    if (ret < 0) {
        CW_LOG_W("加载RKNN3模型失败: %d = %s + %s", ret, this->model_path.c_str(), this->weight_path.c_str());
        return false;
    }
    ret = rknn3_model_init(this->context, &config);
    if (ret < 0) {
        CW_LOG_W("初始化RKNN3模型失败: %d = %s + %s", ret, this->model_path.c_str(), this->weight_path.c_str());
        return false;
    }
    this->tokenizer = new Tokenizer(this->tokenizer_path);
    if (!this->tokenizer) {
        CW_LOG_W("加载分词器失败: %s", this->tokenizer_path.c_str());
        return false;
    }
    this->embedding_fd = open(this->embedding_path.c_str(), O_RDONLY);
    if (this->embedding_fd == -1) {
        CW_LOG_W("打开嵌入文件失败: %s", this->embedding_path.c_str());
        return false;
    }
    if (fstat(this->embedding_fd, &emb_st) == -1) {
        CW_LOG_W("获取嵌入文件信息失败: %s", this->embedding_path.c_str());
        return false;
    }
    this->embedding_data = (float16*) mmap(NULL, emb_st.st_size, PROT_READ, MAP_PRIVATE, this->embedding_fd, 0);
    if (this->embedding_data == MAP_FAILED) {
        CW_LOG_W("映射嵌入文件失败: %s", this->embedding_path.c_str());
        return false;
    }
    this->vocab_size = this->tokenizer->get_size();
    this->embedding_dim = (emb_st.st_size / this->vocab_size) / sizeof(float16);
    return true;
}

rknn3_session* caiwei::context::RKNN3Context::get_session(const caiwei::text::CompletionsRequest& request, ContextSession* context_session) {
    const rknn3_sampling_params sampling_params = {
        .top_k             = 1,
        .top_p             = request.top_p.value_or(0.9F),
        .temperature       = request.temperature.value_or(1.0F),
        .repeat_penalty    = 1.2f,
        .frequency_penalty = request.frequency_penalty.value_or(0.0F),
        .presence_penalty  = request.presence_penalty.value_or(0.0F)
    };
    int n_params = 1;
    rknn3_llm_param params{};
    params.logits_name           = "caiwei-logits";
    params.max_context_len       = request.max_tokens.value_or(this->max_token_length);
    params.sampling_param        = sampling_params;
    params.vocab_info.vocab_size = this->tokenizer->get_size();
    params.vocab_info.n_special_eos_id = 1;
    params.vocab_info.n_special_bos_id = 1;
    params.vocab_info.special_bos_id[0] = this->tokenizer->get_bos();
    params.vocab_info.special_eos_id[0] = this->tokenizer->get_eos();
    RKLLMCallback callback{};
    callback.embed_callback     = embed_callback;
    callback.embed_userdata     = context_session;
    callback.result_callback    = result_callback;
    callback.result_userdata    = context_session;
    callback.tokenizer_callback = tokenizer_callback;
    callback.tokenizer_userdata = context_session;
    rknn3_session* session = rknn3_session_init(this->context, &params, n_params);
    if (!session) {
        CW_LOG_W("初始化RKNN3会话失败");
        return nullptr;
    }
    int ret = rknn3_session_set_callback(session, &callback);
    if (ret < 0) {
        CW_LOG_W("设置RKNN3会话回调失败: %d", ret);
        rknn3_session_destroy(session);
        return nullptr;
    }
    return session;
}

std::generator<caiwei::text::Result> caiwei::context::RKNN3Context::generate(const caiwei::text::CompletionsRequest& request) {
    ContextSession context_session;
    context_session.tokenizer = this->tokenizer;
    context_session.embedding_dim = this->embedding_dim;
    context_session.embedding_data = this->embedding_data;
    context_session.b_thinking = this->tokenizer->piece_to_token(this->special_token.b_thinking);
    context_session.e_thinking = this->tokenizer->piece_to_token(this->special_token.e_thinking);
    context_session.b_toolcall = this->tokenizer->piece_to_token(this->special_token.b_toolcall);
    context_session.e_toolcall = this->tokenizer->piece_to_token(this->special_token.e_toolcall);
    // TODO 多模态输入数据多态实现
    rknn3_llm_tensor tensor{};
    tensor.name     = "input_embeds";
    // TODO
    tensor.prompt   = "prompt";
    tensor.embed    = NULL;
    tensor.tokens   = NULL;
    tensor.n_tokens = 0;
    tensor.enable_thinking = false;
    int n_inputs = 1;
    rknn3_llm_input inputs[1];
    rknn3_llm_infer_param llm_infer_param;
    llm_infer_param.keep_history = 0;
    llm_infer_param.max_new_tokens = request.max_tokens.value_or(this->max_token_length);
    inputs[0].role       = "user";
    inputs[0].input_type = RKNN3_LLM_INPUT_PROMPT;
    inputs[0].llm_input  = tensor;
    context_session.llm_begin_time = std::chrono::system_clock::now();
    rknn3_session* session = this->get_session(request, &context_session);
    if (!session) {
        co_return;
    }
    // int ret = rknn3_session_run(session, inputs, n_inputs, &llm_infer_param);
    int ret = rknn3_session_run_async(session, inputs, n_inputs, &llm_infer_param);
    {
        std::unique_lock<std::mutex> lock(context_session.mutex);
        while (!context_session.end) {
            context_session.cv.wait_for(lock, std::chrono::seconds(8));
            if (context_session.token.empty()) {
                continue;
            }
            for (auto& ret : context_session.token) {
                co_yield std::move(ret);
            }
            context_session.token.clear();
        }
    }
    context_session.llm_end_time = std::chrono::system_clock::now();
    if (ret < 0) {
        CW_LOG_W("RKNN3会话运行失败: %d", ret);
    } else {
        #ifdef CAIWEI_DEBUG
        RKLLMRunState state{};
        ret = rknn3_session_query_state(session, &state);
        if (ret < 0) {
            CW_LOG_W("RKNN3会话查询状态失败: %d", ret);
        } else {
            context_session.n_decode_tokens  = state.n_decode_tokens;
            context_session.n_prefill_tokens = state.n_prefill_tokens;
            printf_session_perf(&context_session);
        }
        #endif
    }
    rknn3_session_destroy(session);
}

int caiwei::context::embed_callback(void* userdata, int32_t* tokens, uint64_t n_tokens, void* embed, uint64_t len) {
    caiwei::context::ContextSession* session = (caiwei::context::ContextSession*) userdata;
    if (len != n_tokens * session->embedding_dim * sizeof(float16)) {
        CW_LOG_W("嵌入数据长度错误");
        return -1;
    }
    unsigned char* embed_ptr = (unsigned char*) embed;
    for (int n = 0; n < n_tokens; ++n) {
        std::memcpy(
            embed_ptr + n * session->embedding_dim * sizeof(float16),
            session->embedding_data + tokens[n] * session->embedding_dim,
            session->embedding_dim * sizeof(float16)
        );
    }
    session->n_prefill_tokens = n_tokens;
    return 0;
}

int caiwei::context::result_callback(void* userdata, RKLLMResult* result, LLMCallState state) {
    caiwei::context::ContextSession* session   = (caiwei::context::ContextSession*) userdata;
    caiwei::context::Tokenizer     * tokenizer = session->tokenizer;
    if (state == RKLLM_RUN_ERROR) {
        CW_LOG_W("RKNN3会话运行错误");
        std::lock_guard<std::mutex> lock(session->mutex);
        session->token.push_back(caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_LENGTH, static_cast<uint32_t>(session->n_prefill_tokens), session->n_decode_tokens });
        session->end = true;
        session->cv.notify_one();
    } else if (state == RKLLM_RUN_WAITING) {
        CW_LOG_W("RKNN3会话运行告警");
    } else if (state == RKLLM_RUN_FINISH) {
        CW_LOG_W("RKNN3会话运行完成");
        std::lock_guard<std::mutex> lock(session->mutex);
        if (session->toolcall) {
            session->token.push_back(caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_TOOL_CALLS, static_cast<uint32_t>(session->n_prefill_tokens), session->n_decode_tokens });
        } else {
            session->token.push_back(caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_STOP, static_cast<uint32_t>(session->n_prefill_tokens), session->n_decode_tokens });
        }
        session->end = true;
        session->cv.notify_one();
    } else if (state == RKLLM_RUN_MAX_NEW_TOKEN_REACHED) {
        CW_LOG_W("RKNN3会话超过最大次元数量");
        std::lock_guard<std::mutex> lock(session->mutex);
        session->token.push_back(caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_MAX_TOKENS, static_cast<uint32_t>(session->n_prefill_tokens), session->n_decode_tokens });
        session->end = true;
        session->cv.notify_one();
    } else if (state == RKLLM_RUN_STOP) {
        CW_LOG_W("RKNN3会话运行停止");
        std::lock_guard<std::mutex> lock(session->mutex);
        if (session->toolcall) {
            session->token.push_back(caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_TOOL_CALLS, static_cast<uint32_t>(session->n_prefill_tokens), session->n_decode_tokens });
        } else {
            session->token.push_back(caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_STOP, static_cast<uint32_t>(session->n_prefill_tokens), session->n_decode_tokens });
        }
        session->end = true;
        session->cv.notify_one();
    } else if (state == RKLLM_RUN_NORMAL) {
        std::lock_guard<std::mutex> lock(session->mutex);
        if (session->first) {
            session->llm_first_time = std::chrono::system_clock::now();
            session->first = false;
        }
        session->n_decode_tokens += result->num_tokens;
        for (int i = 0; i < result->num_tokens; ++i) {
            int32_t token_id = result->token_ids[i];
            if (token_id == session->b_thinking) {
                session->thinking = true;
            } else if (token_id == session->e_thinking) {
                session->thinking = false;
            } else if (token_id == session->b_toolcall) {
                session->toolcall = true;
                session->result_toolcall.increment();
            } else if (token_id == session->e_toolcall) {
                // TOOLCALL不要修改状态
                session->result_toolcall.finish();
                session->token.push_back(caiwei::text::Result{ session->thinking, session->toolcall, &session->result_toolcall });
            } else {
                std::string piece = tokenizer->token_to_piece(token_id);
                #if CAIWEI_DEBUG
                std::printf("%s", piece.c_str());
                std::fflush(stdout);
                #endif
                if (session->toolcall) {
                    session->result_toolcall.put_token(std::move(piece));
                    session->token.push_back(caiwei::text::Result{ session->thinking, session->toolcall, &session->result_toolcall });
                } else {
                    session->token.push_back(caiwei::text::Result{ session->thinking, session->toolcall, piece });
                }
            }
        }
    }
    return 0;
}

int caiwei::context::tokenizer_callback(void* userdata, const char* text, int32_t text_len, int32_t* tokens, int32_t n_tokens_max) {
    caiwei::context::ContextSession* session   = (caiwei::context::ContextSession*) userdata;
    caiwei::context::Tokenizer     * tokenizer = session->tokenizer;
    int n_tokens = tokenizer->tokenize(text, text_len, tokens, n_tokens_max);
    if (n_tokens <= 0) {
        CW_LOG_W("分词失败: %s", text);
        return n_tokens;
    }
    return n_tokens;
}

void caiwei::context::printf_session_perf(caiwei::context::ContextSession* session) {
    std::printf("\n--------------------------------------------------------------------------------------\n");
    std::printf(" %-12s  %-15s  %-8s  %-23s  %-23s\n",  "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
    std::printf("--------------------------------------------------------------------------------------\n");
    size_t prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(session->llm_first_time - session->llm_begin_time).count();
    int prefill_n_tokens = session->n_prefill_tokens;
    float prefill_tpt = prefill_n_tokens == 0 ? 0.0F : prefill_ms / prefill_n_tokens;  
    float prefill_tps = prefill_n_tokens == 0 ? 0.0F : 1e3f / prefill_ms * prefill_n_tokens; 
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n", "Prefill", prefill_ms, prefill_n_tokens, prefill_tpt, prefill_tps);
    size_t decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(session->llm_end_time - session->llm_first_time).count();
    int decode_n_tokens = session->n_decode_tokens;
    float decode_tpt = decode_n_tokens == 0 ? 0.0f : decode_ms / decode_n_tokens;
    float decode_tps = decode_n_tokens == 0 ? 0.0f : 1e3f / decode_ms * decode_n_tokens;
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n", "Generate", decode_ms, decode_n_tokens, decode_tpt, decode_tps);
    printf("--------------------------------------------------------------------------------------\n");    
}
