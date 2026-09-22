#include "caiwei/runtime/rknn3.hpp"

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>

#include "llama-cpp.h"

static void printf_tensor_info(rknn3_context context);

caiwei::context::RKNN3Context::RKNN3Context(
    std::string model_path, std::string weight_path, std::string embedding_path, std::string tokenizer_path,
    int32_t max_token_length, caiwei::text::SpecialToken special_token
) : model_path(std::move(model_path)), weight_path(std::move(weight_path)),
    tokenizer_path(std::move(tokenizer_path)), embedding_path(std::move(embedding_path)),
    max_token_length(max_token_length), special_token(std::move(special_token)) {
}

caiwei::context::RKNN3Context::~RKNN3Context() {
    // if (this->speedup) {
    //     speedup_destroy(this->speedup);
    //     this->speedup = nullptr;
    // }
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
    int ret = rknn3_init(&this->context, nullptr);
    if (ret < 0) {
        CW_LOG_W("加载RKNN3上下文失败: %d", ret);
        return false;
    }
    rknn3_sdk_version version;
    ret = rknn3_query(this->context, RKNN3_QUERY_SDK_VERSION, &version, sizeof(rknn3_sdk_version));
    if (ret < 0) {
        CW_LOG_W("查询RKNN3模型版本失败: %d = %s", ret, this->model_path.c_str());
        return false;
    }
    CW_LOG_I("加载RKNN3模型版本: %s = %s - %s", this->model_path.c_str(), version.api_version, version.drv_version);
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
    rknn3_llm_config llm_config{};
    ret = rknn3_query(this->context, RKNN3_QUERY_LLM_CONFIG, &llm_config, sizeof(rknn3_llm_config));
    if (ret != RKNN3_SUCCESS) {
        printf("rknn3_query llm config failed! ret=%d", ret);
        return false;
    }
    this->chat_template.set_template(llm_config.chat_template, this->special_token.bos, this->special_token.eos);
    this->max_token_length = std::min(this->max_token_length, (int32_t) llm_config.max_ctx_len);
    printf_tensor_info(this->context);
    // TODO 移到 load embedding
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

bool caiwei::context::ContextSession::init_output_tensors(int n_output_tensors) {
    this->model_output.resize(n_output_tensors);
    this->output_tensors.resize(n_output_tensors);
    for (int i = 0; i < n_output_tensors; ++i) {
        this->output_tensors[i].attr = new rknn3_tensor_attr;
        this->output_tensors[i].attr->index = i;
        int ret = rknn3_query(context, RKNN3_QUERY_OUTPUT_ATTR, this->output_tensors[i].attr, sizeof(rknn3_tensor_attr));
        if (ret < 0) {
            printf("rknn3_query fail! ret=%d\n", ret);
            return false;
        }
        this->output_tensors[i].mem = rknn3_create_mem(context, this->output_tensors[i].attr->aligned_size, this->output_tensors[i].attr->core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        this->model_output[i].resize(this->output_tensors[i].attr->n_elems);
        CW_LOG_I("输出结果: %s = %d", this->output_tensors[i].attr->name, this->output_tensors[i].attr->n_elems);
    }
    return true;
}

caiwei::context::ContextSession::~ContextSession() {
    for (int i = 0; i < this->output_tensors.size(); ++i) {
        if (this->output_tensors[i].attr) {
            delete this->output_tensors[i].attr;
            this->output_tensors[i].attr = nullptr;
        }
        if (this->output_tensors[i].mem) {
            rknn3_destroy_mem(context, this->output_tensors[i].mem);
            this->output_tensors[i].mem = nullptr;
        }
    }
}

rknn3_session* caiwei::context::RKNN3Context::get_session(rknn3_sampling_params sampling_params, ContextSession* context_session) {
    int n_params = 1;
    rknn3_llm_param params{};
    params.logits_name            = "output";
    params.max_context_len        = this->max_token_length;
    params.sampling_param         = sampling_params;
    params.vocab_info.vocab_size  = this->tokenizer->get_size();
    params.vocab_info.linefeed_id = this->tokenizer->get_nl();
    params.vocab_info.n_special_bos_id = 1;
    params.vocab_info.n_special_eos_id = 2;
    params.vocab_info.special_bos_id[0] = this->tokenizer->get_bos();
    params.vocab_info.special_eos_id[0] = this->tokenizer->get_eos();
    params.vocab_info.special_eos_id[1] = this->tokenizer->get_eot();
    RKLLMCallback callback{};
    callback.embed_callback     = embed_callback;
    callback.embed_userdata     = context_session;
    callback.result_callback    = result_callback;
    callback.result_userdata    = context_session;
    callback.tokenizer_callback = tokenizer_callback;
    callback.tokenizer_userdata = context_session;
    if (!context_session->output_tensors.empty()) {
        callback.output_callback = output_callback;
        callback.output_userdata = context_session;
        callback.output_tensors = context_session->output_tensors.data();
        callback.n_output_tensors = context_session->output_tensors.size();
    }
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
    #ifdef ENABLE_SPEEDUP
    SpeedUPConfig g_speedup_config = {
        .tau = 0.90f,
        .min_ratio = 0.6f,
        .max_ratio = 0.75f,
        .stride = 16
    };
    this->speedup = speedup_create(&g_speedup_config);
    if (!this->speedup) {
        return false;
    }
    ret = speedup_attach_mrope_callback(this->speedup,
        this->context,
                                             session,
                                             &callback);
    if (ret != 0) {
        printf("[SpeedUP] attach persistent input_callback failed, ret=%d; fallback to normal inference without SpeedUP\n", ret);
        speedup_destroy(this->speedup);
        this->speedup = NULL;
    } else {
        printf("[SpeedUP] persistent input_callback attached, version=%s\n",
               speedup_get_version());
    }
    #endif
    return session;
}

std::vector<rknn3_llm_input> caiwei::context::RKNN3Context::get_inputs(rknn3_session* session, const caiwei::text::CompletionsRequest& request) {
    rknn3_llm_tensor tensor{};
    tensor.name     = "input_embeds";
    // TODO
    tensor.prompt   = "你好，解释一下碧螺萧萧";
    tensor.embed    = NULL;
    tensor.tokens   = NULL;
    tensor.n_tokens = 0;
    tensor.enable_thinking = false;
    std::vector<rknn3_llm_input> inputs(1);
    inputs[0].input_type = RKNN3_LLM_INPUT_PROMPT;
    inputs[0].llm_input  = tensor;
    return inputs;
}

std::generator<caiwei::text::Result> caiwei::context::RKNN3Context::generate(caiwei::text::CompletionsRequest& request) {
    ContextSession context_session;
    context_session.tokenizer = this->tokenizer;
    context_session.embedding_dim = this->embedding_dim;
    context_session.embedding_data = this->embedding_data;
    context_session.b_thinking = this->tokenizer->piece_to_token(this->special_token.b_thinking);
    context_session.e_thinking = this->tokenizer->piece_to_token(this->special_token.e_thinking);
    context_session.b_toolcall = this->tokenizer->piece_to_token(this->special_token.b_toolcall);
    context_session.e_toolcall = this->tokenizer->piece_to_token(this->special_token.e_toolcall);
    // TODO 自动释放
    const rknn3_sampling_params sampling_params = {
        .top_k             = request.top_k.value_or(1),
        .top_p             = request.top_p.value_or(0.9F),
        .temperature       = request.temperature.value_or(1.0F),
        .repeat_penalty    = request.repeat_penalty.value_or(1.2F),
        .frequency_penalty = request.frequency_penalty.value_or(0.0F),
        .presence_penalty  = request.presence_penalty.value_or(0.0F)
    };
    rknn3_session_ptr session{ this->get_session(sampling_params, &context_session) };
    if (!session) {
        co_return;
    }
    // TODO 多模态输入数据多态实现
    std::vector<rknn3_llm_input> inputs = this->get_inputs(session.get(), request);
    rknn3_llm_infer_param llm_infer_param;
    llm_infer_param.keep_history = 0;
    llm_infer_param.max_new_tokens = request.max_completion_tokens.value_or(this->max_token_length);
    context_session.llm_begin_time = std::chrono::system_clock::now();
    context_session.first = true;
    // int ret = rknn3_session_run(session.get(), inputs.data(), inputs.size(), &llm_infer_param);
    int ret = rknn3_session_run_async(session.get(), inputs.data(), inputs.size(), &llm_infer_param);
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
        CW_LOG_I("RKNN3会话完成: %d = %d = %d", ret, context_session.n_decode_tokens, context_session.n_prefill_tokens);
        #ifdef CAIWEI_DEBUG
        RKLLMRunState state{};
        ret = rknn3_session_query_state(session.get(), &state);
        if (ret < 0) {
            CW_LOG_W("RKNN3会话查询状态失败: %d", ret);
        } else {
            CW_LOG_I("RKNN3会话完成: %d = %d = %d", ret, state.n_decode_tokens, state.n_prefill_tokens);
            // context_session.n_decode_tokens  = state.n_decode_tokens;
            // context_session.n_prefill_tokens = state.n_prefill_tokens;
            printf_session_perf(&context_session);
        }
        #endif
    }
}

int caiwei::context::embed_callback(void* userdata, int32_t* tokens, uint64_t n_tokens, void* embed, uint64_t len) {
    std::printf("embed_callback: %d\n", len);
    std::fflush(stdout);
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
    return 0;
}

int caiwei::context::output_callback(void* userdata, rknn3_tensor* output_tensors, uint32_t n_output_tensors, LLMOutputCallbackState state) {
    std::printf("output_callback: %d\n", state);
    std::fflush(stdout);
    caiwei::context::ContextSession* session = (caiwei::context::ContextSession*) userdata;
    if (state == RKLLM_OUTPUT_CALLBACK_PREFILL_FINISHED) {
        if (session->first) {
            session->first = false;
            session->llm_first_time = std::chrono::system_clock::now();
        }
        std::vector<std::vector<float>>& model_output = session->model_output;
        for (int i = 0; i < n_output_tensors; i++) {
            auto& output = model_output[i];
            output.resize(output_tensors[i].attr->n_elems);
            for (int j = 0; j < output_tensors[i].attr->n_elems; j++) {
                output[j] = fp16_to_fp32(((float16 *) output_tensors[i].mem->virt_addr)[j]);
            }
        }
    }
    return 0;
}

int caiwei::context::result_callback(void* userdata, RKLLMResult* result, LLMCallState state) {
    std::printf("result_callback: %d\n", state);
    std::fflush(stdout);
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
        CW_LOG_I("RKNN3会话运行完成");
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
        session->token.push_back(caiwei::text::Result{ false, false, caiwei::text::FINISH_REASON_LENGTH, static_cast<uint32_t>(session->n_prefill_tokens), session->n_decode_tokens });
        session->end = true;
        session->cv.notify_one();
    } else if (state == RKLLM_RUN_STOP) {
        CW_LOG_I("RKNN3会话运行停止");
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
            session->first = false;
            session->llm_first_time = std::chrono::system_clock::now();
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
                session->result_toolcall.reset();
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
    session->n_prefill_tokens = n_tokens;
    return n_tokens;
}

void caiwei::context::printf_session_perf(caiwei::context::ContextSession* session) {
    std::printf("\n--------------------------------------------------------------------------------------\n");
    std::printf(" %-12s  %-15s  %-8s  %-23s  %-23s\n",  "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
    std::printf("--------------------------------------------------------------------------------------\n");
    float prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(session->llm_first_time - session->llm_begin_time).count();
    int prefill_n_tokens = session->n_prefill_tokens;
    float prefill_tpt = prefill_n_tokens == 0 ? 0.0F : prefill_ms / prefill_n_tokens;
    float prefill_tps = prefill_n_tokens == 0 ? 0.0F : 1e3f / prefill_ms * prefill_n_tokens;
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n", "Prefill", prefill_ms, prefill_n_tokens, prefill_tpt, prefill_tps);
    float decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(session->llm_end_time - session->llm_first_time).count();
    int decode_n_tokens = session->n_decode_tokens;
    float decode_tpt = decode_n_tokens == 0 ? 0.0f : decode_ms / decode_n_tokens;
    float decode_tps = decode_n_tokens == 0 ? 0.0f : 1e3f / decode_ms * decode_n_tokens;
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n", "Generate", decode_ms, decode_n_tokens, decode_tpt, decode_tps);
    printf("--------------------------------------------------------------------------------------\n");
    // printf(" Vision latency = %.2f ms, FPS = %.2f\n", 
    //        (int)session->vision_latency / 1000.f, 1000.f * 1000.f / (int)session->vision_latency);
    // printf("--------------------------------------------------------------------------------------\n");
    //        if (p->audio_latency > 0) {
        //         printf(" Audio latency = %.2f ms, FPS = %.2f\n", 
        //             (int)p->audio_latency / 1000.f, 1000.f * 1000.f / (int)p->audio_latency);
        //     }
        //     float total_inference_us = (float)(p->llm_end_time - inference_start);
        //     float rtf = (audio_duration_sec > 0) ? (total_inference_us / 1000.0f / 1000.0f / audio_duration_sec) : 0.0f;
    //     float ttft_include_encoder = prefill_ms + (int)p->audio_latency / 1000.0f;
    //     printf("\n");
    //     printf(" Audio Duration = %.2f s\n", audio_duration_sec);
    //     printf(" Total Inference = %.2f ms\n", total_inference_us / 1000.0f);
    //     printf(" RTF = %.4f (%.2fx)\n", rtf, rtf);
    //     printf(" TTFT (include encoder) = %.2f ms\n", ttft_include_encoder);
    // printf("--------------------------------------------------------------------------------------\n");
}

static void dump_tensor_attr(rknn3_tensor_attr *attrs) {
    std::string shape_str = "";
    for (int j = 0; j < attrs->n_dims; j++)
    {
        shape_str += std::to_string(attrs->shape[j]);
        if (j < attrs->n_dims - 1)
        {
            shape_str += ", ";
        }
    }

    std::string stride_str = "";
    for (int j = 0; j < attrs->n_stride; j++)
    {
        stride_str += std::to_string(attrs->stride[j]);
        if (j < attrs->n_stride - 1)
        {
            stride_str += ", ";
        }
    }

    printf("    name=%s, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%ld, layout=%s, dtype=%s, core_id=%d, "
           "qnt_type=%s\n",
           attrs->name, attrs->n_dims, shape_str.c_str(), stride_str.c_str(), attrs->aligned_size, rknn3_get_layout_string(attrs->layout),
           rknn3_get_type_string(attrs->dtype), attrs->core_id, rknn3_get_qnt_type_string(attrs->qnt_type));
}

static void printf_tensor_info(rknn3_context context) {
    rknn3_input_output_num io_num;
    int ret = rknn3_query(context, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN3_SUCCESS)
    {
        printf("rknn3_query io_num fail! ret=%d\n", ret);
        return;
    }

    // query input tensors info
    printf("input tensors:\n");
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        rknn3_tensor_attr attr;
        attr.index = i;
        ret = rknn3_query(context, RKNN3_QUERY_INPUT_ATTR, &attr, sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn3_query error! ret=%d\n", ret);
            continue;
        }
        dump_tensor_attr(&attr);
    }

    printf("output tensors:\n");
    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        rknn3_tensor_attr attr;
        attr.index = i;
        ret = rknn3_query(context, RKNN3_QUERY_OUTPUT_ATTR, &attr, sizeof(rknn3_tensor_attr));
        if (ret != RKNN3_SUCCESS)
        {
            printf("rknn3_query fail! ret=%d\n", ret);
            continue;
        }
        dump_tensor_attr(&attr);
    }
}
