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
    for (int i = 0; i < n_output_tensors; i++)
    {
        if (output_tensors[i].attr)
        {
            free(output_tensors[i].attr);
            output_tensors[i].attr = NULL;
        }
        if (output_tensors[i].mem)
        {
            rknn3_destroy_mem(rknn_app_ctx.rknn_ctx, output_tensors[i].mem);
            output_tensors[i].mem = NULL;
        }
    }
    if (this->speedup) {
        speedup_destroy(this->speedup);
        this->speedup = nullptr;
    }
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
    // TODO 理解测试
    config.user_mem_internal = 1; // 使用用户管理的internal内存
    rknn3_devices devs;
    // Query available devices
    memset(&devs, 0, sizeof(devs));
    ret = rknn3_find_devices(&devs);
    if (ret != RKNN3_SUCCESS) {
        printf("rknn3_find_devices failed! ret=%d\n", ret);
        return -1;
    }
    printf("Found %d RK182X devices\n", devs.n_devices);
    for (int i = 0; i < devs.n_devices; i++) {
        printf("  Device %d: transfer_type=%s, id=%s\n", i, devs.devices[i].type, devs.devices[i].id);
    }

    // Select the first device
    if (devs.n_devices == 0) {
        printf("No RK182X devices found\n");
        return -1;
    } else if (devs.n_devices == 1) {
        // If only one device found, the init_extend can be NULL
        printf("Info: Only one device found (id=%s), init_extend can be NULL\n", devs.devices[0].id);
    } else {
        printf("Multiple devices found, using the first one (id=%s)\n", devs.devices[0].id);
    }
    rknn3_init_extend init_extend{};
    init_extend.device_id = devs.devices[0].id;
    int ret = rknn3_init(&this->context, &init_extend);
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
        // // Set Chat Template
        // ret = rknn3_session_set_chat_template(session, system_prompt, prompt_prefix, prompt_postfix);
        // if (ret < 0)
        // {
        //     printf("Failed to set chat template\n");
        //     return -1;
        // }
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

bool caiwei::context::RKNN3Context::init_output() {
    this->model_output.resize(n_output_tensors);
    for (int i = 0; i < n_output_tensors; i++)
    {
        output_tensors[i].attr = (rknn3_tensor_attr *)malloc(sizeof(rknn3_tensor_attr));
        // Query output tensor info according to the output tensor index
        output_tensors[i].attr->index = output_tensors_index[i];
        ret = rknn3_query(rknn_app_ctx.rknn_ctx, RKNN3_QUERY_OUTPUT_ATTR, output_tensors[i].attr, sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn3_query fail! ret=%d\n", ret);
            goto out;
        }

        output_tensors[i].mem =
            rknn3_create_mem(rknn_app_ctx.rknn_ctx, output_tensors[i].attr->aligned_size, output_tensors[i].attr->core_id, RKNN3_FLAG_MEMORY_CACHEABLE);

            this->model_output[i].resize(output_tensors[i].attr->n_elems);
        // model_output = (float *)malloc(output_tensors[i].attr->n_elems * sizeof(float));
        // if (!model_output)
        // {
        //     printf("Failed to allocate memory for model output!\n");
        // }
    }
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
    params.vocab_info.linefeed_id = this->tokenizer->get_nl();
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
    if (this->n_output_tensors != 0) {
        callback.output_callback = output_callback;
        callback.output_userdata = &model_output;
        callback.output_tensors = output_tensors.data();
        callback.n_output_tensors = n_output_tensors;
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
    tensor.prompt   = "prompt";
    tensor.embed    = NULL;
    tensor.tokens   = NULL;
    tensor.n_tokens = 0;
    tensor.enable_thinking = false;
    std::vector<rknn3_llm_input> inputs(1);
    inputs[0].input_type = RKNN3_LLM_INPUT_PROMPT;
    inputs[0].llm_input  = tensor;
    return inputs;
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
    rknn3_session* session = this->get_session(request, &context_session);
    if (!session) {
        co_return;
    }
    // TODO 多模态输入数据多态实现
    std::vector<rknn3_llm_input> inputs = this->get_inputs(session, request);
    rknn3_llm_infer_param llm_infer_param;
    llm_infer_param.keep_history = 0;
    llm_infer_param.max_new_tokens = request.max_tokens.value_or(this->max_token_length);
    context_session.llm_begin_time = std::chrono::system_clock::now();
    // int ret = rknn3_session_run(session, inputs.data(), inputs.size(), &llm_infer_param);
    int ret = rknn3_session_run_async(session, inputs.data(), inputs.size(), &llm_infer_param);
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

int output_callback(void *userdata, rknn3_tensor *output_tensors, uint32_t n_output_tensors, LLMOutputCallbackState state)
{

    printf("\noutput_callback: state = %d\n", state);
    if (state != RKLLM_OUTPUT_CALLBACK_PREFILL_FINISHED)
    {
        return 0;
    }
    else if (state == RKLLM_OUTPUT_CALLBACK_PREFILL_FINISHED)
    {
        if (first_decode)
        {
            first_token = getCurrentTimeUs();
            first_decode = false;
        }

        std::vector<std::vector<float16>>& model_output = *((std::vector<std::vector<float16>>*)userdata);

        for (int i = 0; i < n_output_tensors; i++)
        {
            printf("output_callback: output[%d]->attr->index = %d\n", i, output_tensors[i].attr->index);
            printf("output_callback: output[%d]->attr->name = %s\n", i, output_tensors[i].attr->name);
            printf("output_callback: output[%d]->mem->size = %lu\n", i, output_tensors[i].mem->size);
            for (int j = 0; j < output_tensors[i].attr->n_elems; j++)
            {
                model_output[i][j] = fp16_to_fp32(((float16 *)output_tensors[i].mem->virt_addr)[j]);

                if (j < 10)
                {
                    printf("output_callback: output[%d][%d] = %f\n", i, j, model_output[j]);
                }
            }
        }
    }

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
    printf("--------------------------------------------------------------------------------------\n");
    
    // printf(" Vision latency = %.2f ms, FPS = %.2f\n", 
    //        (int)session->vision_latency / 1000.f, 1000.f * 1000.f / (int)session->vision_latency);

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
}

caiwei::context::RKNN3CVContext::RKNN3CVContext(std::string path, int c, int h, int w) : path(std::move(path)), input_data_length(c * h * w) {
}

caiwei::context::RKNN3CVContext::~RKNN3CVContext() {

    if (this->inputs != NULL) {
        for (int i = 0; i < this->io_num.n_input; i++) {
            if (this->inputs[i].mem) {
                rknn3_destroy_mem(this->rknn_ctx, this->inputs[i].mem);
            }
            if (this->inputs[i].attr != NULL) {
                free(this->inputs[i].attr);
                this->inputs[i].attr = NULL;
            }
        }
        this->inputs.clear();
    }

    if (this->outputs != NULL) {
        for (int i = 0; i < this->io_num.n_output; i++) {
            if (this->outputs[i].mem) {
                rknn3_destroy_mem(this->rknn_ctx, this->outputs[i].mem);
            }
            if (this->outputs[i].attr != NULL) {
                free(this->outputs[i].attr);
                this->outputs[i].attr = NULL;
            }
        }
        this->outputs.clear();
    }
    if (this->context != 0) {
        rknn3_destroy(this->context);
        this->context = 0;
    }
}

static void dump_tensor_attr(rknn3_tensor_attr* attrs)
{
    std::string shape_str = "";
    for (int j = 0; j < attrs->n_dims; j++) {
      shape_str += std::to_string(attrs->shape[j]);
      if (j < attrs->n_dims - 1) {
        shape_str += ", ";
      }
    }

    std::string stride_str = "";
    for (int j = 0; j < attrs->n_stride; j++) {
      stride_str += std::to_string(attrs->stride[j]);
      if (j < attrs->n_stride - 1) {
        stride_str += ", ";
      }
    }

    printf("Tensor: name=%s, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%ld, layout=%s, dtype=%s, core_id=%d, "
           "qnt_type=%s\n",
           attrs->name, attrs->n_dims, shape_str.c_str(), stride_str.c_str(), attrs->aligned_size, rknn3_get_layout_string(attrs->layout),
           rknn3_get_type_string(attrs->dtype), attrs->core_id, rknn3_get_qnt_type_string(attrs->qnt_type));
}

bool caiwei::context::RKNN3CVContext::load_model() {
    rknn3_config config;
    config.run_core_mask = 0xFF;
    int ret = rknn3_init(&this->context, nullptr);
    if (ret < 0) {
        CW_LOG_W("加载模型失败: %s", this->path.c_str());
        return false;
    }
    ret = rknn3_load_model_from_path(this->context, this->path.c_str(), nullptr);
    if (ret < 0) {
        CW_LOG_W("加载模型失败: %s", this->path.c_str());
        return false;
    }
    ret = rknn3_model_init(this->context, &config);
    if (ret < 0) {
        CW_LOG_W("初始化模型失败: %s", this->path.c_str());
        return false;
    }
    rknn3_input_output_num io_num;
    ret = rknn3_query(this->context, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        CW_LOG_W("查询参数数量失败: %s", this->path.c_str());
        return false;
    }
    this->input_size = io_num.n_input;
    this->output_size = io_num.n_output;
    this->input_attrs.resize(io_num.n_input);
    for (int i = 0; i < io_num.n_input; i++)
    {
        this->input_attrs[i].index = i;
        ret = rknn3_query(this->context, RKNN3_QUERY_INPUT_ATTR, this->input_attrs + i, sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return false;
        }
        dump_tensor_attr(this->input_attrs + i);
    }

    std_attrs.resize(io_num.n_output);
    for (int i = 0; i < io_num.n_output; i++)
    {
        std_attrs[i].index = i;
        ret = rknn3_query(this->context, RKNN3_QUERY_OUTPUT_ATTR, std->output_attrs + i, sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            return false;
        }
        dump_tensor_attr(this->output_attrs + i);
    }
    this->inputs.resize(io_num.n_input);
    this->outputs.resize(io_num.n_output);
    for (int i = 0; i < io_num.n_input; i++) {
        this->inputs[i].mem  = rknn3_create_mem(ctx, input_attrs[i].aligned_size, input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        this->inputs[i].attr = new rknn3_tensor_attr;
        this->inputs[i].attr->index = i;
        if (this->inputs[i].mem == nullptr || this->inputs[i].attr == nullptr)
        {
            printf("create input tensor memory failed, index=%d\n", i);
            return false;
        }
        memcpy(this->inputs[i].attr, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
        dump_tensor_attr(this->inputs[i].attr);
    }
    for (int i = 0; i < io_num.n_output; i++) {
        this->outputs[i].mem  = rknn3_create_mem(this->, output_attrs[i].aligned_size, output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        this->outputs[i].attr = new rknn3_tensor_attr;
        this->outputs[i].attr->index = i;
        if (this->outputs[i].mem == nullptr || this->outputs[i].attr == nullptr)
        {
            printf("create output tensor memory failed, index=%d\n", i);
            return false;
        }
        memcpy(this->outputs[i].attr, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
        dump_tensor_attr(this->outputs[i].attr);
    }
}
