#include <chrono>

#include "llama.h"

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>

#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/runtime/rknn3.hpp"

static int embed_callback    (void* userdata, int32_t* tokens, uint64_t num_tokens, void* embed, uint64_t len);
static int output_callback   (void* userdata, rknn3_tensor* output_tensors, uint32_t n_output_tensors, LLMOutputCallbackState state);
static int result_callback   (void* userdata, RKLLMResult* result, LLMCallState state);
static int tokenizer_callback(void* userdata, const char* text, int32_t text_len, int32_t* tokens, int32_t n_tokens_max);

static bool init_tensor(rknn3_context& context, rknn3_tensor& tensor, int i, rknn3_query_cmd query_cmd, const char* name, uint64_t size_override);
static void release_tensor(rknn3_context& context, rknn3_tensor& tensor, const char* name);

caiwei::context::RKNN3Context::RKNN3Context(
    std::string model_path,
    std::string weight_path,
    std::string embedding_path,
    std::string tokenizer_path,
    int32_t max_token_length,
    caiwei::text::SpecialToken special_token
) : model_path(std::move(model_path)),
    weight_path(std::move(weight_path)),
    embedding_path(std::move(embedding_path)),
    tokenizer_path(std::move(tokenizer_path)),
    max_token_length(max_token_length),
    special_token(std::move(special_token)) {
    CW_LOG_I("创建RKNN3Context: %s", this->model_path.c_str());
}

caiwei::context::RKNN3Context::RKNN3Context(
    std::string model_path,
    std::string weight_path,
    std::string embedding_path,
    std::string tokenizer_path,
    std::string media_model_path,
    std::string media_weight_path,
    int32_t max_token_length,
    caiwei::text::SpecialToken special_token
) : model_path(std::move(model_path)),
    weight_path(std::move(weight_path)),
    embedding_path(std::move(embedding_path)),
    tokenizer_path(std::move(tokenizer_path)),
    media_model_path(std::move(media_model_path)),
    media_weight_path(std::move(media_weight_path)),
    max_token_length(max_token_length),
    special_token(std::move(special_token)) {
    CW_LOG_I("创建RKNN3Context: %s", this->model_path.c_str());
}

caiwei::context::RKNN3Context::~RKNN3Context() {
    CW_LOG_I("释放RKNN3Context: %s", this->model_path.c_str());
    this->release();
}

void caiwei::context::RKNN3Context::release() {
    this->chat_template.reset();
    if (!this->media_input.empty()) {
        for (auto& tensor : this->media_input) {
            release_tensor(this->media_context, tensor, "媒体输入");
        }
        this->media_input.clear();
    }
    if (!this->media_output.empty()) {
        for (auto& tensor : this->media_output) {
            release_tensor(this->media_context, tensor, "媒体输出");
        }
        this->media_output.clear();
    }
    if (!this->internal_tensor_mems.empty()) {
        for (auto*& tensor_mem : this->internal_tensor_mems) {
            if (tensor_mem) {
                CW_LOG_I("释放共享内存: %p", tensor_mem);
                rknn3_destroy_mem(this->context, tensor_mem);
                tensor_mem = nullptr;
            }
        }
        this->internal_tensor_mems.clear();
    }
    if (this->context != 0) {
        CW_LOG_I("释放RKNN3Context context");
        rknn3_destroy(this->context);
        this->context = 0;
    }
    if (this->media_context != 0) {
        CW_LOG_I("释放RKNN3Context media_context");
        rknn3_destroy(this->media_context);
        this->media_context = 0;
    }
    if (this->tokenizer) {
        CW_LOG_I("释放RKNN3Context tokenizer");
        delete this->tokenizer;
        this->tokenizer = nullptr;
    }
    if (this->embedding_data != MAP_FAILED && this->embedding_data != nullptr) {
        CW_LOG_I("释放RKNN3Context embedding_data");
        ::munmap((void*) this->embedding_data, this->embedding_stat.st_size);
        this->embedding_data = nullptr;
    }
    if (this->embedding_fd != -1) {
        CW_LOG_I("释放RKNN3Context embedding_fd");
        ::close(this->embedding_fd);
        this->embedding_fd = -1;
    }
}

bool caiwei::context::RKNN3Context::load_model(
    rknn3_context& context,
    const std::string& model_path,
    const std::string& weight_path,
    bool user_mem_internal,
    uint32_t core_mask
) {
    rknn3_config config{};
    config.priority      = 0;
    config.run_timeout   = 120'000;
    config.run_core_mask = core_mask;
    if (user_mem_internal) {
        config.user_mem_internal = 1;
    }
    rknn3_devices devs{};
    int ret = rknn3_find_devices(&devs);
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("设备查找失败: %d", ret);
        return false;
    }
    if (devs.n_devices <= 0) {
        CW_LOG_W("没有可用设备");
        return false;
    }
    const int rknn_id = std::max(0, std::min(caiwei::env::get_int("CAIWEI_RKNN_ID"), (int) devs.n_devices - 1));
    rknn3_init_extend extend{};
    extend.device_id = devs.devices[rknn_id].id;
    ret = rknn3_init(&context, &extend);
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("设备加载失败: %d", ret);
        return false;
    }
    CW_LOG_I("加载设备成功: %d => %s", rknn_id, extend.device_id);
    ret = rknn3_load_model_from_path(context, model_path.c_str(), weight_path.c_str());
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("模型读取失败: %d = %s", ret, model_path.c_str());
        return false;
    }
    ret = rknn3_model_init(context, &config);
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("模型加载失败: %d = %s", ret, model_path.c_str());
        return false;
    }
    rknn3_sdk_version version{};
    ret = rknn3_query(context, RKNN3_QUERY_SDK_VERSION, &version, sizeof(rknn3_sdk_version));
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("模型版本查询失败: %d = %s", ret, model_path.c_str());
        return false;
    }
    CW_LOG_I("模型加载成功: %s = %s - %s", model_path.c_str(), version.api_version, version.drv_version);
    return true;
}

bool caiwei::context::RKNN3Context::load_model(bool user_mem_internal, uint32_t core_mask) {
    if (!this->load_model(this->context, this->model_path, this->weight_path, user_mem_internal, core_mask)) {
        return false;
    }
    if (!this->load_tokenizer()) {
        return false;
    }
    if (!this->load_embedding()) {
        return false;
    }
    rknn3_llm_config llm_config{};
    int ret = rknn3_query(this->context, RKNN3_QUERY_LLM_CONFIG, &llm_config, sizeof(rknn3_llm_config));
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("模型配置查询失败: %d = %s", ret, this->model_path.c_str());
        return false;
    }
    const int32_t max_ctx_len = llm_config.max_ctx_len;
    if (max_ctx_len <= 0 || this->max_token_length <= 0) {
        CW_LOG_W("模型输入长度无效: %d - %d", max_ctx_len, this->max_token_length);
        return false;
    }
    const int32_t min_ctx_len = std::min(this->max_token_length, max_ctx_len);
    CW_LOG_I("模型最大输入长度: min(%d, %d) = %d", this->max_token_length, max_ctx_len, min_ctx_len);
    this->max_token_length = min_ctx_len;
    return true;
}

bool caiwei::context::RKNN3Context::load_media_model(bool user_mem_internal, uint32_t core_mask) {
    if (!this->load_model(this->media_context, this->media_model_path, this->media_weight_path, user_mem_internal, core_mask)) {
        return false;
    }
    if (!this->init_media_tensors()) {
        return false;
    }
    if (!this->init_internal_tensor_mems()) {
        return false;
    }
    return true;
}

bool caiwei::context::RKNN3Context::load_template() {
    rknn3_llm_config llm_config{};
    int ret = rknn3_query(this->context, RKNN3_QUERY_LLM_CONFIG, &llm_config, sizeof(rknn3_llm_config));
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("模型配置查询失败: %d = %s", ret, this->model_path.c_str());
        return false;
    }
    if (this->chat_template.set_template(llm_config.chat_template, this->special_token.bos, this->special_token.eos)) {
        return true;
    } else {
        CW_LOG_W("模板加载失败: %s", this->model_path.c_str());
        return false;
    }
}

bool caiwei::context::RKNN3Context::load_tokenizer() {
    this->tokenizer = new Tokenizer();
    if (!this->tokenizer->load(this->tokenizer_path)) {
        CW_LOG_W("分词模型加载失败: %s", this->tokenizer_path.c_str());
        return false;
    }
    this->vocab_size = this->tokenizer->get_size();
    CW_LOG_I("分词模型加载成功: %s = %d", this->tokenizer_path.c_str(), this->vocab_size);
    return true;
}

bool caiwei::context::RKNN3Context::load_embedding() {
    this->embedding_fd = ::open(this->embedding_path.c_str(), O_RDONLY);
    if (this->embedding_fd == -1) {
        CW_LOG_W("嵌入文件打开失败: %s", this->embedding_path.c_str());
        return false;
    }
    if (::fstat(this->embedding_fd, &this->embedding_stat) == -1) {
        CW_LOG_W("嵌入文件查询失败: %s", this->embedding_path.c_str());
        return false;
    }
    this->embedding_data = (float16*) ::mmap(nullptr, this->embedding_stat.st_size, PROT_READ, MAP_PRIVATE, this->embedding_fd, 0);
    if (this->embedding_data == MAP_FAILED) {
        CW_LOG_W("嵌入文件映射失败: %s", this->embedding_path.c_str());
        return false;
    }
    this->embedding_dim = this->embedding_stat.st_size / this->vocab_size / sizeof(float16);
    CW_LOG_I("嵌入文件加载成功: %s = %d", this->embedding_path.c_str(), this->embedding_dim);
    return true;
}

bool caiwei::context::RKNN3Context::init_media_tensors() {
    rknn3_input_output_num io_num{};
    int ret = rknn3_query(this->media_context, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(rknn3_input_output_num));
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("媒体输入输出查询失败: %d = %s", ret, this->media_model_path.c_str());
        return false;
    }
    this->media_input.resize(io_num.n_input);
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        auto& tensor = this->media_input[i];
        if (!init_tensor(this->media_context, tensor, i, RKNN3_QUERY_INPUT_ATTR, "媒体输入", 0)) {
            return false;
        }
        // 裁剪模型参考官网
        if (tensor.attr->layout != RKNN3_TENSOR_NCHW && tensor.attr->layout != RKNN3_TENSOR_NHWC) {
            CW_LOG_W("不支持的数据布局: %s", rknn3_get_layout_string(tensor.attr->layout));
            return false;
        }
    }
    this->media_output.resize(io_num.n_output);
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        auto& tensor = this->media_output[i];
        if (!init_tensor(this->media_context, tensor, i, RKNN3_QUERY_OUTPUT_ATTR, "媒体输出", 0)) {
            return false;
        }
    }
    return true;
}

bool caiwei::context::RKNN3Context::init_internal_tensor_mems() {
    uint32_t core_num_llm   = 0;
    uint32_t core_num_media = 0;
    int ret = rknn3_query(this->context, RKNN3_QUERY_CORE_NUMBER, &core_num_llm, sizeof(uint32_t));
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("查询核心配置失败: %d - %s", ret, this->model_path.c_str());
        return false;
    }
    ret = rknn3_query(this->media_context, RKNN3_QUERY_CORE_NUMBER, &core_num_media, sizeof(uint32_t));
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("查询核心配置失败: %d - %s", ret, this->media_model_path.c_str());
        return false;
    }
    std::vector<rknn3_core_mem_size> core_mem_size_llm(core_num_llm);
    std::vector<rknn3_core_mem_size> core_mem_size_media(core_num_media);
    ret = rknn3_query(this->context, RKNN3_QUERY_CORE_MEM_SIZE, core_mem_size_llm.data(), sizeof(rknn3_core_mem_size) * core_num_llm);
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("查询核心内存失败: %d", ret);
        return false;
    }
    ret = rknn3_query(this->media_context, RKNN3_QUERY_CORE_MEM_SIZE, core_mem_size_media.data(), sizeof(rknn3_core_mem_size) * core_num_media);
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("查询核心内存失败: %d", ret);
        return false;
    }
    const uint32_t core_num_max = std::max(core_num_llm, core_num_media);
    for (uint32_t i = 0; i < core_num_max; ++i) {
        rknn3_tensor_mem* mem = nullptr;
        uint64_t internal_size = 0;
        if (i < core_num_llm && i < core_num_media) {
            if (core_mem_size_llm[i].core_id != core_mem_size_media[i].core_id) {
                CW_LOG_W("模型核心匹配失败: %u = %d != %d", i, core_mem_size_llm[i].core_id, core_mem_size_media[i].core_id);
                return false;
            }
            internal_size = std::max(core_mem_size_llm[i].internal_size, core_mem_size_media[i].internal_size);
            mem = rknn3_create_mem(this->context, internal_size, core_mem_size_llm[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        } else if (i < core_num_llm) {
            internal_size = core_mem_size_llm[i].internal_size;
            mem = rknn3_create_mem(this->context, internal_size, core_mem_size_llm[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        } else if (i < core_num_media) {
            internal_size = core_mem_size_media[i].internal_size;
            mem = rknn3_create_mem(this->context, internal_size, core_mem_size_media[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        } else {
            // -
        }
        if (mem == nullptr) {
            CW_LOG_W("共享内存创建失败: %u", i);
            return false;
        }
        CW_LOG_I("共享内存创建成功: %u = %p -> %" PRIu64 " == %" PRIu64, i, mem, mem->size, internal_size);
        this->internal_tensor_mems.push_back(mem);
    }
    ret = rknn3_set_internal_mem(this->context, this->internal_tensor_mems.data(), core_num_llm);
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("共享内存设置失败: %d", ret);
        return false;
    }
    ret = rknn3_set_internal_mem(this->media_context, this->internal_tensor_mems.data(), core_num_media);
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("共享内存设置失败: %d", ret);
        return false;
    }
    return true;
}

rknn3_session* caiwei::context::RKNN3Context::get_session(ContextSession* context_session, const caiwei::text::CompletionsRequest* request) {
    rknn3_sampling_params sampling_params{};
    if (request != nullptr) {
        sampling_params.top_k             = request->top_k.value_or(10); // 注意: 建议设置不要超过20
        sampling_params.top_p             = request->top_p.value_or(0.95F);
        sampling_params.temperature       = request->temperature.value_or(0.8F);
        sampling_params.repeat_penalty    = request->repeat_penalty.value_or(1.04F);
        sampling_params.frequency_penalty = request->frequency_penalty.value_or(0.0F);
        sampling_params.presence_penalty  = request->presence_penalty.value_or(0.0F);
    }
    rknn3_llm_param llm_params{};
    llm_params.logits_name            = "logits";
    llm_params.sampling_param         = sampling_params;
    llm_params.max_context_len        = this->max_token_length;
    llm_params.vocab_info.vocab_size  = this->tokenizer->get_size();
    llm_params.vocab_info.linefeed_id = this->tokenizer->get_nl();
    llm_params.vocab_info.n_special_bos_id = 1;
    llm_params.vocab_info.n_special_eos_id = 1;
    llm_params.vocab_info.special_bos_id[0] = this->tokenizer->get_bos();
    llm_params.vocab_info.special_eos_id[0] = this->tokenizer->get_eos();
    RKLLMCallback callback{};
    callback.embed_callback     = embed_callback;
    callback.embed_userdata     = context_session;
    callback.result_callback    = result_callback;
    callback.result_userdata    = context_session;
    callback.tokenizer_callback = tokenizer_callback;
    callback.tokenizer_userdata = context_session;
    if (!context_session->output_tensors.empty()) {
        callback.output_callback  = output_callback;
        callback.output_userdata  = context_session;
        callback.output_tensors   = context_session->output_tensors.data();
        callback.n_output_tensors = context_session->output_tensors.size();
    }
    rknn3_session* session = rknn3_session_init(this->context, &llm_params, 1);
    if (!session) {
        CW_LOG_W("会话创建失败");
        return nullptr;
    }
    int ret = rknn3_session_set_chat_template(session, "", "", "");
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("会话模板设置失败: %d", ret);
        rknn3_session_destroy(session);
        return nullptr;
    }
    ret = rknn3_session_set_callback(session, &callback);
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("会话回调设置失败: %d", ret);
        rknn3_session_destroy(session);
        return nullptr;
    }
    return session;
}

std::vector<rknn3_llm_input> caiwei::context::RKNN3Context::get_inputs(std::string& /* prompt */, ContextSession* /* context_session */, const caiwei::text::CompletionsRequest& /* request */) {
    CW_LOG_W("输入逻辑没有实现");
    return {};
}

std::vector<std::vector<float>> caiwei::context::RKNN3Context::pooling(uint32_t& prompt_tokens, const std::vector<const std::string*>& prompt_list, bool add_special, bool parse_special) {
    if (prompt_list.empty()) {
        CW_LOG_W("输入为空");
        return {};
    }
    ContextSession context_session;
    context_session.context        = this->context;
    context_session.media_context  = this->media_context;
    context_session.tokenizer      = this->tokenizer;
    context_session.embedding_dim  = this->embedding_dim;
    context_session.embedding_data = this->embedding_data;
    context_session.add_special    = add_special;
    context_session.parse_special  = parse_special;
    if (!context_session.init_output_tensors()) {
        return {};
    }
    context_session.pool_result.resize(context_session.output_tensors.size());
    rknn3_session_ptr session{ this->get_session(&context_session, nullptr) };
    if (!session) {
        return {};
    }
    for (size_t i = 0; i < context_session.output_tensors.size(); ++i) {
        context_session.pool_result[i].resize(context_session.output_tensors[i].attr->n_elems);
    }
    std::vector<std::vector<float>> result;
    for (const auto& prompt : prompt_list) {
        context_session.n_prefill_tokens = 0;
        std::vector<rknn3_llm_input> inputs(1);
        rknn3_llm_tensor tensor {
            .name     = "input_embeds",
            .prompt   = prompt->c_str(),
            .embed    = nullptr,
            .tokens   = nullptr,
            .n_tokens = 0,
            .enable_thinking = false
        };
        inputs[0].input_type = RKNN3_LLM_INPUT_PROMPT;
        inputs[0].llm_input  = tensor;
        rknn3_llm_infer_param llm_infer_param{};
        llm_infer_param.keep_history   = 0;
        llm_infer_param.max_new_tokens = 1; // 即使prefill_only也不能零
        llm_infer_param.prefill_only   = true;
        auto b_time = std::chrono::steady_clock::now();
        int ret = rknn3_session_run(session.get(), inputs.data(), inputs.size(), &llm_infer_param);
        auto e_time = std::chrono::steady_clock::now();
        context_session.prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(e_time - b_time).count();
        if (ret != RKNN3_SUCCESS) {
            CW_LOG_W("会话同步运行失败: %d", ret);
            return {};
        }
        for (size_t i = 0; i < context_session.output_tensors.size(); ++i) {
            auto& pool_result = context_session.pool_result[i];
            caiwei::context::normalize_l2(pool_result.data(), pool_result.data(), pool_result.size());
            result.push_back(pool_result);
        }
        prompt_tokens += context_session.n_prefill_tokens;
        #if ENABLE_CAIWEI_PERF
        rknn3_perf_session_print(session.get(), &context_session);
        #endif
    }
    return result;
}

std::generator<caiwei::text::CompletionResult> caiwei::context::RKNN3Context::generate(const caiwei::text::CompletionsRequest& request) {
    if (request.messages.empty()) {
        CW_LOG_W("输入为空");
        co_return;
    }
    ContextSession context_session;
    context_session.context        = this->context;
    context_session.media_context  = this->media_context;
    context_session.tokenizer      = this->tokenizer;
    context_session.embedding_dim  = this->embedding_dim;
    context_session.embedding_data = this->embedding_data;
    context_session.b_thinking = this->tokenizer->piece_to_token(this->special_token.b_thinking);
    context_session.e_thinking = this->tokenizer->piece_to_token(this->special_token.e_thinking);
    context_session.b_toolcall = this->tokenizer->piece_to_token(this->special_token.b_toolcall);
    context_session.e_toolcall = this->tokenizer->piece_to_token(this->special_token.e_toolcall);
    rknn3_session_ptr session{ this->get_session(&context_session, &request) };
    if (!session) {
        co_return;
    }
    std::string prompt;
    auto b_time = std::chrono::steady_clock::now();
    std::vector<rknn3_llm_input> inputs = this->get_inputs(prompt, &context_session, request);
    auto e_time = std::chrono::steady_clock::now();
    context_session.feature_ms = std::chrono::duration_cast<std::chrono::milliseconds>(e_time - b_time).count();
    if (inputs.empty()) {
        CW_LOG_W("输入为空");
        co_return;
    }
    CW_LOG_D("提示词: %s", prompt.c_str());
    rknn3_llm_infer_param llm_infer_param{};
    llm_infer_param.keep_history   = 0;
    llm_infer_param.max_new_tokens = request.get_max_completion_tokens(this->max_token_length);
    b_time = std::chrono::steady_clock::now();
    int ret = rknn3_session_run_async(session.get(), inputs.data(), inputs.size(), &llm_infer_param);
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("会话异步运行失败: %d", ret);
        co_return;
    }
    auto raii_guard = caiwei::context::RaiiGuard([&]() {
        rknn3_session_stop(session.get());
        #if ENABLE_CAIWEI_PERF
        rknn3_perf_session_print(session.get(), &context_session);
        #endif
    });
    while (true) {
        bool stop = false;
        std::vector<caiwei::text::CompletionResult> batch;
        {
            std::unique_lock<std::mutex> lock(context_session.mutex);
            if (context_session.token_result.empty() && !context_session.stop) {
                context_session.condition.wait_for(lock, std::chrono::seconds(8));
            }
            if (!context_session.token_result.empty()) {
                if (context_session.prefill_ms == 0) {
                    e_time = std::chrono::steady_clock::now();
                    context_session.prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(e_time - b_time).count();
                    // 此处不要刷新时间
                }
                batch.swap(context_session.token_result);
            }
            stop = context_session.stop;
        }
        for (auto& result : batch) {
            co_yield std::move(result);
        }
        if (stop) {
            break;
        }
    }
    e_time = std::chrono::steady_clock::now();
    context_session.generate_ms = std::chrono::duration_cast<std::chrono::milliseconds>(e_time - b_time).count();
}

caiwei::context::ContextSession::~ContextSession() {
    this->release_output_tensors();
    this->release_deepstack_tensors();
}

bool caiwei::context::ContextSession::init_output_tensors(int n_output_tensors) {
    this->release_output_tensors();
    this->output_tensors.resize(n_output_tensors);
    for (int i = 0; i < n_output_tensors; ++i) {
        auto& tensor = this->output_tensors[i];
        if (!init_tensor(this->context, tensor, i, RKNN3_QUERY_OUTPUT_ATTR, "模型输出", 0)) {
            return false;
        }
    }
    return true;
}

void caiwei::context::ContextSession::release_output_tensors() {
    if (!this->output_tensors.empty()) {
        for (auto& tensor : this->output_tensors) {
            release_tensor(this->context, tensor, "模型输出");
        }
        this->output_tensors.clear();
    }
}

bool caiwei::context::ContextSession::init_deepstack_tensors(const std::vector<rknn3_tensor>& media_output, int length, int size) {
    this->release_deepstack_tensors();
    this->deepstack_tensors.resize(size);
    for (size_t i = 0; i < this->deepstack_tensors.size(); ++i) {
        auto& tensor = this->deepstack_tensors[i];
        if (!init_tensor(this->context, tensor, i + 2, RKNN3_QUERY_INPUT_ATTR, "deepstack", media_output[i + 1].attr->aligned_size * length)) {
            return false;
        }
    }
    return true;
}

void caiwei::context::ContextSession::release_deepstack_tensors() {
    if (!this->deepstack_tensors.empty()) {
        for (auto& tensor : this->deepstack_tensors) {
            release_tensor(this->context, tensor, "deepstack");
        }
        this->deepstack_tensors.clear();
    }
}

static int embed_callback(void* userdata, int32_t* tokens, uint64_t n_tokens, void* embed, uint64_t len) {
    caiwei::context::ContextSession* session = (caiwei::context::ContextSession*) userdata;
    const int embedding_size = session->embedding_dim * sizeof(float16);
    if (len != n_tokens * embedding_size) {
        CW_LOG_W("嵌入数据长度错误");
        return -1;
    }
    unsigned char* embed_ptr = (unsigned char*) embed;
    for (uint64_t n = 0; n < n_tokens; ++n) {
        std::memcpy(
            embed_ptr + n * embedding_size,
            session->embedding_data + tokens[n] * session->embedding_dim,
            embedding_size
        );
    }
    return 0;
}

static int output_callback(void* userdata, rknn3_tensor* output_tensors, uint32_t n_output_tensors, LLMOutputCallbackState state) {
    caiwei::context::ContextSession* session = (caiwei::context::ContextSession*) userdata;
    if (state == RKLLM_OUTPUT_CALLBACK_PREFILL_FINISHED) {
        std::vector<std::vector<float>>& pool_result = session->pool_result;
        for (uint32_t i = 0; i < n_output_tensors; ++i) {
            auto& tensor = output_tensors[i];
            auto& output = pool_result[i];
            output.resize(tensor.attr->n_elems);
            for (uint32_t j = 0; j < tensor.attr->n_elems; ++j) {
                output[j] = fp16_to_fp32(((float16*) tensor.mem->virt_addr)[j]);
            }
        }
    }
    return 0;
}

static int result_callback(void* userdata, RKLLMResult* result, LLMCallState state) {
    caiwei::context::ContextSession* session   = (caiwei::context::ContextSession*) userdata;
    caiwei::context::Tokenizer     * tokenizer = session->tokenizer;
    std::lock_guard<std::mutex> lock(session->mutex);
    if (state == RKLLM_RUN_NORMAL) {
        if (session->stop) {
            session->condition.notify_one();
            return 0;
        }
        for (int i = 0; i < result->num_tokens; ++i) {
            const int32_t token_id = result->token_ids[i];
            if (token_id == LLAMA_TOKEN_NULL) {
                CW_LOG_W("采样失败: %d", token_id);
                session->token_result.push_back(caiwei::text::CompletionResult{ caiwei::text::FINISH_REASON_STOP, session->n_prefill_tokens, session->n_generate_tokens });
                session->stop = true;
                session->condition.notify_one();
                return 0;
            }
            ++session->n_generate_tokens;
            if (tokenizer->is_eog(token_id)) {
                #if ENABLE_CAIWEI_PERF
                std::printf("\n");
                std::fflush(stdout);
                #endif
                CW_LOG_D("采样结束: %d", token_id);
                session->token_result.push_back(caiwei::text::CompletionResult{ session->thinking, session->toolcall, session->n_prefill_tokens, session->n_generate_tokens });
                session->stop = true;
                session->condition.notify_one();
                return 0;
            }
            if (token_id == session->b_thinking) {
                session->thinking = true;
            } else if (token_id == session->e_thinking) {
                session->thinking = false;
            } else if (token_id == session->b_toolcall) {
                session->toolcall = true;
                session->toolcall_session.reset();
            } else if (token_id == session->e_toolcall) {
                session->toolcall_session.finish();
                session->token_result.push_back(caiwei::text::CompletionResult{ session->thinking, session->toolcall, "", session->toolcall_session });
            } else {
                // TODO asr_text 151704
                auto& token_buffer = session->token_buffer;
                const int32_t buffer_length = tokenizer->token_to_piece(token_id, token_buffer.get_buffer(), token_buffer.get_length());
                if (buffer_length < 0) {
                    CW_LOG_W("解码失败: %d", buffer_length);
                    session->token_result.push_back(caiwei::text::CompletionResult{ caiwei::text::FINISH_REASON_STOP, session->n_prefill_tokens, session->n_generate_tokens });
                    session->stop = true;
                    session->condition.notify_one();
                    return 0;
                }
                std::string piece = token_buffer.get_token(buffer_length);
                #if ENABLE_CAIWEI_PERF
                std::printf("%s", piece.c_str());
                std::fflush(stdout);
                #endif
                session->token_result.push_back(caiwei::text::CompletionResult{ session->thinking, session->toolcall, std::move(piece), session->toolcall_session });
            }
        }
        session->condition.notify_one();
    } else if (state == RKLLM_RUN_FINISH) {
        CW_LOG_I("会话运行完成");
        if (session->stop) {
            session->condition.notify_one();
            return 0;
        }
        session->token_result.push_back(caiwei::text::CompletionResult{ session->thinking, session->toolcall, session->n_prefill_tokens, session->n_generate_tokens });
        session->stop = true;
        session->condition.notify_one();
    } else if (state == RKLLM_RUN_STOP) {
        CW_LOG_I("会话运行停止");
        if (session->stop) {
            session->condition.notify_one();
            return 0;
        }
        session->token_result.push_back(caiwei::text::CompletionResult{ caiwei::text::FINISH_REASON_STOP, session->n_prefill_tokens, session->n_generate_tokens });
        session->stop = true;
        session->condition.notify_one();
    } else if (state == RKLLM_RUN_ERROR) {
        CW_LOG_W("会话运行错误");
        if (session->stop) {
            session->condition.notify_one();
            return 0;
        }
        session->token_result.push_back(caiwei::text::CompletionResult{ caiwei::text::FINISH_REASON_STOP, session->n_prefill_tokens, session->n_generate_tokens });
        session->stop = true;
        session->condition.notify_one();
    } else if (state == RKLLM_RUN_WAITING) {
        CW_LOG_W("会话运行告警");
    } else if (state == RKLLM_RUN_MAX_NEW_TOKEN_REACHED) {
        CW_LOG_W("会话词元超量");
        if (session->stop) {
            session->condition.notify_one();
            return 0;
        }
        session->token_result.push_back(caiwei::text::CompletionResult{ caiwei::text::FINISH_REASON_LENGTH, session->n_prefill_tokens, session->n_generate_tokens });
        session->stop = true;
        session->condition.notify_one();
    } else {
        CW_LOG_W("会话未知状态: %d", state);
    }
    return 0;
}

static int tokenizer_callback(void* userdata, const char* text, int32_t text_len, int32_t* tokens, int32_t n_tokens_max) {
    caiwei::context::ContextSession* session   = (caiwei::context::ContextSession*) userdata;
    caiwei::context::Tokenizer     * tokenizer = session->tokenizer;
    const int n_tokens = tokenizer->tokenize(text, text_len, tokens, n_tokens_max, session->add_special, session->parse_special);
    if (n_tokens <= 0) {
        CW_LOG_W("分词失败: %s", text);
        return n_tokens;
    }
    CW_LOG_D("分词成功: %d => %s", n_tokens, text);
    session->n_prefill_tokens += n_tokens;
    return n_tokens;
}

void caiwei::context::rknn3_perf_session_print(rknn3_session* session, caiwei::context::ContextSession* context_session) {
    RKLLMRunState state{};
    int ret = rknn3_session_query_state(session, &state);
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("会话状态查询失败: %d", ret);
        return;
    }
    CW_LOG_I("会话完成: %" PRIu64 "=%" PRIu32 " | %" PRIu64 "=%" PRIu32, state.n_prefill_tokens, context_session->n_prefill_tokens, state.n_decode_tokens, context_session->n_generate_tokens);
    context_session->n_prefill_tokens  = state.n_prefill_tokens;
    context_session->n_generate_tokens = state.n_decode_tokens;
    std::printf("\n");
    std::printf("--------------------------------------------------------------------------------------\n");
    std::printf(" %-12s  %-15s  %-8s  %-23s  %-23s\n", "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
    std::printf("--------------------------------------------------------------------------------------\n");
    const float prefill_ms = context_session->prefill_ms;
    const uint32_t prefill_n_tokens = context_session->n_prefill_tokens;
    if (prefill_n_tokens > 0 && prefill_ms > 0.0F) {
        const float prefill_tpt =        prefill_ms / prefill_n_tokens;
        const float prefill_tps = 1e3F / prefill_ms * prefill_n_tokens;
        std::printf(" %-12s  %-15.2f  %-8u  %-23.2f  %-23.2f\n", "Prefill", prefill_ms, prefill_n_tokens, prefill_tpt, prefill_tps);
    }
    const float generate_ms = context_session->generate_ms;
    const uint32_t generate_n_tokens = context_session->n_generate_tokens;
    if (generate_n_tokens > 0 && generate_ms > 0.0F) {
        const float generate_tpt =        generate_ms / generate_n_tokens;
        const float generate_tps = 1e3F / generate_ms * generate_n_tokens;
        std::printf(" %-12s  %-15.2f  %-8u  %-23.2f  %-23.2f\n", "Generate", generate_ms, generate_n_tokens, generate_tpt, generate_tps);
    }
    const float feature_ms = context_session->feature_ms;
    const int audio_frames = context_session->audio_frames;
    const int image_frames = context_session->image_frames;
    if (audio_frames > 0 && feature_ms > 0.0F) {
        std::printf("--------------------------------------------------------------------------------------\n");
        const float fps = audio_frames * 1000.0F / feature_ms;
        std::printf(" Audio latency = %.2f ms, FPS = %.2f\n", feature_ms, fps);
    }
    if (image_frames > 0 && feature_ms > 0.0F) {
        std::printf("--------------------------------------------------------------------------------------\n");
        const float fps = image_frames * 1000.0F / feature_ms;
        std::printf(" Image latency = %.2f ms, FPS = %.2f\n", feature_ms, fps);
    }
    std::printf("--------------------------------------------------------------------------------------\n");
    std::printf("\n");
}

void caiwei::context::rknn3_dump_tensor_attr(const char* name, rknn3_tensor_attr* attr) {
    std::string shape = "";
    for (uint32_t j = 0; j < attr->n_dims; j++) {
        shape += std::to_string(attr->shape[j]);
        if (j < attr->n_dims - 1) {
            shape += ", ";
        }
    }
    std::string stride = "";
    for (uint32_t j = 0; j < attr->n_stride; j++) {
        stride += std::to_string(attr->stride[j]);
        if (j < attr->n_stride - 1) {
            stride += ", ";
        }
    }
    CW_LOG_I(
        "%s: %u => name: %s aligned_size: %" PRIu64 " n_dims: %u n_elems: %u n_stride: %u shape: [%s] stride: [%s] core_id: %d "
        "layout: %s dtype: %s qnt_type: %s",
        name,
        attr->index,
        attr->name,
        attr->aligned_size,
        attr->n_dims,
        attr->n_elems,
        attr->n_stride,
        shape.c_str(),
        stride.c_str(),
        attr->core_id,
        rknn3_get_layout_string(attr->layout),
        rknn3_get_type_string(attr->dtype),
        rknn3_get_qnt_type_string(attr->qnt_type)
    );
}

static bool init_tensor(rknn3_context& context, rknn3_tensor& tensor, int i, rknn3_query_cmd query_cmd, const char* name, uint64_t size_override) {
    tensor.attr = new rknn3_tensor_attr{};
    tensor.attr->index = i;
    int ret = rknn3_query(context, query_cmd, tensor.attr, sizeof(rknn3_tensor_attr));
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("%s参数查询失败: %d = %d", name, i, ret);
        return false;
    }
    caiwei::context::rknn3_dump_tensor_attr(name, tensor.attr);
    if (size_override == 0) {
        tensor.mem = rknn3_create_mem(context, tensor.attr->aligned_size, tensor.attr->core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
    } else {
        tensor.mem = rknn3_create_mem(context, size_override, tensor.attr->core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
    }
    if (tensor.mem == nullptr) {
        CW_LOG_W("%s内存创建失败: %d", name, i);
        return false;
    }
    CW_LOG_D("%s内存创建成功: %d = %" PRIu64, name, i, tensor.mem->size);
    return true;
}

#if CAIWEI_DEBUG
static void release_tensor(rknn3_context& context, rknn3_tensor& tensor, const char* name) {
#else
static void release_tensor(rknn3_context& context, rknn3_tensor& tensor, const char* /* name */) {
#endif
    if (tensor.mem) {
        CW_LOG_D("释放%s内存: %p", name, tensor.mem);
        rknn3_destroy_mem(context, tensor.mem);
        tensor.mem = nullptr;
    }
    if (tensor.attr) {
        CW_LOG_D("释放%s参数: %p", name, tensor.attr);
        delete tensor.attr;
        tensor.attr = nullptr;
    }
}
