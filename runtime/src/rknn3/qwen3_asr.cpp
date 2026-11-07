#include "caiwei/log.hpp"
#include "caiwei/audio_tool.hpp"
#include "caiwei/runtime/rknn3.hpp"

#define N_FFT 400
#define HOP_LENGTH 160
#define DOWN_SAMPLE_RATE 8

static int get_n_per_window_token(int window_size);
static int get_n_left_token(int window_size, int n_frame);
static int get_n_audio(int window_size, int n_frame);

caiwei::context::ASRRKNN3Context::ASRRKNN3Context(
    std::string model_path,
    std::string weight_path,
    std::string embedding_path,
    std::string tokenizer_path,
    std::string media_model_path,
    std::string media_weight_path,
    int32_t max_token_length,
    caiwei::text::SpecialToken special_token,
    caiwei::runtime::Runtime* runtime
) : ASRContext(runtime),
    RKNN3Context(
        std::move(model_path),
        std::move(weight_path),
        std::move(embedding_path),
        std::move(tokenizer_path),
        std::move(media_model_path),
        std::move(media_weight_path),
        max_token_length,
        std::move(special_token)
    ) {
}

caiwei::context::ASRRKNN3Context::~ASRRKNN3Context() {
}

bool caiwei::context::ASRRKNN3Context::load_mel_filters() {
    return caiwei::audio::load_mel_filters(this->n_mels, N_FFT, this->mel_filters);
}

bool caiwei::context::ASRRKNN3Context::init_prompt_token() {
    std::string prefix;
    std::string suffix;
    prefix
        .append(this->special_token.bos)
        .append(caiwei::text::ROLE_USER)
        .append("\n")
        .append(this->special_token.b_audio);
    suffix
        .append(this->special_token.e_audio)
        .append(this->special_token.eos)
        .append("\n")
        .append(this->special_token.bos)
        .append(caiwei::text::ROLE_ASSISTANT)
        .append("\n");
    CW_LOG_I("音频开始提示词: %s", prefix.c_str());
    CW_LOG_I("音频结束提示词: %s", suffix.c_str());
    int n_prefix_token = this->tokenizer->tokenize(prefix.c_str(), prefix.size(), this->prefix_token.data(), this->prefix_token.size(), false, true);
    if (n_prefix_token < 0) {
        this->prefix_token.resize(-n_prefix_token);
        n_prefix_token = this->tokenizer->tokenize(prefix.c_str(), prefix.size(), this->prefix_token.data(), this->prefix_token.size(), false, true);
    }
    int n_suffix_token = this->tokenizer->tokenize(suffix.c_str(), suffix.size(), this->suffix_token.data(), this->suffix_token.size(), false, true);
    if (n_suffix_token < 0) {
        this->suffix_token.resize(-n_suffix_token);
        n_suffix_token = this->tokenizer->tokenize(suffix.c_str(), suffix.size(), this->suffix_token.data(), this->suffix_token.size(), false, true);
    }
    if (n_prefix_token <= 0) {
        CW_LOG_W("音频开始提示词编码失败: %d", n_prefix_token);
        return false;
    }
    if (n_suffix_token <= 0) {
        CW_LOG_W("音频结束提示词编码失败: %d", n_suffix_token);
        return false;
    }
    CW_LOG_I("音频开始提示词词元数量: %d", n_prefix_token);
    CW_LOG_I("音频结束提示词词元数量: %d", n_suffix_token);
    return true;
}

bool caiwei::context::ASRRKNN3Context::load() {
    if (!(
        this->load_model(true) &&
        this->load_media_model(true)
    )) {
        return false;
    }
    if (this->media_input.empty() || this->media_output.empty()) {
        CW_LOG_W("模型没有加载成功");
        return false;
    }
    auto& input_tensor  = this->media_input[0];
    auto& output_tensor = this->media_output[0];
    if (input_tensor.attr->n_dims != 4 || output_tensor.attr->n_dims != 2) {
        CW_LOG_W("模型维度错误: %d != 4 || %d != 2", input_tensor.attr->n_dims, output_tensor.attr->n_dims);
        return false;
    }
    this->batch_size    = input_tensor.attr->shape[0];
    this->n_mels        = input_tensor.attr->shape[2];
    this->window_size   = input_tensor.attr->shape[3];
    this->n_chunk_token = output_tensor.attr->shape[0];
    this->chunk_length  = output_tensor.attr->shape[1];
    if (this->chunk_length != this->embedding_dim) {
        CW_LOG_W("模型维度错误: %d != %d", this->chunk_length, this->embedding_dim);
        return false;
    }
    const int batch_n_chunk_token = this->batch_size * get_n_per_window_token(this->window_size);
    if (batch_n_chunk_token != this->n_chunk_token) {
        CW_LOG_W("模型维度错误: %d != %d", batch_n_chunk_token, this->n_chunk_token);
        return false;
    }
    return this->load_mel_filters() && this->init_prompt_token();
}

std::generator<caiwei::text::CompletionResult> caiwei::context::ASRRKNN3Context::run(const caiwei::text::CompletionsRequest& request) {
    return this->generate(request);
}

bool caiwei::context::ASRRKNN3Context::fill_media(
    int& n_frames,
    int& n_tokens,
    std::vector<float16>& audio_embeds,
    const std::vector<uint8_t>& audio_data
) {
    n_frames = audio_data.size() * sizeof(uint8_t) / sizeof(int16_t);
    if (n_frames < (N_FFT + HOP_LENGTH) * 2) {
        CW_LOG_W("音频数据长度不足: %d", n_frames);
        return false;
    }
    int ret = 0;
    const int n_frame = std::floor((n_frames - N_FFT) / HOP_LENGTH ) + 1;
    const int n_audio = get_n_audio(this->window_size, n_frame);
    const int n_prefix_token = this->prefix_token.size();
    const int n_suffix_token = this->suffix_token.size();
    n_tokens = n_audio + n_prefix_token + n_suffix_token;
    CW_LOG_D("音频数据长度: %d => %d : %d = %d", n_frames, n_frame, n_audio, n_tokens);
    if (n_tokens > this->max_token_length) {
        CW_LOG_W("音频数据词元数量超过限制: %d > %d", n_tokens, this->max_token_length);
        return false;
    }
    std::vector<float> audio_data_f32(n_frames);
    const int16_t* audio_data_ptr = reinterpret_cast<const int16_t*>(audio_data.data());
    std::transform(audio_data_ptr, audio_data_ptr + n_frames, audio_data_f32.data(), [](int16_t v) {
        return static_cast<float>(v) / 32768.0F;
    });
    std::vector<float> audio_feature(this->n_mels * n_frame, 0.0F);
    caiwei::audio::log_mel_spectrogram(
        audio_data_f32.data(),
        audio_data_f32.size(),
        this->mel_filters.data(),
        N_FFT,
        HOP_LENGTH,
        this->n_mels,
        n_frame * HOP_LENGTH,
        audio_feature
    );
    audio_embeds.resize(this->chunk_length * n_tokens);
    const int batch_chunk_size = this->batch_size * this->window_size;
    for(int chunk = 0; chunk * batch_chunk_size < n_frame; ++chunk) {
        int n_chunk_token = this->n_chunk_token;
        int chunk_size = std::min(batch_chunk_size, n_frame - chunk * batch_chunk_size);
        float* dst_data = (float*) this->media_input[0].mem->virt_addr;    
        if (chunk_size < batch_chunk_size) {
            std::memset(dst_data, 0, this->n_mels * batch_chunk_size * sizeof(float));
            n_chunk_token = chunk_size / this->window_size * get_n_per_window_token(this->window_size) + get_n_left_token(this->window_size, chunk_size);
        }
        for (int batch = 0; batch < this->batch_size; ++batch) {
            int dst_pos = this->n_mels * batch * this->window_size;
            int src_pos = chunk * batch_chunk_size + batch * this->window_size;
            int batch_data_size = std::max(0, std::min(this->window_size, chunk_size - batch * this->window_size));
            if (batch_data_size > 0) {
                for (int i = 0; i < this->n_mels; ++i) {
                    float* dst = dst_data + dst_pos + i * this->window_size;
                    float* src = audio_feature.data() + i * n_frame + src_pos;
                    std::memcpy(dst, src, batch_data_size * sizeof(float));
                }
            }
        }
        for (auto& v : this->media_input) {
            ret = rknn3_mem_sync(this->media_context, v.mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
            if (ret != RKNN3_SUCCESS) {
                CW_LOG_W("输入失败: %d", ret);
                return false;
            }
        }
        ret = rknn3_run(this->media_context, this->media_input.data(), this->media_input.size(), this->media_output.data(), this->media_output.size());
        if (ret != RKNN3_SUCCESS) {
            CW_LOG_W("计算失败: %d", ret);
            return false;
        }
        for (auto& v : this->media_output) {
            ret = rknn3_mem_sync(this->media_context, v.mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
            if (ret != RKNN3_SUCCESS) {
                CW_LOG_W("输出失败: %d", ret);
                return false;
            }
        }
        std::memcpy(
            audio_embeds.data() + (chunk * this->n_chunk_token + n_prefix_token) * this->chunk_length,
            this->media_output[0].mem->virt_addr,
            n_chunk_token * this->chunk_length * sizeof(float16)
        );
    }
    for (size_t i = 0; i < this->prefix_token.size(); ++i) {
        std::memcpy(
            audio_embeds.data() + i * this->chunk_length,
            this->embedding_data + this->prefix_token[i] * this->embedding_dim,
            this->embedding_dim * sizeof(float16)
        );
    }
    for (size_t i = 0; i < this->suffix_token.size(); ++i) {
        std::memcpy(
            audio_embeds.data() + (n_audio + n_prefix_token + i) * this->chunk_length,
            this->embedding_data + this->suffix_token[i] * this->embedding_dim,
            this->embedding_dim * sizeof(float16)
        );
    }
    return true;
}

std::vector<rknn3_llm_input> caiwei::context::ASRRKNN3Context::get_inputs(std::string& /* prompt */, ContextSession* context_session, const caiwei::text::CompletionsRequest& request) {
    std::vector<uint8_t> audio_data;
    for (const auto& message : request.messages) {
        for (const auto& audio : message.audio_data) {
            audio_data.insert(audio_data.end(), audio.data.begin(), audio.data.end());
        }
    }
    if (audio_data.empty()) {
        CW_LOG_W("音频数据为空");
        return {};
    }
    int n_frames;
    int n_tokens;
    {
        std::lock_guard<std::mutex> lock(this->media_mutex);
        if (!this->fill_media(n_frames, n_tokens, context_session->audio_embeds, audio_data)) {
            return {};
        }
        context_session->audio_frames = n_frames;
    }
    rknn3_llm_tensor tensor = {
        .name     = "input_embeds",
        .prompt   = nullptr,
        .embed    = context_session->audio_embeds.data(),
        .tokens   = nullptr,
        .n_tokens = n_tokens,
        .enable_thinking = request.enable_thinking.value_or(false)
    };
    std::vector<rknn3_llm_input> inputs(1);
    inputs[0].input_type = RKNN3_LLM_INPUT_EMBED;
    inputs[0].llm_input  = tensor;
    return inputs;
}

static int get_n_per_window_token(int window_size) {
    return std::ceil(1.0 * window_size / DOWN_SAMPLE_RATE);
}

static int get_n_left_token(int window_size, int n_frame) {
    return get_n_per_window_token(n_frame % window_size);
}

static int get_n_audio(int window_size, int n_frame) {
    const int n_left_token = get_n_left_token(window_size, n_frame);
    const int n_per_window_token = get_n_per_window_token(window_size);
    return std::floor(n_frame / window_size) * n_per_window_token + n_left_token;
}
