#include "caiwei/log.hpp"
#include "caiwei/audio_tool.hpp"
#include "caiwei/runtime/rknn3.hpp"

#define N_FFT 400
#define HOP_LENGTH 160
#define DOWN_SAMPLE_RATE 8
#define MEL_FILTERS_PATH "mel_128_filters.txt"

caiwei::context::ASRRKNN3Context::ASRRKNN3Context(
    std::string model_path, std::string weight_path, std::string embedding_path, std::string tokenizer_path,
    std::string media_model_path, std::string media_weight_path,
    int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime
) : ASRContext(runtime),
    RKNN3Context(
    std::move(model_path), std::move(weight_path), std::move(embedding_path), std::move(tokenizer_path),
    std::move(media_model_path), std::move(media_weight_path), max_token_length, special_token) {
}

caiwei::context::ASRRKNN3Context::~ASRRKNN3Context() {
}

bool caiwei::context::ASRRKNN3Context::load_mel_filters() {
    int n_mels = this->media_input[0].attr->shape[2];
    int mels_filters_size = N_FFT / 2 + 1;
    int max_lines = n_mels * mels_filters_size;
    this->mel_filters.resize(max_lines);
    int line_count = 0;
    FILE* file = fopen(MEL_FILTERS_PATH, "r");
    auto* data = this->mel_filters.data();
    if (file == NULL) {
        printf("read mel_filters fail! Please check if the file \"%s\" exists.\n", MEL_FILTERS_PATH);
        return false;
    }
    while (line_count < max_lines && fscanf(file, "%f", &data[line_count]) == 1) {
        line_count++;
    }
    fclose(file);
    CW_LOG_I("n_mels = %d, mels_filters_size = %d", n_mels, mels_filters_size);
    return true;
}

bool caiwei::context::ASRRKNN3Context::init_prompt_token() {
    std::string prefix;
    std::string suffix;
    prefix.append(this->special_token.bos).append("user\n").append(this->special_token.b_audio);
    suffix.append(this->special_token.e_audio).append(this->special_token.eos).append("\n").append(this->special_token.bos).append("assistant\n");
    CW_LOG_I("音频开始token = %s", prefix.c_str());
    CW_LOG_I("音频结束token = %s", suffix.c_str());
    int n_prefix_token = this->tokenizer->tokenize(prefix.c_str(), prefix.size(), this->prefix_token.data(), this->prefix_token.size());
    if (n_prefix_token < 0) {
        this->prefix_token.resize(-n_prefix_token);
        n_prefix_token = this->tokenizer->tokenize(prefix.c_str(), prefix.size(), this->prefix_token.data(), this->prefix_token.size());
    }
    int n_suffix_token = this->tokenizer->tokenize(suffix.c_str(), suffix.size(), this->suffix_token.data(), this->suffix_token.size());
    if (n_suffix_token < 0) {
        this->suffix_token.resize(-n_suffix_token);
        n_suffix_token = this->tokenizer->tokenize(suffix.c_str(), suffix.size(), this->suffix_token.data(), this->suffix_token.size());
    }
    CW_LOG_I("n_prefix_token = %d, n_suffix_token = %d", n_prefix_token, n_suffix_token);
    return true;
}

bool caiwei::context::ASRRKNN3Context::load() {
    return this->load_model(true) &&
    this->load_media_model(true) &&
    this->init_internal_tensor_mems() &&
    this->load_mel_filters() &&
    this->init_prompt_token();
}

std::generator<caiwei::text::Result> caiwei::context::ASRRKNN3Context::run(const caiwei::text::CompletionsRequest& request) {
    return this->generate(request);
}

static int get_n_audio(int window_size, int audio_len) {
    int n_frame = floor((audio_len - N_FFT) / HOP_LENGTH ) + 1;
    int left_data_len = n_frame % window_size;
    int n_left_token = ceil(1.0 * left_data_len / DOWN_SAMPLE_RATE);
    int n_token_per_window = ceil(1.0 * window_size / DOWN_SAMPLE_RATE);
    return floor(n_frame / window_size) * n_token_per_window + n_left_token;
}

std::vector<rknn3_llm_input> caiwei::context::ASRRKNN3Context::get_inputs(std::string& prompt, ContextSession* context_session, const caiwei::text::CompletionsRequest& request) {
    // TODO 线程安全
    bool enable_thinking = request.enable_thinking.value_or(false);
    int prefix_n_token = this->prefix_token.size();
    int suffix_n_token = this->suffix_token.size();
    std::vector<uint8_t> audio_frame;
    if (request.messages.empty()) {
        CW_LOG_W("音频数据为空");
        return {};
    }
    for (const auto& message : request.messages) {
        for (const auto& audio : message.audio_data) {
            audio_frame.insert(audio_frame.end(), audio.data.begin(), audio.data.end());
        }
    }
    if (audio_frame.empty()) {
        CW_LOG_W("音频数据为空");
        return {};
    }
    const int num_frames = audio_frame.size() * sizeof(uint8_t) / sizeof(int16_t);
    context_session->audio_frames = num_frames;
    int batch_size = this->media_input[0].attr->shape[0];
    int n_mels = this->media_input[0].attr->shape[2];
    int window_size = this->media_input[0].attr->shape[3];
    int n_audio = get_n_audio(window_size, num_frames);
    this->audio_embeds.resize(this->media_output[0].attr->shape[1] * (n_audio + prefix_n_token + suffix_n_token));
    int n_frame = floor((num_frames - N_FFT) / HOP_LENGTH ) + 1;
    std::vector<float> padding_feature(n_mels * n_frame, 0.0F);
    int actual_len;
    std::vector<float> audio_data(num_frames);
    const int16_t* audio_data_ptr = reinterpret_cast<const int16_t*>(audio_frame.data());
    std::transform(audio_data_ptr, audio_data_ptr + num_frames, audio_data.data(), [](int16_t v) {
        return static_cast<float>(v) / 32768.0F;
    });
    int ret = 0;
    caiwei::audio::log_mel_spectrogram(audio_data.data(), audio_data.size(), this->mel_filters.data(), N_FFT, HOP_LENGTH, n_mels, n_frame * HOP_LENGTH, padding_feature, &actual_len);
    float* audio_input_feature = padding_feature.data();
    for(int chunk = 0; chunk * batch_size * window_size < n_frame; chunk++) {
        int real_data_len = std::min(batch_size * window_size, n_frame - chunk * batch_size * window_size);
        int n_chunk_token = this->media_output[0].attr->shape[0];
        float* dst_base = (float*) this->media_input[0].mem->virt_addr;    
        if (real_data_len < batch_size * window_size) {
            memset(dst_base, 0, (batch_size * n_mels * window_size) * sizeof(float));
            n_chunk_token = real_data_len / window_size * ceil(1.0 * window_size / DOWN_SAMPLE_RATE) + ceil(1.0 * (real_data_len % window_size) / DOWN_SAMPLE_RATE);
        }
        for (int b = 0; b < batch_size; b++) {
            int batch_offset = b * n_mels * window_size;
            int src_frame_start = chunk * batch_size * window_size + b * window_size;
            int batch_real_len = std::max(0, std::min(window_size, real_data_len - b * window_size));
            if (batch_real_len > 0) {
                for (int i = 0; i < n_mels; i++) {
                    float* dst = dst_base + batch_offset + i * window_size;
                    float* src = audio_input_feature + i * n_frame + src_frame_start;
                    memcpy(dst, src, batch_real_len * sizeof(float));
                }
            }
        }
        for (auto& v : this->media_input) {
            ret = rknn3_mem_sync(this->media_context, v.mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
            if (ret != RKNN3_SUCCESS) {
                printf("rknn3_mem_sync input[%d] failed! ret=%d\n", 0, ret);
            }
        }
        ret = rknn3_run(this->media_context, this->media_input.data(), this->media_output.size(), this->media_output.data(), this->media_output.size());
        if (ret < 0) {
            printf("rknn_run fail! ret=%d\n", ret);
        }
        for (auto& v : this->media_output) {
            ret = rknn3_mem_sync(this->media_context, v.mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
            if (ret != RKNN3_SUCCESS) {
                printf("rknn3_mem_sync output[%d] failed! ret=%d\n", 0, ret);
            }
        }
        memcpy(
            this->audio_embeds.data() + ((chunk * this->media_output[0].attr->shape[0] + prefix_n_token) * this->media_output[0].attr->shape[1]),
            this->media_output[0].mem->virt_addr,
            n_chunk_token * this->media_output[0].attr->shape[1] * sizeof(float16)
        );
    }
    for (int i = 0; i < this->prefix_token.size(); ++i) {
        memcpy(
            this->audio_embeds.data() + i * this->media_output[0].attr->shape[1],
            this->embedding_data + this->prefix_token[i] * this->embedding_dim,
            this->embedding_dim * sizeof(float16)
        );
    }
    for (int i = 0; i < this->suffix_token.size(); ++i) {
        memcpy(
            this->audio_embeds.data() + this->media_output[0].attr->shape[1] * (n_audio + prefix_n_token + i),
            this->embedding_data + this->suffix_token[i] * this->embedding_dim,
            this->embedding_dim * sizeof(float16)
        );
    }
    rknn3_llm_tensor tensor = {
        .name     = nullptr,
        .prompt   = nullptr,
        .embed    = this->audio_embeds.data(),
        .tokens   = nullptr,
        .n_tokens = n_audio + prefix_n_token + suffix_n_token,
        .enable_thinking = enable_thinking
    };
    std::vector<rknn3_llm_input> inputs(1);
    inputs[0].input_type = RKNN3_LLM_INPUT_EMBED;
    inputs[0].llm_input = tensor;
    return inputs;
}
