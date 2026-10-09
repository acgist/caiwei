#include "caiwei/log.hpp"
#include "caiwei/audio_tool.hpp"
#include "caiwei/runtime/rknn3.hpp"

#define HOP_LENGTH 160
#define N_FFT 400
#define MEL_FILTERS_PATH "mel_128_filters.txt"
#define DOWN_SAMPLE_RATE 8

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

bool caiwei::context::ASRRKNN3Context::load() {
    return this->load_model(true) && this->load_media_model(true) && this->init_internal_mems(0xFF, 0xFF);
}

std::generator<caiwei::text::Result> caiwei::context::ASRRKNN3Context::run(caiwei::text::CompletionsRequest& request) {
    return this->generate(request);
}

static int get_n_audio(int window_size, int audio_len) {
    int n_frame = floor((audio_len - N_FFT) / HOP_LENGTH ) + 1;
    int left_data_len = n_frame % window_size;
    int n_left_token = ceil(1.0 * left_data_len / DOWN_SAMPLE_RATE);
    int n_token_per_window = ceil(1.0 * window_size / DOWN_SAMPLE_RATE);
    return floor(n_frame/window_size) * n_token_per_window + n_left_token;
}

std::vector<rknn3_llm_input> caiwei::context::ASRRKNN3Context::get_inputs(rknn3_session* session, ContextSession* context_session, caiwei::text::CompletionsRequest& request) {
    int prefix_n_token = 9;
    int suffix_n_token = 11;
    auto audio_frame = request.messages[0].audio_data[0];
    int num_frames = audio_frame.data.size() / sizeof(int16_t);
    int batch_size = this->media_input[0].attr->shape[0];
    int n_mels = this->media_input[0].attr->shape[2];
    int window_size = this->media_input[0].attr->shape[3];
    int n_audio = get_n_audio(window_size, num_frames);
    int mels_filters_size = N_FFT / 2 + 1;
    std::vector<float> mel_filters;
    mel_filters.resize(n_mels * mels_filters_size);
    int ret = caiwei::audio::read_mel_filters(MEL_FILTERS_PATH, mel_filters.data(), n_mels * mels_filters_size);
    if (ret != 0) {
        printf("read mel_filters fail! Please check if the file \"%s\" exists.\n", MEL_FILTERS_PATH);
        return {};
    }
    static std::vector<float16> embed;
    embed.resize(this->media_output[0].attr->shape[1] * (n_audio + prefix_n_token + suffix_n_token));
    int n_frame = floor((num_frames - N_FFT) / HOP_LENGTH ) + 1;
    std::vector<float> padded_feature(n_mels * n_frame, 0.0f);
    int actual_len;
    std::vector<float> audio_data(audio_frame.data.size() * sizeof(uint8_t) / sizeof(int16_t));
    const int16_t* audio_data_ptr = reinterpret_cast<const int16_t*>(audio_frame.data.data());
    std::transform(audio_data_ptr, audio_data_ptr + audio_frame.data.size() * sizeof(uint8_t) / sizeof(int16_t), audio_data.data(), [](int16_t v) {
        return static_cast<float>(v) / 32768.0F;
    });
    caiwei::audio::audio_preprocess(audio_data.data(), audio_data.size(), mel_filters.data(), N_FFT, HOP_LENGTH, n_mels, n_frame * HOP_LENGTH, padded_feature, &actual_len);
    float* audio_input_feat = padded_feature.data();
    for(int chunk=0; chunk*batch_size*window_size < n_frame; chunk++) {
        int real_data_len = std::min(batch_size * window_size, n_frame - chunk * batch_size * window_size);
        int n_chunk_token = this->media_output[0].attr->shape[0];
        float* dst_base = (float*)this->media_input[0].mem->virt_addr;    
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
                    float* src = (float*)audio_input_feat + i * n_frame + src_frame_start;
                    memcpy(dst, src, batch_real_len * sizeof(float));
                }
            }
        }
        // sync inputs
        for (auto& v : this->media_input)
        {
            ret = rknn3_mem_sync(this->media_context, v.mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
            if (ret != RKNN3_SUCCESS)
            {
                printf("rknn3_mem_sync input[%d] failed! ret=%d\n", 0, ret);
            }
        }
        // Run
        ret = rknn3_run(this->media_context, this->media_input.data(), this->media_output.size(), this->media_output.data(), this->media_output.size());
        if (ret < 0)
        {
            printf("rknn_run fail! ret=%d\n", ret);
        }
        // Sync Outputs
        for (auto& v : this->media_output)
        {
            ret = rknn3_mem_sync(this->media_context, v.mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
            if (ret != RKNN3_SUCCESS)
            {
                printf("rknn3_mem_sync output[%d] failed! ret=%d\n", 0, ret);
            }
        }
        memcpy(
            (float16*) embed.data() + ((chunk * this->media_output[0].attr->shape[0] + prefix_n_token) * this->media_output[0].attr->shape[1]),
            (float16*) this->media_output[0].mem->virt_addr,
            n_chunk_token * this->media_output[0].attr->shape[1] * sizeof(float16)
        );
    }
    static std::string prefix = "<|im_start|>user\n<|audio_start|>";
    static std::string suffix = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n";
    static std::vector<int32_t> prefix_token;
    static std::vector<int32_t> suffix_token;
    prefix_token.resize(prefix_n_token);
    suffix_token.resize(suffix_n_token);
    int n_prefix_token = this->tokenizer->tokenize(prefix.c_str(), prefix.size(), prefix_token.data(), prefix_token.size());
    int n_suffix_token = this->tokenizer->tokenize(suffix.c_str(), suffix.size(), suffix_token.data(), suffix_token.size());
    CW_LOG_I("n_prefix_token = %d, n_suffix_token = %d", n_prefix_token, n_suffix_token);
    for (int i = 0; i < prefix_token.size(); ++i) {
        memcpy(
            embed.data() + i * this->media_output[0].attr->shape[1],
            this->embedding_data + prefix_token[i] * this->embedding_dim,
            this->embedding_dim * sizeof(float16)
        );
    }
    for (int i = 0; i < suffix_token.size(); ++i) {
        memcpy(
            embed.data() + this->media_output[0].attr->shape[1] * (n_audio + prefix_n_token + i),
            this->embedding_data + suffix_token[i] * this->embedding_dim,
            this->embedding_dim * sizeof(float16)
        );
    }
    rknn3_llm_tensor tensor = {
        .name = NULL,
        .prompt = NULL,
        .embed = embed.data(),
        .tokens = NULL,
        .n_tokens = n_audio + prefix_n_token + suffix_n_token,
        .enable_thinking = false
    };
    std::vector<rknn3_llm_input> inputs(1);
    inputs[0].input_type = RKNN3_LLM_INPUT_EMBED;
    inputs[0].llm_input = tensor;
    return inputs;
}
