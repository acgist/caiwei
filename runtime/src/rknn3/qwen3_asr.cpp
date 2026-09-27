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
    return this->load_model() && this->load_media_model();
}

std::generator<caiwei::text::Result> caiwei::context::ASRRKNN3Context::run(caiwei::text::CompletionsRequest& request) {
    return this->generate(request);
}

static int get_n_audio(int window_size, int audio_len) {
    int n_frame = floor((audio_len - N_FFT) / HOP_LENGTH ) + 1;
    int left_data_len = n_frame % audio_ctx->window_size;
    int n_left_token = ceil(1.0 * left_data_len / DOWN_SAMPLE_RATE);
    int n_token_per_window = ceil(1.0 * audio_ctx->window_size / DOWN_SAMPLE_RATE);
    return floor(n_frame/audio_ctx->window_size) * n_token_per_window + n_left_token;
}

std::vector<rknn3_llm_input> caiwei::context::ASRRKNN3Context::get_inputs(rknn3_session* session, ContextSession* context_session, caiwei::text::CompletionsRequest& request) {
    static std::string prefix = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>";
    static std::string suffix = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n";
    static std::vector<int32_t> prefix_token;
    static std::vector<int32_t> suffix_token;
    for (int i = 0; i < prefix_token.size(); ++i) {
        memcpy(app_ctx->prefix_embeds + i * app_ctx->audio.embeds_shape[1], embedding_info->embedding_data + prefix_tokens[i] * embedding_info->embedding_dim, embedding_info->embedding_dim * sizeof(float16));
    }
    for (int i = 0; i < suffix_token.size(); ++i) {
        memcpy(app_ctx->suffix_embeds + i * app_ctx->audio.embeds_shape[1], embedding_info->embedding_data + suffix_tokens[i] * embedding_info->embedding_dim, embedding_info->embedding_dim * sizeof(float16));
    }
    // TODO
    std::vector<float16> embed;
    int batch_size = this->media_input[0].attr->shape[0];
    int n_mels = this->media_input[0].attr->shape[2];
    int window_size = this->media_input[0].attr->shape[3];
    int n_audio = get_n_audio(window_size, 1234); // TODO 读取
    embed.resize(this->media_output[0].attr->shape[1] * n_audio);
    memcpy(embed, app_ctx->prefix_embeds, PREFIX_N_TOKENS * app_ctx->audio.embeds_shape[1] * sizeof(float16));
    memcpy(embed+PREFIX_N_TOKENS*app_ctx->audio.embeds_shape[1], audio_embeds, n_audio_tokens * app_ctx->audio.embeds_shape[1] * sizeof(float16));
    memcpy(embed+(PREFIX_N_TOKENS+n_audio_tokens)*app_ctx->audio.embeds_shape[1], app_ctx->postfix_embeds, POSTFIX_N_TOKENS * app_ctx->audio.embeds_shape[1] * sizeof(float16));

    int mels_filters_size = N_FFT / 2 + 1;
    std::vector<float> mel_filters;
    mel_filters.resize(n_mels * mels_filters_size);
    ret = read_mel_filters(MEL_FILTERS_PATH, mel_filters, n_mels * mels_filters_size);
    if (ret != 0) {
        printf("read mel_filters fail! Please check if the file \"%s\" exists.\n", MEL_FILTERS_PATH);
        free(mel_filters);
        return -1;
    }
    int n_frame = floor((audio->num_frames - N_FFT) / HOP_LENGTH ) + 1;
    std::vector<float> padded_feature(n_mels * n_frame, 0.0f);
    int actual_len;
    audio_preprocess(audio, mel_filters, N_FFT, HOP_LENGTH, n_mels, n_frame * HOP_LENGTH, padded_feature, &actual_len);
    float* audio_input_feat = padded_feature.data();

    for(int chunk=0; chunk*batch_size*window_size < n_frame; chunk++) {
        int real_data_len = std::min(batch_size * window_size, n_frame - chunk * batch_size * window_size);
        int n_chunk_token = audio_ctx->embeds_shape[0];
        float* dst_base = (float*)audio_ctx->inputs[0].mem->virt_addr;    
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
        for (int i = 0; i < audio_ctx->io_num.n_input; i++)
        {
            ret = rknn3_mem_sync(audio_ctx->rknn_ctx, audio_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
            if (ret != RKNN3_SUCCESS)
            {
                printf("rknn3_mem_sync input[%d] failed! ret=%d\n", i, ret);
                goto out;
            }
        }
        // Run
        ret = rknn3_run(audio_ctx->rknn_ctx, audio_ctx->inputs, audio_ctx->io_num.n_input, audio_ctx->outputs, audio_ctx->io_num.n_output);
        if (ret < 0)
        {
            printf("rknn_run fail! ret=%d\n", ret);
            goto out;
        }
        // Sync Outputs
        for (int i = 0; i < audio_ctx->io_num.n_output; i++)
        {
            ret = rknn3_mem_sync(audio_ctx->rknn_ctx, audio_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
            if (ret != RKNN3_SUCCESS)
            {
                printf("rknn3_mem_sync output[%d] failed! ret=%d\n", i, ret);
                goto out;
            }
        }
        // Get Output
        memcpy((float16*)audio_embeds+(chunk*audio_ctx->embeds_shape[0]*audio_ctx->embeds_shape[1]), (float16*)audio_ctx->outputs[0].mem->virt_addr, n_chunk_token*audio_ctx->embeds_shape[1]*sizeof(float16));
    }

    rknn3_llm_tensor tensor;
    tensor.embed = embed.data();
    std::vector<rknn3_llm_input> inputs(1);
    inputs[0].input_type = RKNN3_LLM_INPUT_EMBED;
    inputs[0].llm_input = tensor;
    for (int i = 0; i < 3; ++i) {
        inputs[i + 1].input_type = RKNN3_LLM_INPUT_AUX;
        inputs[i + 1].aux_input  = deepstack_tensors[i];
    }
    return inputs;
}
