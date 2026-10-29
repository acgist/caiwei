#include "caiwei/log.hpp"
#include "caiwei/runtime/llamacpp.hpp"

caiwei::context::ASRLlamaCPPContext::ASRLlamaCPPContext(
    std::string path,
    std::string mtmd_path,
    int32_t max_token_length,
    caiwei::text::SpecialToken special_token,
    caiwei::runtime::Runtime* runtime
) : ASRContext(runtime),
    LlamaCPPContext(std::move(path), std::move(mtmd_path), max_token_length, std::move(special_token)) {
}

caiwei::context::ASRLlamaCPPContext::~ASRLlamaCPPContext() {
}

bool caiwei::context::ASRLlamaCPPContext::load() {
    return this->load_model() && this->load_mtmd_model();
}

std::generator<caiwei::text::CompletionResult> caiwei::context::ASRLlamaCPPContext::run(const caiwei::text::CompletionsRequest& request) {
    return this->generate(request);
}

caiwei::context::caiwei_llama_batch caiwei::context::ASRLlamaCPPContext::get_inputs(std::string& prompt, llama_context* context, const caiwei::text::CompletionsRequest& request) const {
    mtmd::bitmaps bitmaps;
    prompt
        .append(this->special_token.bos)
        .append(caiwei::text::ROLE_USER)
        .append("\n");
    for (const auto& message : request.messages) {
        if (message.role != caiwei::text::ROLE_USER) {
            continue;
        }
        if (message.audio_data.empty()) {
            continue;
        }
        for (const auto& audio : message.audio_data) {
            const int num_frames = audio.data.size() * sizeof(uint8_t) / sizeof(int16_t);
            std::vector<float> audio_data(num_frames);
            const int16_t* audio_data_ptr = reinterpret_cast<const int16_t*>(audio.data.data());
            std::transform(audio_data_ptr, audio_data_ptr + num_frames, audio_data.data(), [](int16_t v) {
                return static_cast<float>(v) / 32768.0F;
            });
            prompt.append(this->special_token.media_marker);
            bitmaps.entries.emplace_back(mtmd_bitmap_init_from_audio(audio_data.size(), audio_data.data()));
        }
    }
    prompt
        .append(this->special_token.eos)
        .append("\n")
        .append(this->special_token.bos)
        .append(caiwei::text::ROLE_ASSISTANT)
        .append("\n");
    caiwei_llama_batch caiwei_batch{};
    if (this->get_mtmd_inputs(caiwei_batch, bitmaps, prompt, context, request)) {
        caiwei_batch.success = true;
        return caiwei_batch;
    } else {
        CW_LOG_W("音频编码失败");
        caiwei_batch.success = false;
        caiwei_batch.finish_reason = caiwei::text::FINISH_REASON_STOP;
        return caiwei_batch;
    }
}
