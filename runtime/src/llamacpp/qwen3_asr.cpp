#include "caiwei/log.hpp"
#include "caiwei/runtime/llamacpp.hpp"

/**
 * 参考代码
 * deps\llama.cpp\tools\mtmd\mtmd-cli.cpp
 */

caiwei::context::ASRLlamaCPPContext::ASRLlamaCPPContext(std::string path, std::string mtmd_path, std::string media_marker, int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime)
    : LlamaCPPContext(std::move(path), std::move(mtmd_path), std::move(media_marker), max_token_length, special_token),
    ASRContext(runtime) {
}

caiwei::context::ASRLlamaCPPContext::~ASRLlamaCPPContext() {
}

bool caiwei::context::ASRLlamaCPPContext::load() {
    return this->load_model() && this->load_mtmd();
}

std::generator<caiwei::text::Result> caiwei::context::ASRLlamaCPPContext::run(caiwei::text::CompletionsRequest& request) {
    return this->generate_mtmd(request);
}

void caiwei::context::ASRLlamaCPPContext::build_bitmaps(caiwei::text::CompletionsRequest& request, mtmd::bitmaps& bitmaps) {
    for (auto& message : request.messages) {
        if (message.role != caiwei::text::ROLE_USER) {
            continue;
        }
        if (message.audio_data.empty()) {
            continue;
        }
        std::string content{};
        for (const auto& audio : message.audio_data) {
            std::vector<float> audio_data(audio.data.size() * sizeof(uint8_t) / sizeof(int16_t));
            const int16_t* audio_data_ptr = reinterpret_cast<const int16_t*>(audio.data.data());
            std::transform(audio_data_ptr, audio_data_ptr + audio.data.size() * sizeof(uint8_t) / sizeof(int16_t), audio_data.data(), [](int16_t v) {
                return static_cast<float>(v) / 32768.0F;
            });
            content += this->media_marker;
            bitmaps.entries.emplace_back(mtmd_bitmap_init_from_audio(audio_data.size(), audio_data.data()));
        }
        message.content = std::move(content);
    }
}
