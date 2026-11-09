#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/runtime/llamacpp.hpp"

caiwei::context::VLMLlamaCPPContext::VLMLlamaCPPContext(
    std::string path,
    std::string mtmd_path,
    int32_t max_token_length,
    caiwei::text::SpecialToken special_token,
    caiwei::runtime::Runtime* runtime
) : VLMContext(runtime),
    LlamaCPPContext(std::move(path), std::move(mtmd_path), max_token_length, std::move(special_token)) {
}

caiwei::context::VLMLlamaCPPContext::~VLMLlamaCPPContext() {
}

bool caiwei::context::VLMLlamaCPPContext::load() {
    const int min_tokens = caiwei::env::get_int("CAIWEI_VLM_IMAGE_MIN_TOKEN");
    const int max_tokens = caiwei::env::get_int("CAIWEI_VLM_IMAGE_MAX_TOKEN");
    return this->load_model() && this->load_mtmd_model(min_tokens, max_tokens) && this->load_template();
}

std::generator<caiwei::text::CompletionResult> caiwei::context::VLMLlamaCPPContext::run(const caiwei::text::CompletionsRequest& request) {
    return this->generate(request);
}

static mtmd_bitmap* fill_media(int width, int height, int channels, const std::vector<uint8_t>& data) {
    if (data.empty() || width <= 0 || height <= 0 || channels != 3) {
        CW_LOG_W("图片格式错误: %d, %d, %d", width, height, channels);
        return nullptr;
    }
    // 不用手动缩放使用使用image_min_tokens/image_max_tokens自动适配
    return mtmd_bitmap_init(width, height, data.data());
}

static mtmd_bitmap* fill_media(const std::vector<caiwei::media::VideoFrame>& video) {
    auto* ptr = const_cast<std::vector<caiwei::media::VideoFrame>*>(&video);
    return mtmd_bitmap_init_lazy(
        nullptr,
        nullptr,
        ptr,
        [](
            size_t chunk_idx,
            void*  user_data,
            mtmd_bitmap** out_bitmap,
            char       ** /* out_text */
        ) {
            const std::vector<caiwei::media::VideoFrame>& video = *(std::vector<caiwei::media::VideoFrame>*) user_data;
            if (chunk_idx >= video.size()) {
                return -1;
            }
            const auto& frame  = video[chunk_idx];
            if (frame.data.empty() || frame.width <= 0 || frame.height <= 0 || frame.channels != 3) {
                CW_LOG_W("视频格式错误: %d, %d, %d", frame.width, frame.height, frame.channels);
                return -1;
            }
            // 不用手动缩放使用使用image_min_tokens/image_max_tokens自动适配
            *out_bitmap = mtmd_bitmap_init(frame.width, frame.height, frame.data.data());
            return 0;
        } 
    );
}

caiwei::context::caiwei_llama_batch caiwei::context::VLMLlamaCPPContext::get_inputs(std::string& prompt, llama_context* context, const caiwei::text::CompletionsRequest& request) const {
    prompt = this->chat_template.apply(this->special_token, request, this->special_token.media_marker, this->special_token.media_marker, this->special_token.media_marker);
    caiwei_llama_batch caiwei_batch{};
    if (prompt.empty()) {
        CW_LOG_W("提示词为空");
        caiwei_batch.success = false;
        caiwei_batch.finish_reason = caiwei::text::FINISH_REASON_STOP;
        return caiwei_batch;
    }
    mtmd::bitmaps bitmaps;
    for (const auto& message : request.messages) {
        for (const auto& image : message.image_data) {
            bitmaps.entries.emplace_back(fill_media(image.width, image.height, image.channels, image.data));
        }
        for (const auto& video : message.video_data) {
            bitmaps.entries.emplace_back(fill_media(video));
            mtmd_bitmap_set_mergeable(bitmaps.entries.back().ptr.get(), true);
        }
    }
    if (this->get_mtmd_inputs(caiwei_batch, bitmaps, prompt, context, request)) {
        caiwei_batch.success = true;
        return caiwei_batch;
    } else {
        caiwei_batch.success = false;
        if (caiwei_batch.finish_reason.empty()) {
            caiwei_batch.finish_reason = caiwei::text::FINISH_REASON_STOP;
        }
        return caiwei_batch;
    }
}
