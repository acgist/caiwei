#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/runtime/llamacpp.hpp"

caiwei::context::VLMLlamaCPPContext::VLMLlamaCPPContext(
    std::string path,
    std::string mtmd_path,
    int32_t max_token_length,
    caiwei::text::SpecialToken special_token,
    caiwei::runtime::Runtime* runtime
) :
    VLMContext(runtime),
    LlamaCPPContext(std::move(path), std::move(mtmd_path), max_token_length, std::move(special_token)) {
}

caiwei::context::VLMLlamaCPPContext::~VLMLlamaCPPContext() {
}

bool caiwei::context::VLMLlamaCPPContext::load() {
    const int min_tokens = caiwei::env::get_int("CAIWEI_VLM_IMAGE_MIN_TOKEN");
    const int max_tokens = caiwei::env::get_int("CAIWEI_VLM_IMAGE_MAX_TOKEN");
    this->release();
    if (!(this->load_model() && this->load_mtmd_model(min_tokens, max_tokens) && this->load_template())) {
        this->release();
        return false;
    }
    return true;
}

std::generator<caiwei::text::CompletionResult>
caiwei::context::VLMLlamaCPPContext::run(const caiwei::text::CompletionsRequest& request) {
    return this->generate(request);
}

inline static mtmd_bitmap* fill_media(const caiwei::media::ImageFrame& image) {
    if (!image.check()) {
        CW_LOG_W("图片格式错误: %u, %u, %u", image.width, image.height, image.channels);
        return nullptr;
    }
    // 不用手动缩放使用使用image_min_tokens/image_max_tokens自动适配
    return mtmd_bitmap_init(image.width, image.height, image.data.data());
}

inline static mtmd_bitmap* fill_media(const std::vector<caiwei::media::VideoFrame>& video) {
    if (video.empty()) {
        CW_LOG_W("视频为空");
        return nullptr;
    }
    auto* ptr = const_cast<std::vector<caiwei::media::VideoFrame>*>(&video);
    return mtmd_bitmap_init_lazy(
        nullptr,
        nullptr,
        ptr,
        [](
            size_t chunk_idx, void* user_data, mtmd_bitmap** out_bitmap, char** /* out_text */
        ) {
            const std::vector<caiwei::media::VideoFrame>& video = *(std::vector<caiwei::media::VideoFrame>*) user_data;
            if (chunk_idx >= video.size()) {
                return -1;
            }
            const auto& frame = video[chunk_idx];
            if (!frame.check()) {
                CW_LOG_W("视频格式错误: %u, %u, %u", frame.width, frame.height, frame.channels);
                return -2;
            }
            // 不用手动缩放使用使用image_min_tokens/image_max_tokens自动适配
            *out_bitmap = mtmd_bitmap_init(frame.width, frame.height, frame.data.data());
            return 0;
        }
    );
}

caiwei::context::caiwei_llama_batch
caiwei::context::VLMLlamaCPPContext::get_inputs(std::string& prompt, const caiwei::text::CompletionsRequest& request) const {
    prompt = this->chat_template.apply(
        this->special_token,
        request,
        this->special_token.media_marker,
        this->special_token.media_marker,
        this->special_token.media_marker
    );
    caiwei_llama_batch caiwei_batch {};
    if (prompt.empty()) {
        CW_LOG_W("提示词为空");
        caiwei_batch.fail(caiwei::text::FINISH_REASON_STOP);
        return caiwei_batch;
    }
    [[maybe_unused]] uint64_t image_count = 0;
    [[maybe_unused]] uint64_t video_count = 0;
    [[maybe_unused]] uint64_t frame_count = 0;
    mtmd::bitmaps bitmaps;
    for (const auto& message : request.messages) {
        for (const auto& image : message.image_data) {
            mtmd_bitmap* bitmap = fill_media(image);
            if (bitmap == nullptr) {
                caiwei_batch.fail(caiwei::text::FINISH_REASON_STOP);
                return caiwei_batch;
            }
            bitmaps.entries.emplace_back(bitmap);
            ++image_count;
        }
        for (const auto& video : message.video_data) {
            mtmd_bitmap* bitmap = fill_media(video);
            if (bitmap == nullptr) {
                caiwei_batch.fail(caiwei::text::FINISH_REASON_STOP);
                return caiwei_batch;
            }
            mtmd_bitmap_set_mergeable(bitmap, true);
            bitmaps.entries.emplace_back(bitmap);
            frame_count += video.size();
            ++video_count;
        }
    }
    CW_LOG_D("图片数量: %" PRIu64 " 视频数量: %" PRIu64 " 视频帧数: %" PRIu64, image_count, video_count, frame_count);
    bool success = false;
    try {
        success = this->get_mtmd_inputs(caiwei_batch, bitmaps, prompt, request);
    } catch (const std::exception& e) {
        CW_LOG_E("媒体处理异常: %s", e.what());
    }
    if (success) {
        return caiwei_batch;
    }
    caiwei_batch.fail(caiwei::text::FINISH_REASON_STOP);
    return caiwei_batch;
}
