#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/image_tool.hpp"
#include "caiwei/runtime/llamacpp.hpp"

caiwei::context::VLMLlamaCPPContext::VLMLlamaCPPContext(
    std::string path,
    std::string mtmd_path,
    int32_t max_token_length,
    caiwei::text::SpecialToken special_token,
    caiwei::runtime::Runtime* runtime
) : VLMContext(runtime),
    LlamaCPPContext(std::move(path), std::move(mtmd_path), max_token_length, special_token) {
}

caiwei::context::VLMLlamaCPPContext::~VLMLlamaCPPContext() {
}

bool caiwei::context::VLMLlamaCPPContext::load() {
    return this->load_model() && this->load_mtmd_model();
}

std::generator<caiwei::text::CompletionResult> caiwei::context::VLMLlamaCPPContext::run(const caiwei::text::CompletionsRequest& request) {
    return this->generate(request);
}

static mtmd_bitmap* fill_media(int width, int height, int channels, const std::vector<uint8_t>& data) {
    const int w = caiwei::env::get_int("CAIWEI_VLM_WIDTH");
    const int h = caiwei::env::get_int("CAIWEI_VLM_HEIGHT");
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    caiwei::image::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale);
    std::vector<uint8_t> dst(dst_w * dst_h * channels);
    std::vector<uint8_t> pad(    w *     h * channels);
    caiwei::image::resize(data.data(), dst.data(), width, height, dst_w, dst_h);
    caiwei::image::padding(dst.data(), pad.data(), dst_w, dst_h, pad_w, pad_h, w, h);
    return mtmd_bitmap_init(w, h, pad.data());
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
            const auto& data   = frame.data;
            const int width    = frame.width;
            const int height   = frame.height;
            const int channels = frame.channels;
            const int w = caiwei::env::get_int("CAIWEI_VLM_WIDTH");
            const int h = caiwei::env::get_int("CAIWEI_VLM_HEIGHT");
            float scale;
            int dst_w, dst_h, pad_w, pad_h;
            caiwei::image::resize(frame.width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale);
            std::vector<uint8_t> dst(dst_w * dst_h * channels);
            std::vector<uint8_t> pad(    w *     h * channels);
            caiwei::image::resize(data.data(), dst.data(), width, height, dst_w, dst_h);
            caiwei::image::padding(dst.data(), pad.data(), dst_w, dst_h, pad_w, pad_h, w, h);
            *out_bitmap = mtmd_bitmap_init(w, h, pad.data());
            return 0;
        } 
    );
}

caiwei::context::caiwei_llama_batch caiwei::context::VLMLlamaCPPContext::get_inputs(std::string& prompt, llama_context* context, const caiwei::text::CompletionsRequest& request) const {
    mtmd::bitmaps bitmaps;
    for (const auto& message : request.messages) {
        if (message.role != caiwei::text::ROLE_USER) {
            continue;
        }
        if (!message.content.has_value()) {
            continue;
        }
        if (std::holds_alternative<std::string>(message.content.value())) {
            continue;
        }
        const std::vector<caiwei::text::CompletionsRequestMessageContentItem>& items = std::get<std::vector<caiwei::text::CompletionsRequestMessageContentItem>>(message.content.value());
        auto iter = items.begin();
        size_t image_index = 0;
        size_t video_index = 0;
        while (iter != items.end()) {
            if (iter->type == "text") {
                // -
            } else if (iter->type == "image" || iter->type == "image_url") {
                if (image_index < message.image_data.size()) {
                    const auto& image = message.image_data[image_index];
                    bitmaps.entries.emplace_back(fill_media(image.width, image.height, image.channels, image.data));
                    image_index++;
                }
            } else if (iter->type == "video" || iter->type == "video_url") {
                if (video_index < message.video_data.size()) {
                    const auto& video = message.video_data[video_index];
                    bitmaps.entries.emplace_back(fill_media(video));
                    mtmd_bitmap_set_mergeable(bitmaps.entries.back().ptr.get(), true);
                    video_index++;
                }
            } else {
                CW_LOG_W("未知类型: %s", iter->type.value_or("-").c_str());
            }
            iter++;
        }
        if (image_index < message.image_data.size()) {
            for (; image_index < message.image_data.size();) {
                const auto& image = message.image_data[image_index];
                bitmaps.entries.emplace_back(fill_media(image.width, image.height, image.channels, image.data));
                image_index++;
            }
        }
        if (video_index < message.video_data.size()) {
            for (; video_index < message.video_data.size();) {
                const auto& video = message.video_data[video_index];
                bitmaps.entries.emplace_back(fill_media(video));
                mtmd_bitmap_set_mergeable(bitmaps.entries.back().ptr.get(), true);
                video_index++;
            }
        }
    }
    prompt = this->chat_template.apply(this->special_token, request, this->special_token.media_marker, this->special_token.media_marker, this->special_token.media_marker);
    caiwei_llama_batch caiwei_batch{};
    if (this->get_mtmd_inputs(caiwei_batch, bitmaps, prompt, context, request)) {
        caiwei_batch.success = true;
        return caiwei_batch;
    } else {
        CW_LOG_W("图片编码失败");
        caiwei_batch.success = false;
        caiwei_batch.finish_reason = caiwei::text::FINISH_REASON_STOP;
        return caiwei_batch;
    }
}
