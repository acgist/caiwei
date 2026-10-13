#include "caiwei/log.hpp"
#include "caiwei/image_tool.hpp"
#include "caiwei/runtime/llamacpp.hpp"

caiwei::context::VLMLlamaCPPContext::VLMLlamaCPPContext(std::string path, std::string mtmd_path, int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime)
    : LlamaCPPContext(std::move(path), std::move(mtmd_path), max_token_length, special_token),
    VLMContext(runtime) {
}

caiwei::context::VLMLlamaCPPContext::~VLMLlamaCPPContext() {
}

bool caiwei::context::VLMLlamaCPPContext::load() {
    return this->load_model() && this->load_mtmd();
}

std::generator<caiwei::text::Result> caiwei::context::VLMLlamaCPPContext::run(const caiwei::text::CompletionsRequest& request) {
    return this->generate_mtmd(request);
}

static mtmd_bitmap* fill_media(int width, int height, int channels, const std::vector<uint8_t>& data) {
    const int w = 384;
    const int h = 384;
    float scale;
    // TODO
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
            void * user_data,
            mtmd_bitmap ** out_bitmap,
            char ** out_text
        ) {
            std::vector<caiwei::media::VideoFrame>& video = *(std::vector<caiwei::media::VideoFrame>*)user_data;
            if (chunk_idx >= video.size()) {
                return -1;
            }
            auto& frame = video[chunk_idx];
            int width = frame.width;
            int height = frame.height;
            int channels = frame.channels;
            const std::vector<uint8_t>& data = frame.data;
            const int w = 384;
            const int h = 384;
            float scale;
            // TODO
            int dst_w, dst_h, pad_w, pad_h;
            caiwei::image::resize(frame.width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale);
            std::vector<uint8_t> dst(dst_w * dst_h * channels);
            std::vector<uint8_t> pad(    w *     h * channels);
            caiwei::image::resize(data.data(), dst.data(), width, height, dst_w, dst_h);
            caiwei::image::padding(dst.data(), pad.data(), dst_w, dst_h, pad_w, pad_h, w, h);
            *out_bitmap = mtmd_bitmap_init(w, h, pad.data());
            // TODO 判断内存是否释放
            return 0;
       } 
    );
}

bool caiwei::context::VLMLlamaCPPContext::build_mtmd(std::string& prompt, const caiwei::text::CompletionsRequest& request, mtmd::bitmaps& bitmaps) {
    int image_count = 0;
    int video_count = 0;
    for (const auto& message : request.messages) {
        if (message.role != caiwei::text::ROLE_USER) {
            continue;
        }
        if (std::holds_alternative<std::string>(message.content.value())) {
            continue;
        }
        const std::vector<caiwei::text::CompletionsRequestMessageContentItem>& items = std::get<std::vector<caiwei::text::CompletionsRequestMessageContentItem>>(message.content.value());
        auto iter = items.begin();
        int image_index = 0;
        int video_index = 0;
        while (iter != items.end()) {
            if (iter->type == "text") {
                // -
            } else if (iter->type == "image" || iter->type == "image_url") {
                if (image_index < message.image_data.size()) {
                    const auto& image = message.image_data[image_index];
                    bitmaps.entries.emplace_back(fill_media(image.width, image.height, image.channels, image.data));
                    image_index++;
                    image_count++;
                }
            } else if (iter->type == "video" || iter->type == "video_url") {
                if (video_index < message.video_data.size()) {
                    const auto& video = message.video_data[video_index];
                    bitmaps.entries.emplace_back(fill_media(video));
                    mtmd_bitmap_set_mergeable(bitmaps.entries.back().ptr.get(), true);
                    video_index++;
                    video_count++;
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
                image_count++;
            }
        }
        if (video_index < message.video_data.size()) {
            for (; video_index < message.video_data.size();) {
                const auto& video = message.video_data[video_index];
                bitmaps.entries.emplace_back(fill_media(video));
                mtmd_bitmap_set_mergeable(bitmaps.entries.back().ptr.get(), true);
                video_index++;
                video_count++;
            }
        }
    }
    prompt = this->chat_template.apply(this->special_token, request, this->special_token.media_marker, this->special_token.media_marker, this->special_token.media_marker);
    return true;
}
