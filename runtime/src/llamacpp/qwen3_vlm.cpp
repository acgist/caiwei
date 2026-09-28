#include "caiwei/log.hpp"
#include "caiwei/runtime/llamacpp.hpp"

/**
 * 参考代码
 * deps\llama.cpp\tools\mtmd\mtmd-cli.cpp
 */

caiwei::context::VLMLlamaCPPContext::VLMLlamaCPPContext(std::string path, std::string mtmd_path, std::string media_marker, int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime)
    : LlamaCPPContext(std::move(path), std::move(mtmd_path), std::move(media_marker), max_token_length, special_token),
    VLMContext(runtime) {
}

caiwei::context::VLMLlamaCPPContext::~VLMLlamaCPPContext() {
}

bool caiwei::context::VLMLlamaCPPContext::load() {
    return this->load_model() && this->load_mtmd();
}

std::generator<caiwei::text::Result> caiwei::context::VLMLlamaCPPContext::run(caiwei::text::CompletionsRequest& request) {
    return this->generate_mtmd(request);
}

void caiwei::context::VLMLlamaCPPContext::build_bitmaps(caiwei::text::CompletionsRequest& request, mtmd::bitmaps& bitmaps) {
    for (auto& message : request.messages) {
        if (message.role != caiwei::text::ROLE_USER) {
            continue;
        }
        if (std::holds_alternative<std::string>(message.content.value())) {
            continue;
        }
        std::vector<caiwei::text::CompletionsRequestMessageContentItem>& items = std::get<std::vector<caiwei::text::CompletionsRequestMessageContentItem>>(message.content.value());
        auto iter = items.begin();
        int image_index = 0;
        int video_index = 0;
        std::string content{};
        while (iter != items.end()) {
            if (iter->type == "text") {
                content += iter->text.value_or("");
            } else if (iter->type == "image" || iter->type == "image_url") {
                if (image_index >= message.image_data.size()) {
                    continue;
                }
                content += this->media_marker;
                const auto& image = message.image_data[image_index];
                bitmaps.entries.emplace_back(mtmd_bitmap_init(image.width, image.height, image.data.data()));
                image_index++;
            } else if (iter->type == "video" || iter->type == "video_url") {
                if (video_index >= message.video_data.size()) {
                    continue;
                }
                // TODO 时间戳 <0.5 seconds>
                const auto& video = message.video_data[video_index];
                for (const auto& frame : video) {
                    content += this->media_marker;
                    bitmaps.entries.emplace_back(mtmd_bitmap_init(frame.width, frame.height, frame.data.data()));
                    mtmd_bitmap_set_mergeable(bitmaps.entries.back().ptr.get(), true);
                }
                video_index++;
            } else {
                CW_LOG_W("未知类型: %s", iter->type.value_or("-").c_str());
            }
            iter++;
        }
        message.content = std::move(content);
    }
}
