#include "test.hpp"

#include "nlohmann/json.hpp"

#include "caiwei/image_tool.hpp"

#include <generator>

extern "C" {
#include "libavcodec/avcodec.h"
}

void test_vlm() {
    caiwei::media::ImageFrame image_frame;
    {
        const int w = 384;
        const int h = 384;
        float scale;
        int dst_w, dst_h, pad_w, pad_h;
        int width, height, channels;
        auto* data = stbi_load("./caiwei.jpg", &width, &height, &channels, STBI_default);
        caiwei::image::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale);
        std::vector<uint8_t> dst(dst_w * dst_h * channels);
        caiwei::image::resize(data, dst.data(), width, height, dst_w, dst_h);
        image_frame.width    = dst_w;
        image_frame.height   = dst_h;
        image_frame.channels = channels;
        image_frame.data = std::move(dst);
        stbi_image_free(data);
    }
    caiwei::text::CompletionsRequest request;
    request.messages.push_back(caiwei::text::CompletionsRequestMessage {
        .role = "user",
        .content = std::vector<caiwei::text::CompletionsRequestMessageContentItem> {
            caiwei::text::CompletionsRequestMessageContentItem {
                .type = "text",
                .text = "这个人物图片还是风景图片"
            },
            caiwei::text::CompletionsRequestMessageContentItem {
                .type = "image",
            }
        }
    });
    request.messages.back().image_data.push_back(std::move(image_frame));
    auto ptr = caiwei::manager::get_context<caiwei::context::VLMContext, caiwei::text::CompletionsRequest, std::generator<caiwei::text::Result>>("qwen3-vlm");
    if (!ptr) {
        return;
    }
    // CAIWEI_FOR_EACH(100)
    for (const auto& value : ptr->run(request)) {
        // -
    }
    // CAIWEI_FOR_EACH_END
}

int main() {
    #if ENABLE_CAIWEI_RUNTIME_RKNN3
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "VLM,QWEN,qwen3-vlm,/data/model/Qwen3-4B/Qwen3-4B.rknn|/data/model/Qwen3-4B/Qwen3-4B.weight|/data/model/Qwen3-4B/Qwen3-4B.embed.bin|/data/model/Qwen3-4B/Qwen3-4B.tokenizer.gguf");
    #else
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "VLM,QWEN,qwen3-vlm,D:/tmp/model/llama.cpp/Qwen3-VL-2B-Instruct-Q4_K_M.gguf|D:/tmp/model/llama.cpp/mmproj-Qwen3-VL-2B-Instruct-Q4_K_M.gguf");
    #endif
    caiwei::test::init_test();
    test_vlm();
    caiwei::test::stop_test();
    return 0;
}
