#include "test.hpp"

#include "nlohmann/json.hpp"

#include "caiwei/image_tool.hpp"

#include <generator>

extern "C" {
#include "libavcodec/avcodec.h"
}

static void read_image(caiwei::media::ImageFrame& frame, const char* path) {
    const int w = 384;
    const int h = 384;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    int width, height, channels;
    auto* data = stbi_load(path, &width, &height, &channels, STBI_default);
    caiwei::image::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale);
    std::vector<uint8_t> dst(dst_w * dst_h * channels);
    caiwei::image::resize(data, dst.data(), width, height, dst_w, dst_h);
    frame.width    = dst_w;
    frame.height   = dst_h;
    frame.channels = channels;
    frame.data = std::move(dst);
    stbi_image_free(data);
}

static void resize_image(caiwei::media::VideoFrame& frame) {
    const int w = 384;
    const int h = 384;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    caiwei::image::resize(frame.width, frame.height, w, h, dst_w, dst_h, pad_w, pad_h, scale);
    std::vector<uint8_t> dst(dst_w * dst_h * frame.channels);
    caiwei::image::resize(frame.data.data(), dst.data(), frame.width, frame.height, dst_w, dst_h);
    frame.width    = dst_w;
    frame.height   = dst_h;
    frame.channels = frame.channels;
    frame.data = std::move(dst);
}

void test_vlm() {
    caiwei::text::CompletionsRequest request;
    request.messages.push_back(caiwei::text::CompletionsRequestMessage {
        .role = "user",
        .content = std::vector<caiwei::text::CompletionsRequestMessageContentItem> {
            // caiwei::text::CompletionsRequestMessageContentItem {
            //     .type = "text",
            //     .text = "这个人物图片还是风景图片"
            // },
            // caiwei::text::CompletionsRequestMessageContentItem {
            //     .type = "image",
            // },
            // caiwei::text::CompletionsRequestMessageContentItem {
            //     .type = "image",
            // }
            caiwei::text::CompletionsRequestMessageContentItem {
                .type = "text",
                .text = "请简单告诉我视频里面发生了什么。"
            },
            caiwei::text::CompletionsRequestMessageContentItem {
                .type = "video",
            }
        }
    });
    // caiwei::media::ImageFrame frame1;
    // caiwei::media::ImageFrame frame2;
    // read_image(frame1, "./acgist.jpg");
    // read_image(frame2, "./caiwei.jpg");
    // request.messages.back().image_data.emplace_back(std::move(frame1));
    // request.messages.back().image_data.emplace_back(std::move(frame2));
    std::vector<caiwei::media::VideoFrame> video_frames;
    int frame_index = 0;
    caiwei::media::MediaDemuxer media_demuxer("file", "caiwei.mp4", [](caiwei::media::AudioFrame& frame) {
        return true;
    }, [&video_frames, &frame_index](caiwei::media::VideoFrame& frame) {
        if (++frame_index % 8 == 0) {
            resize_image(frame);
            video_frames.emplace_back(std::move(frame));
        }
        return video_frames.size() < 8;
    });
    media_demuxer.open(
        caiwei::media::AudioInfo(1, 16000, AV_SAMPLE_FMT_S16),
        caiwei::media::VideoInfo(640, 0, AV_PIX_FMT_RGB24)
    );
    media_demuxer.stop();
    request.messages.back().video_data.emplace_back(std::move(video_frames));
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
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "VLM,QWEN,qwen3-vlm,/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.rknn|/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.weight|/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.embed.bin|/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.tokenizer.gguf|/data/model/Qwen3-VL-2B-Instruct/vlm/Qwen3-VL-2B-Instruct.rknn|/data/model/Qwen3-VL-2B-Instruct/vlm/Qwen3-VL-2B-Instruct.weight");
    #else
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "VLM,QWEN,qwen3-vlm,E:/model/llama.cpp/Qwen3-VL-2B-Instruct-Q8_0.gguf|E:/model/llama.cpp/mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf");
    // caiwei::env::set("CAIWEI_CONTEXT_INFO", "VLM,QWEN,qwen3-vlm,D:/tmp/model/llama.cpp/Qwen3-VL-2B-Instruct-Q4_K_M.gguf|D:/tmp/model/llama.cpp/mmproj-Qwen3-VL-2B-Instruct-Q4_K_M.gguf");
    #endif
    caiwei::test::init_test();
    test_vlm();
    caiwei::test::stop_test();
    return 0;
}
