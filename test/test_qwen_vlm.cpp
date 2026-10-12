#include "test.hpp"

#include "nlohmann/json.hpp"

#include "caiwei/image_tool.hpp"

#include <generator>

extern "C" {
#include "libavcodec/avcodec.h"
}

static void read_image(caiwei::media::ImageFrame& frame, const char* path) {
    int width, height, channels;
    auto* data = stbi_load(path, &width, &height, &channels, STBI_default);
    std::vector<uint8_t> vector(width * height * channels);
    std::copy(data, data + width * height * channels, vector.begin());
    frame.width    = width;
    frame.height   = height;
    frame.channels = channels;
    frame.data = std::move(vector);
    stbi_image_free(data);
}

void test_vlm() {
    caiwei::text::CompletionsRequest request;
    request.messages.push_back(caiwei::text::CompletionsRequestMessage {
        .role = "user",
        .content = std::vector<caiwei::text::CompletionsRequestMessageContentItem> {
            caiwei::text::CompletionsRequestMessageContentItem {
                .type = "text",
                .text = "描述下面两张图片内容"
            },
            caiwei::text::CompletionsRequestMessageContentItem {
                .type = "image",
            },
            caiwei::text::CompletionsRequestMessageContentItem {
                .type = "image",
            },
            caiwei::text::CompletionsRequestMessageContentItem {
                .type = "text",
                .text = "描述视频内容"
            },
            caiwei::text::CompletionsRequestMessageContentItem {
                .type = "video",
            },
        }
    });
    caiwei::media::ImageFrame frame1;
    caiwei::media::ImageFrame frame2;
    read_image(frame1, "./acgist.jpg");
    read_image(frame2, "./caiwei.jpg");
    request.messages.back().image_data.emplace_back(std::move(frame1));
    request.messages.back().image_data.emplace_back(std::move(frame2));
    std::vector<caiwei::media::VideoFrame> video_frames;
    int frame_index = 0;
    caiwei::media::MediaDemuxer media_demuxer("file", "caiwei.mp4", [](caiwei::media::AudioFrame& frame) {
        return true;
    }, [&video_frames, &frame_index](caiwei::media::VideoFrame& frame) {
        if (++frame_index % 8 == 0) {
            video_frames.emplace_back(std::move(frame));
        }
        return video_frames.size() < 4;
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
    // std::this_thread::sleep_for(std::chrono::minutes(2));
}

int main() {
    #if ENABLE_CAIWEI_RUNTIME_RKNN3
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "VLM,QWEN,qwen3-vlm,/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.rknn|/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.weight|/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.embed.bin|/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.tokenizer.gguf|/data/model/Qwen3-VL-2B-Instruct/vlm/Qwen3-VL-2B-Instruct.rknn|/data/model/Qwen3-VL-2B-Instruct/vlm/Qwen3-VL-2B-Instruct.weight");
    #elif CAIWEI_OS_UNIX
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "VLM,QWEN,qwen3-vlm,/data/caiwei/deps/llama.cpp/Qwen3-VL-2B-Instruct/Qwen3-VL-2B-Instruct-Q4_K_M.gguf|/data/caiwei/deps/llama.cpp/Qwen3-VL-2B-Instruct/mmproj-Qwen3-VL-2b-Instruct-Q4_K_M.gguf");
    #elif CAIWEI_OS_WIN
    // caiwei::env::set("CAIWEI_CONTEXT_INFO", "VLM,QWEN,qwen3-vlm,E:/model/llama.cpp/Qwen3-VL-2B-Instruct-Q8_0.gguf|E:/model/llama.cpp/mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf");
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "VLM,QWEN,qwen3-vlm,D:/tmp/model/llama.cpp/Qwen3-VL-2B-Instruct-Q4_K_M.gguf|D:/tmp/model/llama.cpp/mmproj-Qwen3-VL-2B-Instruct-Q4_K_M.gguf");
    #else
    #endif
    caiwei::test::init_test();
    test_vlm();
    caiwei::test::stop_test();
    return 0;
}
