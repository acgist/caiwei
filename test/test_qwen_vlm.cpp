#include "test.hpp"

#include <thread>
#include <vector>

[[maybe_unused]]
void vlm_run(caiwei::manager::VLMWrapper* ptr, const caiwei::text::CompletionsRequest& request) {
    ALWAYS_ASSERT(ptr != nullptr);
    for (const auto& value : ptr->run(request)) {
        if (!value.finish_reason.empty()) {
            CW_LOG_I("完成: %s", value.finish_reason.c_str());
        }
    }
}

[[maybe_unused]]
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
    frame1.load_from_file("./acgist.jpg");
    frame2.load_from_file("./caiwei.jpg");
    request.messages.back().image_data.emplace_back(std::move(frame1));
    request.messages.back().image_data.emplace_back(std::move(frame2));
    std::vector<caiwei::media::VideoFrame> video_frames;
    size_t frame_index = 0;
    caiwei::media::MediaDemuxer media_demuxer("file", "caiwei.mp4");
    media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        [](caiwei::media::AudioFrame& /* frame */) {
            return true;
        },
        [&video_frames, &frame_index](caiwei::media::VideoFrame& frame) {
            if (++frame_index % 8 == 0) {
                video_frames.emplace_back(std::move(frame));
            }
            return video_frames.size() < 4;
        }
    );
    request.messages.back().video_data.emplace_back(std::move(video_frames));
    auto ptr = caiwei::manager::get_context<caiwei::context::VLMContext>("qwen3-vlm");
    if (!ptr) {
        return;
    }
    vlm_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
    request.max_completion_tokens = 2;
    vlm_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
    request.max_completion_tokens = 1024;
    request.messages.back().video_data.clear();
    request.messages.back().content = std::vector<caiwei::text::CompletionsRequestMessageContentItem> {
        caiwei::text::CompletionsRequestMessageContentItem {
            .type = "text",
            .text = "描述图片内容"
        },
        caiwei::text::CompletionsRequestMessageContentItem {
            .type = "image",
        },
    };
    CW_LOG_I(
        "测试多一个图片: %zu - %zu - %zu",
        request.messages.back().image_data.size(),
        request.messages.back().video_data.size(),
        std::get<std::vector<caiwei::text::CompletionsRequestMessageContentItem>>(request.messages.back().content.value()).size()
    );
    vlm_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
    request.messages.back().image_data.pop_back();
    request.messages.back().content = std::vector<caiwei::text::CompletionsRequestMessageContentItem> {
        caiwei::text::CompletionsRequestMessageContentItem {
            .type = "text",
            .text = "描述图片内容"
        },
        caiwei::text::CompletionsRequestMessageContentItem {
            .type = "image",
        },
        caiwei::text::CompletionsRequestMessageContentItem {
            .type = "image",
        },
    };
    CW_LOG_I(
        "测试少一个图片: %zu - %zu - %zu",
        request.messages.back().image_data.size(),
        request.messages.back().video_data.size(),
        std::get<std::vector<caiwei::text::CompletionsRequestMessageContentItem>>(request.messages.back().content.value()).size()
    );
    vlm_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
    request.messages.clear();
    CW_LOG_I(
        "测试空数据: %zu",
        request.messages.size()
    );
    vlm_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
}

[[maybe_unused]]
void test_vlm_for() {
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
        }
    });
    caiwei::media::ImageFrame frame1;
    caiwei::media::ImageFrame frame2;
    frame1.load_from_file("./acgist.jpg");
    frame2.load_from_file("./caiwei.jpg");
    request.messages.back().image_data.emplace_back(std::move(frame1));
    request.messages.back().image_data.emplace_back(std::move(frame2));
    auto ptr = caiwei::manager::get_context<caiwei::context::VLMContext>("qwen3-vlm");
    if (!ptr) {
        return;
    }
    caiwei::test::caiwei_wait_enter();
    vlm_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
    CAIWEI_FOR_EACH(100)
    vlm_run(ptr.get(), request);
    CAIWEI_FOR_EACH_END
    caiwei::test::caiwei_wait_enter();
    vlm_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
}

[[maybe_unused]]
void test_vlm_thread() {
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
        }
    });
    caiwei::media::ImageFrame frame1;
    caiwei::media::ImageFrame frame2;
    frame1.load_from_file("./acgist.jpg");
    frame2.load_from_file("./caiwei.jpg");
    request.messages.back().image_data.emplace_back(std::move(frame1));
    request.messages.back().image_data.emplace_back(std::move(frame2));
    auto ptr = caiwei::manager::get_context<caiwei::context::VLMContext>("qwen3-vlm");
    if (!ptr) {
        return;
    }
    std::vector<std::jthread> threads;
    for (int i = 0; i < 2; ++i) {
        threads.push_back(std::jthread([&]() {
            CAIWEI_FOR_EACH(100)
            vlm_run(ptr.get(), request);
            CAIWEI_FOR_EACH_END
        }));
    }
    for (auto& thread : threads) {
        thread.join();
    }
    caiwei::test::caiwei_wait_enter();
}

int main() {
    #if ENABLE_CAIWEI_RUNTIME_RKNN3
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "VLM,QWEN,qwen3-vlm,/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.rknn|/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.weight|/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.embed.bin|/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.tokenizer.gguf|/data/model/Qwen3-VL-2B-Instruct/vlm/Qwen3-VL-2B-Instruct.rknn|/data/model/Qwen3-VL-2B-Instruct/vlm/Qwen3-VL-2B-Instruct.weight");
    #elif CAIWEI_OS_UNIX
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "VLM,QWEN,qwen3-vlm,/data/caiwei/deps/llama.cpp/Qwen3-VL-2B-Instruct/Qwen3-VL-2B-Instruct-Q4_K_M.gguf|/data/caiwei/deps/llama.cpp/Qwen3-VL-2B-Instruct/mmproj-Qwen3-VL-2b-Instruct-Q4_K_M.gguf");
    #elif CAIWEI_OS_WIN
    // caiwei::env::set("CAIWEI_CONTEXT_INFO", "VLM,QWEN,qwen3-vlm,E:/model/llama.cpp/Qwen3-VL-2B-Instruct-Q4_K_M.gguf|E:/model/llama.cpp/mmproj-Qwen3-VL-2b-Instruct-Q4_K_M.gguf");
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "VLM,QWEN,qwen3-vlm,D:/tmp/model/llama.cpp/Qwen3-VL-2B-Instruct-Q4_K_M.gguf|D:/tmp/model/llama.cpp/mmproj-Qwen3-VL-2b-Instruct-Q4_K_M.gguf");
    #else
    #endif
    caiwei::test::init_test();
    test_vlm();
    test_vlm_for();
    // test_vlm_thread();
    // test_vlm_thread();
    caiwei::test::stop_test();
    return 0;
}
