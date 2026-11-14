#include "test.hpp"

#include <thread>
#include <vector>

[[maybe_unused]]
void asr_run(caiwei::manager::ASRWrapper* ptr, const caiwei::text::CompletionsRequest& request) {
    ALWAYS_ASSERT(ptr != nullptr);
    for (const auto& value : ptr->run(request)) {
        if (!value.finish_reason.empty()) {
            CW_LOG_I("完成: %s", value.finish_reason.c_str());
        }
    }
}

[[maybe_unused]]
void test_asr() {
    caiwei::media::AudioFrame audio_frame;
    caiwei::media::MediaDemuxer media_demuxer("file", "caiwei.mp3");
    // caiwei::media::MediaDemuxer media_demuxer("file", "caiwei_noise.mp3");
    // caiwei::media::MediaDemuxer media_demuxer("file", "caiwei_small.mp3");
    media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        [&audio_frame](caiwei::media::AudioFrame& frame) {
            audio_frame.data.insert(audio_frame.data.end(), frame.data.begin(), frame.data.end());
            return true;
        },
        [](caiwei::media::VideoFrame& /* frame */) {
            return true;
        }
    );
    caiwei::text::CompletionsRequest request;
    request.messages.push_back(caiwei::text::CompletionsRequestMessage {
        .role = "user",
        .content = std::vector<caiwei::text::CompletionsRequestMessageContentItem> {
            caiwei::text::CompletionsRequestMessageContentItem {
                .type = "audio",
            }
        }
    });
    request.messages.back().audio_data.push_back(audio_frame);
    auto ptr = caiwei::manager::get_context<caiwei::context::ASRContext>("qwen3-asr");
    if (!ptr) {
        return;
    }
    asr_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
    request.max_completion_tokens = 2;
    asr_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
    request.max_completion_tokens = 1024;
    request.messages.back().audio_data.push_back(audio_frame);
    CW_LOG_I(
        "测试多一个音频: %zu - %zu",
        request.messages.back().audio_data.size(),
        std::get<std::vector<caiwei::text::CompletionsRequestMessageContentItem>>(request.messages.back().content.value()).size()
    );
    asr_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
    request.messages.back().audio_data.pop_back();
    request.messages.back().content = std::vector<caiwei::text::CompletionsRequestMessageContentItem> {
        caiwei::text::CompletionsRequestMessageContentItem {
            .type = "audio",
        },
        caiwei::text::CompletionsRequestMessageContentItem {
            .type = "audio",
        }
    };
    CW_LOG_I(
        "测试少一个音频: %zu - %zu",
        request.messages.back().audio_data.size(),
        std::get<std::vector<caiwei::text::CompletionsRequestMessageContentItem>>(request.messages.back().content.value()).size()
    );
    asr_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
    request.messages.clear();
    CW_LOG_I(
        "测试空数据: %zu",
        request.messages.size()
    );
    asr_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
}

[[maybe_unused]]
void test_asr_for() {
    caiwei::media::AudioFrame audio_frame;
    caiwei::media::MediaDemuxer media_demuxer("file", "caiwei.mp3");
    media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        [&audio_frame](caiwei::media::AudioFrame& frame) {
            audio_frame.data.insert(audio_frame.data.end(), frame.data.begin(), frame.data.end());
            return true;
        },
        [](caiwei::media::VideoFrame& /* frame */) {
            return true;
        }
    );
    caiwei::text::CompletionsRequest request;
    request.messages.push_back(caiwei::text::CompletionsRequestMessage {
        .role = "user",
        .content = std::vector<caiwei::text::CompletionsRequestMessageContentItem> {
            caiwei::text::CompletionsRequestMessageContentItem {
                .type = "audio",
            }
        }
    });
    request.messages.back().audio_data.push_back(std::move(audio_frame));
    auto ptr = caiwei::manager::get_context<caiwei::context::ASRContext>("qwen3-asr");
    if (!ptr) {
        return;
    }
    caiwei::test::caiwei_wait_enter();
    asr_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
    CAIWEI_FOR_EACH(100)
    asr_run(ptr.get(), request);
    CAIWEI_FOR_EACH_END
    caiwei::test::caiwei_wait_enter();
    asr_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
}

[[maybe_unused]]
void test_asr_thread() {
    caiwei::media::AudioFrame audio_frame;
    caiwei::media::MediaDemuxer media_demuxer("file", "caiwei.mp3");
    media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        [&audio_frame](caiwei::media::AudioFrame& frame) {
            audio_frame.data.insert(audio_frame.data.end(), frame.data.begin(), frame.data.end());
            return true;
        },
        [](caiwei::media::VideoFrame& /* frame */) {
            return true;
        }
    );
    caiwei::text::CompletionsRequest request;
    request.messages.push_back(caiwei::text::CompletionsRequestMessage {
        .role = "user",
        .content = std::vector<caiwei::text::CompletionsRequestMessageContentItem> {
            caiwei::text::CompletionsRequestMessageContentItem {
                .type = "audio",
            }
        }
    });
    request.messages.back().audio_data.push_back(std::move(audio_frame));
    auto ptr = caiwei::manager::get_context<caiwei::context::ASRContext>("qwen3-asr");
    if (!ptr) {
        return;
    }
    std::vector<std::jthread> threads;
    for (int i = 0; i < 2; ++i) {
        threads.push_back(std::jthread([&]() {
            CAIWEI_FOR_EACH(100)
            asr_run(ptr.get(), request);
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
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "ASR,QWEN,qwen3-asr,/data/model/Qwen3-ASR-0.6B/llm/Qwen3-ASR-0.6B.rknn|/data/model/Qwen3-ASR-0.6B/llm/Qwen3-ASR-0.6B.weight|/data/model/Qwen3-ASR-0.6B/llm/Qwen3-ASR-0.6B.embed.bin|/data/model/Qwen3-ASR-0.6B/llm/Qwen3-ASR-0.6B.tokenizer.gguf|/data/model/Qwen3-ASR-0.6B/asr/Qwen3-ASR-0.6B.rknn|/data/model/Qwen3-ASR-0.6B/asr/Qwen3-ASR-0.6B.weight");
    #elif CAIWEI_OS_UNIX
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "ASR,QWEN,qwen3-asr,/data/caiwei/deps/llama.cpp/Qwen3-ASR-0.6B/Qwen3-ASR-0.6B-Q4_K_M.gguf|/data/caiwei/deps/llama.cpp/Qwen3-ASR-0.6B/mmproj-Qwen3-ASR-0.6b-Q4_K_M.gguf");
    #elif CAIWEI_OS_WIN
    // caiwei::env::set("CAIWEI_CONTEXT_INFO", "ASR,QWEN,qwen3-asr,E:/model/llama.cpp/Qwen3-ASR-0.6B-Q4_K_M.gguf|E:/model/llama.cpp/mmproj-Qwen3-ASR-0.6b-Q4_K_M.gguf");
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "ASR,QWEN,qwen3-asr,D:/tmp/model/llama.cpp/Qwen3-ASR-0.6B-Q4_K_M.gguf|D:/tmp/model/llama.cpp/mmproj-Qwen3-ASR-0.6b-Q4_K_M.gguf");
    #else
    #endif
    caiwei::test::init_test();
    test_asr();
    test_asr_for();
    // test_asr_thread();
    // test_asr_thread();
    caiwei::test::stop_test();
    return 0;
}
