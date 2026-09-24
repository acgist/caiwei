#include "test.hpp"

#include "nlohmann/json.hpp"

#include <generator>

extern "C" {
#include "libavcodec/avcodec.h"
}

void test_asr() {
    caiwei::media::AudioFrame audio_frame;
    caiwei::media::MediaDemuxer media_demuxer("file", "caiwei.mp3", [&audio_frame](caiwei::media::AudioFrame& frame) {
        audio_frame.data.insert(audio_frame.data.end(), frame.data.begin(), frame.data.end());
        return true;
    }, [](caiwei::media::VideoFrame& frame) {
        return true;
    });
    media_demuxer.open(
        caiwei::media::AudioInfo(1, 16000, AV_SAMPLE_FMT_S16),
        caiwei::media::VideoInfo(640, 0, AV_PIX_FMT_RGB24)
    );
    media_demuxer.stop();
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
    auto ptr = caiwei::manager::get_context<caiwei::context::ASRContext, caiwei::text::CompletionsRequest, std::generator<caiwei::text::Result>>("qwen3-asr");
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
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "ASR,QWEN,qwen3-asr,/data/model/Qwen3-ASR-0.6B/llm/Qwen3-ASR-0.6B.rknn|/data/model/Qwen3-ASR-0.6B/llm/Qwen3-ASR-0.6B.weight|/data/model/Qwen3-ASR-0.6B/llm/Qwen3-ASR-0.6B.embed.bin|/data/model/Qwen3-ASR-0.6B/llm/Qwen3-ASR-0.6B.tokenizer.gguf|/data/model/Qwen3-ASR-0.6B/asr/Qwen3-ASR-0.6B.rknn|/data/model/Qwen3-ASR-0.6B/asr/Qwen3-ASR-0.6B.weight");
    #else
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "ASR,QWEN,qwen3-asr,D:/tmp/model/llama.cpp/Qwen3-ASR-0.6B-Q4_K_M.gguf|D:/tmp/model/llama.cpp/mmproj-Qwen3-ASR-0.6b-Q4_K_M.gguf");
    #endif
    caiwei::test::init_test();
    test_asr();
    caiwei::test::stop_test();
    return 0;
}
