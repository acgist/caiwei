#include "test.hpp"

#include <fstream>

static const auto type = "file";
static const auto audio_url = "./caiwei.mp3";
static const auto video_url = "./caiwei.mp4";

[[maybe_unused]]
void test_format() {
    std::ofstream output("./output.mp4", std::ios_base::binary);
    CAIWEI_ASSERT(output.is_open());
    caiwei::media::MediaFormat media_format([&output](uint32_t length, const uint8_t* data) {
        output.write((const char*) data, length);
        return output.good();
    });
    caiwei::media::MediaMuxer media_muxer(
        caiwei::media::AUDIO_INFO_ENCODE,
        caiwei::media::VIDEO_INFO_ENCODE,
        [&media_format](caiwei::media::MediaType type, AVPacket* packet) {
            return media_format.send(type, packet);
        },
        &media_format
    );
    caiwei::media::MediaDemuxer media_demuxer(type, video_url, &media_muxer);
    bool success = media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        [&media_muxer](caiwei::media::AudioFrame& frame) {
            media_muxer.on_audio(frame);
            return true;
        },
        [&media_muxer](caiwei::media::VideoFrame& frame) {
            media_muxer.on_video(frame);
            return true;
        }
    );
    CAIWEI_ASSERT(success);
    media_muxer.stop();
    media_format.stop();
    output.close();
    CAIWEI_ASSERT(output.good());
}

[[maybe_unused]]
void test_format_audio() {
    caiwei::media::MediaFormat media_format("./output_fwd_file.mp3");
    caiwei::media::MediaDemuxer media_demuxer(type, audio_url, nullptr, &media_format);
    bool success = media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        nullptr,
        nullptr,
        [&media_format](caiwei::media::MediaType type, AVPacket* packet) {
            return media_format.send(type, packet);
        }
    );
    CAIWEI_ASSERT(success);
}

[[maybe_unused]]
void test_format_fwd_file() {
    caiwei::media::MediaFormat media_format("./output_fwd_file.mp4");
    caiwei::media::MediaDemuxer media_demuxer(type, video_url, nullptr, &media_format);
    bool success = media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        nullptr,
        nullptr,
        [&media_format](caiwei::media::MediaType type, AVPacket* packet) {
            return media_format.send(type, packet);
        }
    );
    CAIWEI_ASSERT(success);
}

[[maybe_unused]]
void test_format_fwd_callback() {
    std::ofstream output("./output_fwd_callback.mp4", std::ios_base::binary);
    CAIWEI_ASSERT(output.is_open());
    caiwei::media::MediaFormat media_format([&output](uint32_t length, const uint8_t* data) {
        output.write((const char*) data, length);
        return output.good();
    });
    caiwei::media::MediaDemuxer media_demuxer(type, video_url, nullptr, &media_format);
    bool success = media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        nullptr,
        nullptr,
        [&media_format](caiwei::media::MediaType type, AVPacket* packet) {
            return media_format.send(type, packet);
        }
    );
    CAIWEI_ASSERT(success);
    media_format.stop();
    output.close();
    CAIWEI_ASSERT(output.good());
}

int main() {
    caiwei::test::init_test();
    test_format();
    test_format_audio();
    test_format_fwd_file();
    test_format_fwd_callback();
    caiwei::test::stop_test();
    return 0;
}
