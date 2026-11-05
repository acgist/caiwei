#include "test.hpp"

#include <thread>
#include <fstream>

[[maybe_unused]]
void test_format() {
    auto type = "file";
    auto url  = "./caiwei.mp4";
    std::ofstream output("./output.mp4", std::ios_base::binary);
    caiwei::media::MediaFormat media_format([&output](uint32_t length, const uint8_t* data) {
        output.write((const char*) data, length);
        return true;
    });
    caiwei::media::MediaMuxer media_muxer(
        caiwei::media::AUDIO_INFO_ENCODE,
        caiwei::media::VIDEO_INFO_ENCODE,
        [&media_format](caiwei::media::MediaType type, AVPacket* packet) {
            return media_format.send(type, packet);
        },
        &media_format
    );
    caiwei::media::MediaDemuxer media_demuxer(type, url, &media_muxer);
    media_demuxer.open(
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
    output.close();
}

[[maybe_unused]]
void test_format_audio() {
    auto type = "file";
    auto url  = "./caiwei.mp3";
    caiwei::media::MediaFormat media_format("./output_fwd_file.mp3");
    caiwei::media::MediaDemuxer media_demuxer(type, url, nullptr, &media_format);
    media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        nullptr,
        nullptr,
        [&media_format](caiwei::media::MediaType type, AVPacket* packet) {
            return media_format.send(type, packet);
        }
    );
}

[[maybe_unused]]
void test_format_fwd_file() {
    auto type = "file";
    auto url  = "./caiwei.mp4";
    caiwei::media::MediaFormat media_format("./output_fwd_file.mp4");
    caiwei::media::MediaDemuxer media_demuxer(type, url, nullptr, &media_format);
    media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        nullptr,
        nullptr,
        [&media_format](caiwei::media::MediaType type, AVPacket* packet) {
            return media_format.send(type, packet);
        }
    );
}

[[maybe_unused]]
void test_format_fwd_callback() {
    auto type = "file";
    auto url  = "./caiwei.mp4";
    std::ofstream output("./output_fwd_callback.mp4", std::ios_base::binary);
    caiwei::media::MediaFormat media_format([&output](uint32_t length, const uint8_t* data) {
        output.write((const char*) data, length);
        return true;
    });
    caiwei::media::MediaDemuxer media_demuxer(type, url, nullptr, &media_format);
    media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        nullptr,
        nullptr,
        [&media_format](caiwei::media::MediaType type, AVPacket* packet) {
            return media_format.send(type, packet);
        }
    );
    output.close();
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
