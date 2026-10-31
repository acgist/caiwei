#include "test.hpp"

#include <thread>
#include <fstream>

[[maybe_unused]]
void test_format() {
    // rtp|sdp|file|http|rtmp|rtsp|device
    auto type = "file";
    // 注意端口必须分开
    // ffmpeg -re -i caiwei.mp4 -c:a pcm_alaw -ar 8000 -ac 1 -vn -f rtp -payload_type  8 -ssrc 1000 rtp://127.0.0.1:44444 > audio.sdp
    // ffmpeg -re -i caiwei.mp4 -c:v libx264                 -an -f rtp -payload_type 96 -ssrc 2000 rtp://127.0.0.1:44446 > video.sdp
//     auto url = R"(v=0
// c=IN IP4 0.0.0.0
// o=- 0 0 IN IP4 0.0.0.0
// s=caiwei rtp capture
// t=0 0
// m=audio 44444 RTP/AVP 8
// a=rtcp:44445
// a=rtpmap:8 PCMA/8000/1
// a=recvonly
// m=video 44446 RTP/AVP 96
// a=rtcp:44447
// a=rtpmap:96 H264/90000
// a=fmtp:96 profile-level-id=42e01f;packetization-mode=1;level-asymmetry-allowed=1
// a=recvonly)";
    auto url  = "./caiwei.mp4";
    // auto url  = "https://static.acgist.com/demo/barrage/video.mp4";
    // auto url  = "rtmp://liteavapp.qcloud.com/live/liteavdemoplayerstreamid";
    // auto url  = "rtsp://admin:admin@127.0.0.1:554/h264/ch1/main/av_stream";
    // auto url = R"(audio=麦克风阵列 (适用于数字麦克风的英特尔® 智音技术):video=Integrated Camera)";
    std::ofstream output("./output.mp4", std::ios_base::binary);
    caiwei::media::MediaFormat media_format([&output](uint32_t length, const uint8_t* data) {
        output.write((const char*) data, length);
        // CW_LOG_D("write %d bytes", length);
        return true;
    });
    caiwei::media::MediaMuxer media_muxer(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::AUDIO_INFO_ENCODE,
        caiwei::media::VIDEO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_ENCODE,
        [&media_format](caiwei::media::MediaType type, AVPacket* packet) {
        if (type == caiwei::media::MediaType::AUDIO) {
            // CW_LOG_D("audio packet: %lld %lld %lld %d", packet->pts, packet->dts, packet->duration, packet->size);
        } else if (type == caiwei::media::MediaType::VIDEO) {
            // CW_LOG_D("video packet: %lld %lld %lld %d", packet->pts, packet->dts, packet->duration, packet->size);
        } else {
            // -
        }
        media_format.send(type, packet);
        return true;
    });
    caiwei::media::MediaDemuxer media_demuxer(type, url);
    media_muxer.open();
    media_format.open(media_muxer.audio_codec_ctx, media_muxer.video_codec_ctx);
    media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        [&media_muxer](caiwei::media::AudioFrame& frame) {
            // CW_LOG_D("audioFrame: %llu %llu %zu %u", frame.msec, frame.frames, frame.data.size(), frame.samples);
            media_muxer.on_audio(frame);
            return true;
        },
        [&media_muxer](caiwei::media::VideoFrame& frame) {
            // CW_LOG_D("videoFrame: %llu %llu %zu %dx%d", frame.msec, frame.frames, frame.data.size(), frame.width, frame.height);
            media_muxer.on_video(frame);
            return true;
        }
    );
    output.close();
}

[[maybe_unused]]
void test_format_fwd_file() {
    auto type = "file";
    auto url  = "./caiwei.mp4";
    caiwei::media::MediaFormat media_format("./output_fwd_file.mp4");
    caiwei::media::MediaDemuxer media_demuxer(type, url);
    media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        [](caiwei::media::AudioFrame& frame) {
            return true;
        },
        [](caiwei::media::VideoFrame& frame) {
            return true;
        },
        &media_format
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
    caiwei::media::MediaDemuxer media_demuxer(type, url);
    media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        [](caiwei::media::AudioFrame& frame) {
            return true;
        },
        [](caiwei::media::VideoFrame& frame) {
            return true;
        },
        &media_format
    );
    output.close();
}

int main() {
    caiwei::test::init_test();
    test_format();
    test_format_fwd_file();
    test_format_fwd_callback();
    caiwei::test::stop_test();
    return 0;
}