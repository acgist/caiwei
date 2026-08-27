#include "test.hpp"
#include "caiwei/log.hpp"
#include "caiwei/media.hpp"

#include <thread>
#include <fstream>

extern "C" {

#include "libavcodec/avcodec.h"

}

int main() {
    init_test();
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
    auto url  = "D://download/caiwei.mp4";
    // auto url  = "https://static.acgist.com/demo/barrage/video.mp4";
    // auto url  = "rtmp://liteavapp.qcloud.com/live/liteavdemoplayerstreamid";
    // auto url  = "rtsp://admin:admin@127.0.0.1:554/h264/ch1/main/av_stream";
    // auto url = R"(audio=麦克风阵列 (适用于数字麦克风的英特尔® 智音技术):video=Integrated Camera)";
    std::ofstream output("D://download/output.mp4", std::ios_base::binary);
    caiwei::media::MediaFormat media_format([&output](uint32_t length, const uint8_t* data) {
        output.write((const char*) data, length);
        std::printf("write %d bytes\n", length);
        return true;
    });
    caiwei::media::MediaMuxer media_muxer(
        caiwei::media::AudioInfo(1, 16000, AV_SAMPLE_FMT_S16),
        caiwei::media::VideoInfo(640, 360, AV_PIX_FMT_RGB24),
        caiwei::media::AudioInfo(2, 48000, AV_SAMPLE_FMT_FLTP),
        caiwei::media::VideoInfo(25, 640, 360, AV_PIX_FMT_YUV420P),
        [&media_format](caiwei::media::MediaType type, AVPacket* packet) {
        if (type == caiwei::media::MediaType::AUDIO) {
            std::printf("audio packet: %" PRId64 " %" PRId64 " %" PRId64 " %" PRId32 "\n", packet->pts, packet->dts, packet->duration, packet->size);
        } else if (type == caiwei::media::MediaType::VIDEO) {
            std::printf("video packet: %" PRId64 " %" PRId64 " %" PRId64 " %" PRId32 "\n", packet->pts, packet->dts, packet->duration, packet->size);
        } else {
            // -
        }
        media_format.send(type, packet);
        return true;
    });
    media_muxer.open();
    media_format.open(media_muxer);
    caiwei::media::MediaDemuxer media_demuxer(type, url, [&media_muxer](const caiwei::media::AudioFrame& frame) {
        std::printf("audioFrame: %" PRId64 " %" PRId64 " %" PRId32 " %" PRId32 "\n", frame.msec, frame.frames, frame.data_length, frame.samples);
        media_muxer.on_audio(frame);
        std::fflush(stdout);
        return true;
    }, [&media_muxer](const caiwei::media::VideoFrame& frame) {
        std::printf("videoFrame: %" PRId64 " %" PRId64 " %" PRId32 " %" PRId32 "x%" PRId32 "\n", frame.msec, frame.frames, frame.data_length, frame.width, frame.height);
        media_muxer.on_video(frame);
        std::fflush(stdout);
        // std::this_thread::sleep_for(std::chrono::milliseconds(20));
        return true;
    });
    media_demuxer.open(caiwei::media::AudioInfo(1, 16000, AV_SAMPLE_FMT_S16), caiwei::media::VideoInfo(640, 0, AV_PIX_FMT_RGB24));
    media_demuxer.stop();
    media_format.stop();
    media_muxer.stop();
    output.close();
    stop_test();
    return 0;
}