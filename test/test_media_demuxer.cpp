#include "test.hpp"

#include <thread>
#include <fstream>
#include <filesystem>

extern "C" {
#include "libavutil/base64.h"
}

[[maybe_unused]]
static std::string base64_file(const std::string& type, const std::string& path) {
    if (!std::filesystem::exists(path)) {
        CW_LOG_W("文件无效: %s", path.c_str());
        return "";
    }
    std::ifstream stream(path, std::ios::binary);
    std::string data;
    data.resize(std::filesystem::file_size(path));
    stream.read(data.data(), data.size());
    stream.close();
    std::string output;
    const auto len = AV_BASE64_SIZE(data.size());
    output.resize(len);
    char* ret = av_base64_encode(output.data(), len, (const uint8_t*) data.data(), data.size());
    if (ret == nullptr) {
        CW_LOG_W("编码失败: %s", path.c_str());
        return "";
    }
    output.resize(len - 1);
    return type + output;
}

[[maybe_unused]]
static std::string get_url(const char* type) {
    if (std::strcmp(type, "rtp") == 0) {
        // 注意端口必须分开
        // ffmpeg -re -i caiwei.mp4 -c:a pcm_alaw -ar 8000 -ac 1 -vn -f rtp -payload_type  8 -ssrc 1000 rtp://127.0.0.1:44444 > audio.sdp
        // ffmpeg -re -i caiwei.mp4 -c:v libx264                 -an -f rtp -payload_type 96 -ssrc 2000 rtp://127.0.0.1:44446 > video.sdp
        return R"(v=0
c=IN IP4 0.0.0.0
o=- 0 0 IN IP4 0.0.0.0
s=caiwei rtp capture
t=0 0
m=audio 44444 RTP/AVP 8
a=rtcp:44445
a=rtpmap:8 PCMA/8000/1
a=recvonly
m=video 44446 RTP/AVP 96
a=rtcp:44447
a=rtpmap:96 H264/90000
a=fmtp:96 profile-level-id=42e01f;packetization-mode=1;level-asymmetry-allowed=1
a=recvonly)";
    } else if (std::strcmp(type, "data") == 0) {
        return base64_file("data:audio/mpeg;base64,", "./caiwei.mp3");
        // return base64_file("data:image/jpeg;base64,", "./caiwei.jpg");
        // 视频格式: ts/mkv
        // ffmpeg.exe -ss 0:0:0 -to 0:0:10 -i .\caiwei.mp4 -c copy -f mpegts caiwei.ts
        // return base64_file("data:video/ts;base64,", "./caiwei.ts");
        // ffmpeg.exe -ss 0:0:0 -to 0:0:10 -i .\caiwei.mp4 -c copy caiwei.mkv
        // return base64_file("data:video/mkv;base64,", "./caiwei.mkv");
    } else if (std::strcmp(type, "file") == 0) {
        return "./caiwei.mp4";
    } else if (std::strcmp(type, "http") == 0) {
        return "https://static.acgist.com/demo/barrage/video.mp4";
    } else if (std::strcmp(type, "rtmp") == 0) {
        return "rtmp://liteavapp.qcloud.com/live/liteavdemoplayerstreamid";
    } else if (std::strcmp(type, "rtsp") == 0) {
        return "rtsp://admin:admin@127.0.0.1:554/h264/ch1/main/av_stream";
    } else if (std::strcmp(type, "device") == 0) {
        // ffmpeg -list_devices true -f dshow -i dummy
        // ffmpeg -list_options true -f dshow -i video="deviceName"
        // url: audio=deviceName:video=deviceName
        return "audio=麦克风阵列 (适用于数字麦克风的英特尔® 智音技术):video=Integrated Camera";
    } else {
        return "";
    }
}

[[maybe_unused]]
void test_demuxer() {
    // rtp|data|file|http|rtmp|rtsp|device
    auto type = "file";
    auto url  = get_url(type);
    if (url.empty()) {
        CW_LOG_W("不支持的类型: %s", type);
        return;
    }
    caiwei::player::open_player();
    caiwei::media::MediaDemuxer media_demuxer(type, url);
    media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        [](caiwei::media::AudioFrame& frame) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            return caiwei::player::play_audio(frame.data.data(), frame.data.size());
        },
        [](caiwei::media::VideoFrame& frame) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            return caiwei::player::play_video(frame.data.data(), frame.width * 3);
        }
    );
    caiwei::player::stop_player();
}

int main() {
    caiwei::test::init_test();
    test_demuxer();
    caiwei::test::stop_test();
    return 0;
}
