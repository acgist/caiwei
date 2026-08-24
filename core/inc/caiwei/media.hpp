/**
 * 媒体
 * 
 * TODO OpenCV性能比较
 */
#ifndef CAIWEI_MEDIA_HPP
#define CAIWEI_MEDIA_HPP

#include <string>
#include <vector>
#include <functional>

struct AVCodec;
struct AVFrame;
struct AVPacket;
struct AVStream;
struct SwrContext;
struct SwsContext;
struct AVDictionary;
struct AVCodecContext;
struct AVFormatContext;

namespace caiwei {
namespace media  {

extern void init();
extern void stop();

class Frame {

public:
    uint32_t             data_length;
    std::vector<uint8_t> data;
public:
    Frame(uint32_t size);

};

class AudioFrame : public Frame {

public:
    uint64_t msec;
    uint64_t frames;
    uint32_t samples;
public:
    AudioFrame(uint32_t size);

};

class ImageFrame : public Frame {

public:
    uint32_t width;
    uint32_t height;
public:
    ImageFrame(uint32_t size);
public:
    bool crop  (int x, int y, int width, int height, void* target);
    bool draw  (int x, int y, int width, int height, void* target);
    bool resize(              int width, int height, void* target);

};

class VideoFrame : public ImageFrame {

public:
    uint64_t msec;
    uint64_t frames;
public:
    VideoFrame(uint32_t size);

};

enum class MediaType {
    AUDIO,
    IMAGE,
    VIDEO,
};

using AudioCallback = std::function<bool(const AudioFrame&)>;
using ImageCallback = std::function<bool(const ImageFrame&)>;
using VideoCallback = std::function<bool(const VideoFrame&)>;
using PacketCallback = std::function<bool(MediaType, AVPacket*)>;
using FormatCallback = std::function<bool(MediaType, uint32_t, const uint8_t*)>;

class MediaMuxer {

private:
    int fps;         // 视频帧率
    int width;       // 视频宽度
    int height;      // 视频高度
    int channels;    // 音频通道数
    int bit_rate;    // 音频码率
    int sample_rate; // 音频采样率
    PacketCallback packet_callback; // 编码数据回调
    size_t audio_frames = 0; // 音频累计编码帧数
    size_t video_frames = 0; // 视频累计编码帧数
    AVFrame * audio_frame { nullptr }; // 音频帧
    AVFrame * video_frame { nullptr }; // 视频帧
    AVPacket* audio_packet{ nullptr }; // 音频包
    AVPacket* video_packet{ nullptr }; // 视频包
    const AVCodec  * audio_codec{ nullptr }; // 音频编码器
    const AVCodec  * video_codec{ nullptr }; // 视频编码器
public:
    AVCodecContext * audio_codec_ctx{ nullptr }; // 音频编码器上下文
    AVCodecContext * video_codec_ctx{ nullptr }; // 视频编码器上下文
public:
    MediaMuxer(int fps, int width, int height, int channels, int bit_rate, int sample_rate, PacketCallback packet_callback);
    ~MediaMuxer();
public:
    bool open();
    bool stop();
    bool on_audio(const AudioFrame& frame);
    bool on_video(const VideoFrame& frame);
    bool on_audio(const int nb_samples,              const size_t msec, const size_t frames, const uint32_t data_length, const uint8_t* data);
    bool on_video(const int width, const int height, const size_t msec, const size_t frames, const uint32_t data_length, const uint8_t* data);
private:
    bool open_audio();
    bool open_video();
    bool stop_audio();
    bool stop_video();

};

class MediaFormat {

private:
    bool need_header = true;
    std::vector<uint8_t> buffer;
    AVStream       * audio_stream{ nullptr }; // 音频流
    AVStream       * video_stream{ nullptr }; // 视频流
    AVFormatContext* format_ctx  { nullptr }; // 格式化上下文
public:
    FormatCallback format_callback;
public:
    MediaFormat(FormatCallback format_callback);
    ~MediaFormat();
public:
    bool open(const MediaMuxer& muxer);
    bool stop();
    bool send(AVPacket* packet);
private:
    void send_header ();
    void send_trailer();

};

class MediaDemuxer {

private:
    bool        running; // 是否运行
    std::string type   ; // 媒体类型: rtp|sdp|file|http|rtmp|rtsp|device
    std::string url    ; // 媒体地址
    AudioCallback audio_callback; // 音频回调
    VideoCallback video_callback; // 视频回调
public:
    MediaDemuxer(const std::string& type, const std::string& url, AudioCallback audio_callback, VideoCallback video_callback);
    ~MediaDemuxer();
public:
    bool open();
    bool stop();

};

} // namespace media
} // namespace caiwei
#endif // CAIWEI_MEDIA_HPP
