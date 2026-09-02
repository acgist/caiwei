/**
 * 媒体
 */
#ifndef CAIWEI_MEDIA_HPP
#define CAIWEI_MEDIA_HPP

#include <string>
#include <functional>

#include "caiwei/media_data.hpp"

struct AVCodec;
struct AVFrame;
struct AVPacket;
struct AVStream;
struct SwrContext;
struct SwsContext;
struct AVAudioFifo;
struct AVDictionary;
struct AVCodecContext;
struct AVFormatContext;

namespace caiwei {
namespace media  {

void init();
void stop();

struct AudioInfo {
    int channels;         // 通道数
    int sample_rate;      // 采样率
    int format;           // 格式
    int bytes_per_sample; // 每个采样字节数

    AudioInfo() = default;
    AudioInfo(int channels, int sample_rate, int format);
};

struct VideoInfo {
    int fps;    // 帧率
    int width;  // 宽度
    int height; // 高度
    int format; // 格式

    VideoInfo() = default;
    VideoInfo(         int width, int height, int format);
    VideoInfo(int fps, int width, int height, int format);
};

using AudioCallback = std::function<bool(const AudioFrame&)>;
using ImageCallback = std::function<bool(const ImageFrame&)>;
using VideoCallback = std::function<bool(const VideoFrame&)>;
using PacketCallback = std::function<bool(MediaType, AVPacket*)>;
using FormatCallback = std::function<bool(uint32_t, const uint8_t*)>;

class MediaMuxer {
private:
    PacketCallback packet_callback; // 编码数据回调
    size_t audio_pts = 0; // 音频累计编码采样点数
    size_t video_pts = 0; // 视频累计编码采样点数
    AVFrame * audio_frame { nullptr }; // 音频帧
    AVFrame * video_frame { nullptr }; // 视频帧
    AVPacket* audio_packet{ nullptr }; // 音频包
    AVPacket* video_packet{ nullptr }; // 视频包
    SwrContext* swr_ctx{ nullptr }; // 音频重采样上下文
    SwsContext* sws_ctx{ nullptr }; // 视频重采样上下文
    const AVCodec * audio_codec{ nullptr }; // 音频编码器
    const AVCodec * video_codec{ nullptr }; // 视频编码器
    uint8_t    * audio_ch_buffer[8] { nullptr }; // 音频通道缓存
    AVAudioFifo* audio_fifo         { nullptr }; // 音频编码缓存
public:
    AudioInfo in_audio_info;  // 输入音频
    VideoInfo in_video_info;  // 输入视频
    AudioInfo out_audio_info; // 输出音频
    VideoInfo out_video_info; // 输出视频
    AVCodecContext* audio_codec_ctx{ nullptr }; // 音频编码器上下文
    AVCodecContext* video_codec_ctx{ nullptr }; // 视频编码器上下文
public:
    MediaMuxer(AudioInfo in_audio_info, VideoInfo in_video_info, AudioInfo out_audio_info, VideoInfo out_video_info, PacketCallback packet_callback);
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
    AudioInfo audio_info; // 音频
    VideoInfo video_info; // 视频
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
    bool send(MediaType, AVPacket* packet);
private:
    void send_header ();
    void send_trailer();
};

class MediaDemuxer {
private:
    volatile bool running; // 是否运行
    std::string type; // 媒体类型: rtp|sdp|file|http|rtmp|rtsp|device
    std::string url ; // 媒体地址
    AudioCallback audio_callback; // 音频回调
    VideoCallback video_callback; // 视频回调
public:
    MediaDemuxer(const std::string& type, const std::string& url, AudioCallback audio_callback, VideoCallback video_callback);
    ~MediaDemuxer();
public:
    bool open(AudioInfo audio_info, VideoInfo video_info);
    bool stop();
};

} // namespace media
} // namespace caiwei

#endif // CAIWEI_MEDIA_HPP
