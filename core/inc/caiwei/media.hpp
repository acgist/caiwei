/**
 * 媒体工具
 * 
 * 提供音频、图片、视频处理功能
 * 
 * 媒体：采集
 * 音频：解码、重采样
 * 视频：解码、重采样
 * 图片：解码、缩放、画框、范围框选拷贝
 * 
 * OpenCV性能比较
 */
#ifndef CAIWEI_MEDIA_HPP
#define CAIWEI_MEDIA_HPP

#include <string>
#include <thread>
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

extern SwrContext* init_audio_swr(AVCodecContext* ctx, AVFrame* frame);
extern SwsContext* init_video_sws(AVCodecContext* ctx, AVFrame* frame);

extern void sws_free(SwsContext** sws);

/**
 * 媒体帧
 */
class Frame {

public:
    uint32_t             data_length;
    std::vector<uint8_t> data;
public:
    Frame(uint32_t size);

};

// AudioFrame 数据 时间 时长
class AudioFrame : public Frame {

public:
    uint64_t msec;
    uint64_t frames;
public:
    AudioFrame(uint32_t size);

};

// ImageFrame 数据 缩放 画框 框选拷贝
class ImageFrame : public Frame {

public:
    uint32_t width;
    uint32_t height;
public:
    ImageFrame(uint32_t size);

};

// VideoFrame extend ImageFrame 时间 时长
class VideoFrame : public ImageFrame {

public:
    uint64_t msec;
    uint64_t frames;
public:
    VideoFrame(uint32_t size);

};

enum class MediaType { AUDIO, IMAGE, VIDEO };

using AudioCallback = std::function<bool(const AudioFrame&)>;
using ImageCallback = std::function<bool(const ImageFrame&)>;
using VideoCallback = std::function<bool(const VideoFrame&)>;
using PacketCallback = std::function<bool(MediaType, AVPacket*)>;
using FormatCallback = std::function<bool(MediaType, unsigned int, const uint8_t* const)>;

/**
 * 媒体封装器
 */
class MediaMuxer {
private:
    int fps;
    int width;
    int height;
    PacketCallback packet_callback;
    size_t audio_frames = 0; // 音频累计编码帧数
    size_t video_frames = 0; // 视频累计编码帧数
    AVFrame * audio_frame { nullptr }; // 音频帧
    AVFrame * video_frame { nullptr }; // 视频帧
    AVPacket* audio_packet{ nullptr }; // 音频包
    AVPacket* video_packet{ nullptr }; // 视频包
    const AVCodec  * audio_codec    { nullptr }; // 音频编码器
    const AVCodec  * video_codec    { nullptr }; // 视频编码器
public:
    AVCodecContext * audio_codec_ctx{ nullptr }; // 音频编码器上下文
    AVCodecContext * video_codec_ctx{ nullptr }; // 视频编码器上下文
public:
    MediaMuxer(int fps, int width, int height, PacketCallback packet_callback);
    ~MediaMuxer();
public:
    bool open();
    bool stop();
    bool on_audio(const AudioFrame& frame);
    bool on_video(const VideoFrame& frame);
    bool on_audio(                                   const double /* msec */, const double /* frames */, const uint32_t data_length, const uint8_t* data);
    bool on_video(const int width, const int height, const double /* msec */, const double /* frames */, const uint32_t data_length, const uint8_t* data);
private:
    bool open_audio();
    bool open_video();
    bool stop_audio();
    bool stop_video();
};

/**
 * 媒体格式
 */
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

/**
 * 媒体解封装器
 * 
 * TODO 如果视频变化需要reopen
 */
class MediaDemuxer {

private:
    bool        running; // 是否运行
    std::string type   ; // 媒体类型：rtp|file|http|rtmp|rtsp|device
    std::string url    ; // 媒体地址
    std::thread thread ; // 媒体线程
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
