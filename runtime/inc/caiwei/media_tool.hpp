#ifndef CAIWEI_RUNTIME_MEDIA_TOOL_HPP
#define CAIWEI_RUNTIME_MEDIA_TOOL_HPP

#include <cmath>
#include <atomic>
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <functional>

#include "caiwei/log.hpp"
#include "caiwei/media_data.hpp"

struct AVCodec;
struct AVFrame;
struct AVPacket;
struct AVStream;
struct AVRational;
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
    int channels         = 0;
    int sample_rate      = 0;
    int format           = 0;
    int bytes_per_sample = 0;

    AudioInfo() = default;
    AudioInfo(int channels, int sample_rate, int format);
};

struct VideoInfo {
    int fps    = 0;
    int width  = 0;
    int height = 0;
    int format = 0;

    VideoInfo() = default;
    VideoInfo(         int width, int height, int format);
    VideoInfo(int fps, int width, int height, int format);
};

inline const AudioInfo AUDIO_INFO_DECODE{ 1, 16000, 1 }; // 1 = AV_SAMPLE_FMT_S16
inline const AudioInfo AUDIO_INFO_ENCODE{ 2, 48000, 8 }; // 8 = AV_SAMPLE_FMT_FLTP

inline const VideoInfo VIDEO_INFO_DECODE{     640,   0, 2 }; // 2 = AV_PIX_FMT_RGB24
inline const VideoInfo VIDEO_INFO_ENCODE{ 24, 640, 360, 0 }; // 0 = AV_PIX_FMT_YUV420P

using AudioCallback = std::function<bool(AudioFrame&)>;
using ImageCallback = std::function<bool(ImageFrame&)>;
using VideoCallback = std::function<bool(VideoFrame&)>;
using PacketCallback = std::function<bool(MediaType, AVPacket*)>;
using FormatCallback = std::function<bool(uint32_t, const uint8_t*)>;

class MediaMuxer;

class MediaFormat {
private:
    bool write_header = false;
    std::string filename;
    uint8_t* buffer{ nullptr };
    AVStream* audio_stream{ nullptr }; // 音频流
    AVStream* video_stream{ nullptr }; // 视频流
    AVRational* audio_time_base{ nullptr }; // 原始音频时间基准
    AVRational* video_time_base{ nullptr }; // 原始视频时间基准
    AVFormatContext* format_ctx{ nullptr }; // 格式化上下文
public:
    bool callable = true;
    FormatCallback format_callback;
public:
    MediaFormat(std::string filename);
    MediaFormat(FormatCallback format_callback);
    ~MediaFormat();
    MediaFormat(const MediaFormat&) = delete;
    MediaFormat& operator=(const MediaFormat&) = delete;
    MediaFormat(MediaFormat&&) = delete;
    MediaFormat& operator=(MediaFormat&&) = delete;
public:
    bool open(
        const AVCodecContext* audio_codec_ctx,
        const AVCodecContext* video_codec_ctx,
        const AVRational    * audio_time_base = nullptr,
        const AVRational    * video_time_base = nullptr
    );
    bool stop();
    bool send(MediaType, AVPacket* packet);
private:
    bool send_header ();
    bool send_trailer();
    bool mode_file();
    bool mode_callback();
};

class MediaMuxer {
private:
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
    int audio_buffer_capacity = 0; // 音频通道缓存容量
    uint8_t    * audio_buffer[8] { nullptr }; // 音频通道缓存
    AVAudioFifo* audio_fifo      { nullptr }; // 音频编码缓存
    AudioInfo in_audio_info;  // 输入音频
    AudioInfo out_audio_info; // 输出音频
    VideoInfo in_video_info;  // 输入视频
    VideoInfo out_video_info; // 输出视频
    PacketCallback packet_callback; // 编码数据回调
    MediaFormat* media_format{ nullptr };
    AVCodecContext* audio_codec_ctx{ nullptr }; // 音频编码器上下文
    AVCodecContext* video_codec_ctx{ nullptr }; // 视频编码器上下文
public:
    MediaMuxer(
        AudioInfo out_audio_info,
        VideoInfo out_video_info,
        PacketCallback packet_callback,
        MediaFormat* media_format = nullptr
    );
    ~MediaMuxer();
    MediaMuxer(const MediaMuxer&) = delete;
    MediaMuxer& operator=(const MediaMuxer&) = delete;
    MediaMuxer(MediaMuxer&&) = delete;
    MediaMuxer& operator=(MediaMuxer&&) = delete;
public:
    bool open(AudioInfo in_audio_info, VideoInfo in_video_info);
    bool stop();
    bool on_audio(const AudioFrame& frame);
    bool on_video(const VideoFrame& frame);
    bool on_audio(const int nb_samples,                                  const size_t frames, const uint32_t data_length, const uint8_t* data);
    bool on_video(const int width, const int height, const int channels, const size_t frames, const uint32_t data_length, const uint8_t* data);
private:
    bool open_audio();
    bool open_video();
    bool stop_audio();
    bool stop_video();
    bool build_audio_buffer(const int nb_samples);
    void close_audio_buffer();
};

class MediaDemuxer {
private:
    std::atomic<bool> running { false }; // 是否运行
    std::string type; // 媒体类型: rtp|data|file|http|rtmp|rtsp|device
    std::string url ; // 媒体地址
    MediaMuxer * media_muxer { nullptr };
    MediaFormat* media_format{ nullptr };
public:
    MediaDemuxer(const std::string& type, const std::string& url, MediaMuxer* media_muxer = nullptr, MediaFormat* media_format = nullptr);
    ~MediaDemuxer();
    MediaDemuxer(const MediaDemuxer&) = delete;
    MediaDemuxer& operator=(const MediaDemuxer&) = delete;
    MediaDemuxer(MediaDemuxer&&) = delete;
    MediaDemuxer& operator=(MediaDemuxer&&) = delete;
public:
    bool open(
        AudioInfo audio_info,
        VideoInfo video_info,
        AudioCallback audio_callback,
        VideoCallback video_callback,
        PacketCallback packet_callback = nullptr
    );
    bool stop();
};

bool load_mel_filters(
    const int n_mels,
    const int n_fft,
    std::vector<float>& mel_filters
);

bool log_mel_spectrogram(
    const float* audio,
    int audio_length,
    const float* mel_filters,
    int n_fft,
    int hop_length,
    int n_mels,
    int max_audio_length,
    std::vector<float>& mel_spec
);


std::string to_json(const Cls& cls);
std::string to_json(const Box& box);
std::string to_json(const std::vector<Cls>& cls);
std::string to_json(const std::vector<Box>& box);

inline Box to_box(float box_x, float box_y, float box_w, float box_h, int class_id, float score) {
    return Box(
        std::clamp(box_x - box_w / 2.0F, 0.0F, 1.0F),
        std::clamp(box_y - box_h / 2.0F, 0.0F, 1.0F),
        std::clamp(box_x + box_w / 2.0F, 0.0F, 1.0F),
        std::clamp(box_y + box_h / 2.0F, 0.0F, 1.0F),
        class_id,
        score
    );
}

inline Box to_box_end2end(float box_x1, float box_y1, float box_x2, float box_y2, int class_id, float score) {
    return Box(
        std::clamp(box_x1, 0.0F, 1.0F),
        std::clamp(box_y1, 0.0F, 1.0F),
        std::clamp(box_x2, 0.0F, 1.0F),
        std::clamp(box_y2, 0.0F, 1.0F),
        class_id,
        score
    );
}

inline bool resize(
    const int src_w,
    const int src_h,
    const int w,
    const int h,
    int& dst_w,
    int& dst_h,
    int& pad_w,
    int& pad_h,
    float& scale
) {
    if (src_w <= 0 || src_h <= 0 || w <= 0 || h <= 0) {
        CW_LOG_W("图片尺寸错误: %dx%d -> %dx%d", src_w, src_h, w, h);
        return false;
    }
    scale = std::min(static_cast<float>(w) / src_w, static_cast<float>(h) / src_h);
    dst_w = static_cast<int>(std::lround(src_w * scale)) / 2 * 2;
    dst_h = static_cast<int>(std::lround(src_h * scale)) / 2 * 2;
    pad_w = (w - dst_w) / 2;
    pad_h = (h - dst_h) / 2;
    if (dst_w <= 0 || dst_h <= 0 || pad_w < 0 || pad_h < 0) {
        CW_LOG_W("图片尺寸错误: %dx%d -> %dx%d", src_w, src_h, w, h);
        return false;
    }
    return true;
}

bool resize(
    const uint8_t* __restrict src,
          uint8_t* __restrict dst,
    const int src_w,
    const int src_h,
    const int dst_w,
    const int dst_h,
    const int channels = 3
);

#if ENABLE_CAIWEI_BACKEND_RKNN
bool resize(
    const uint8_t* src,
          uint8_t* dst,
    const int src_w,
    const int src_h,
    const int w,
    const int h,
    const int dst_w,
    const int dst_h,
    const int pad_w,
    const int pad_h,
    const int channels = 3
);
#endif

inline bool padding(
    const uint8_t* __restrict src,
          uint8_t* __restrict pad,
    const int src_w,
    const int src_h,
    const int pad_w,
    const int pad_h,
    const int w,
    const int h,
    const int channels = 3
) {
    if (
        pad_w <  0 || pad_h <  0 ||
        src_w <= 0 || src_h <= 0 ||
        w     <= 0 || h     <= 0 ||
        pad_w + src_w > w ||
        pad_h + src_h > h
    ) {
        CW_LOG_W("图片尺寸错误: %dx%d + %dx%d -> %dx%d", src_w, src_h, pad_w, pad_h, w, h);
        return false;
    }
    if (pad_w == 0 && pad_h == 0) {
        std::copy_n(src, static_cast<size_t>(src_w) * src_h * channels, pad);
        return true;
    }
    const size_t stride = static_cast<size_t>(src_w) * channels;
    for (int y = 0; y < src_h; ++y) {
        const size_t src_offset = stride * y;
        const size_t dst_offset = (static_cast<size_t>(pad_h + y) * w + pad_w) * channels;
        std::copy_n(src + src_offset, stride, pad + dst_offset);
    }
    return true;
}

template<typename T>
inline std::vector<size_t> nms_boxes(const std::vector<T>& boxes, const float iou_threshold) {
    if (boxes.empty()) {
        return {};
    }
    std::vector<size_t> all_index;
    std::vector<size_t> ret_index;
    all_index.reserve(boxes.size());
    ret_index.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        all_index.push_back(i);
    }
    std::sort(all_index.begin(), all_index.end(), [&](size_t a, size_t z) {
        return boxes[a].score > boxes[z].score;
    });
    std::vector<size_t> tmp_index;
    while (!all_index.empty()) {
        tmp_index.clear();
        const size_t top_index = all_index[0];
        const T& top = boxes[top_index];
        ret_index.push_back(top_index);
        const float area_top = (top.x2 - top.x1) * (top.y2 - top.y1);
        for (size_t i = 1; i < all_index.size(); ++i) {
            const size_t cur_index = all_index[i];
            const T& cur = boxes[cur_index];
            if (cur.class_id != top.class_id) {
                tmp_index.push_back(cur_index);
                continue;
            }
            const float inter_x1 = std::max(top.x1, cur.x1);
            const float inter_y1 = std::max(top.y1, cur.y1);
            const float inter_x2 = std::min(top.x2, cur.x2);
            const float inter_y2 = std::min(top.y2, cur.y2);
            const float w = std::max(0.0F, inter_x2 - inter_x1);
            const float h = std::max(0.0F, inter_y2 - inter_y1);
            const float area_inter = w * h;
            const float area_cur = (cur.x2 - cur.x1) * (cur.y2 - cur.y1);
            const float area_union = area_top + area_cur - area_inter;
            const float iou = area_union <= 0.0F ? 0.0F : area_inter / area_union;
            if (iou < iou_threshold) {
                tmp_index.push_back(cur_index);
            }
        }
        all_index.swap(tmp_index);
    }
    return ret_index;
}

template <typename T, typename R>
inline std::vector<R> top_k(const T* score, const int size, const int top_k, const T confidence_threshold) {
    if (size <= 0 || top_k <= 0) {
        return {};
    }
    std::vector<R> ret;
    ret.reserve(size);
    for (int i = 0; i < size; ++i) {
        if (score[i] >= confidence_threshold) {
            ret.push_back({
                .class_id = i,
                .score    = score[i]
            });
        }
    }
    if (ret.size() <= static_cast<size_t>(top_k)) {
        std::sort(ret.begin(), ret.end(), [](const R& a, const R& z) {
            return a.score > z.score;
        });
        return ret;
    }
    std::partial_sort(ret.begin(), ret.begin() + top_k, ret.end(), [](const R& a, const R& z) {
        return a.score > z.score;
    });
    ret.resize(top_k);
    return ret;
}

template <typename T>
inline void transpose(const T* __restrict src, T* __restrict dst, const int C, const int N) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            dst[n * C + c] = src[c * N + n];
        }
    }
}

template<typename T>
void hwc_to_chw(const T* __restrict hwc, T* __restrict chw, const int H, const int W, const int C) {
    const size_t plane_size = static_cast<size_t>(H) * W;
    for (int y = 0; y < H; ++y) {
        const size_t stride = static_cast<size_t>(y) * W;
        for (int x = 0; x < W; ++x) {
            const size_t pixel_idx = stride + x;
            const T*     pixel_hwc = hwc + pixel_idx * C;
            #if defined(__GNUC__)
            #pragma GCC unroll 3
            #endif
            for (int c = 0; c < C; ++c) {
                chw[c * plane_size + pixel_idx] = pixel_hwc[c];
            }
        }
    }
}

inline void hwc_to_chw_u8_i8(const uint8_t* __restrict hwc, int8_t* __restrict chw, const int H, const int W, const int C) {
    const size_t plane_size = static_cast<size_t>(H) * W;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t   pixel_idx = static_cast<size_t>(y) * W + x;
            const uint8_t* hwc_pixel = hwc + pixel_idx * C;
            #if defined(__GNUC__)
            #pragma GCC unroll 3
            #endif
            for (int c = 0; c < C; ++c) {
                chw[c * plane_size + pixel_idx] = static_cast<int8_t>(static_cast<int>(hwc_pixel[c]) - 128);
            }
        }
    }
}

inline void hwc_to_chw_u8_f32(const uint8_t* __restrict hwc, float* __restrict chw, const int H, const int W, const int C, const float scale) {
    const float scale_inv = 1.0F / scale;
    const size_t plane_size = static_cast<size_t>(H) * W;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t   pixel_idx = static_cast<size_t>(y) * W + x;
            const uint8_t* hwc_pixel = hwc + pixel_idx * C;
            #if defined(__GNUC__)
            #pragma GCC unroll 3
            #endif
            for (int c = 0; c < C; ++c) {
                chw[c * plane_size + pixel_idx] = static_cast<float>(hwc_pixel[c]) * scale_inv;
            }
        }
    }
}

} // media
} // caiwei

#endif // CAIWEI_RUNTIME_MEDIA_TOOL_HPP
