#include "caiwei/env.hpp"
#include "caiwei/media.hpp"

extern "C" {

#include "libavutil/opt.h"
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavformat/avformat.h"
#include "libavdevice/avdevice.h"
#include "libswresample/swresample.h"

}

#ifdef ALIANG_VIDEO_BIT_RATE
static const int video_bit_rate = ALIANG_VIDEO_BIT_RATE;
#else
static const int video_bit_rate = 512'000;
#endif

#ifdef ALIANG_VIDEO_GOP_SIZE
static const int video_gop_size = ALIANG_VIDEO_GOP_SIZE;
#else
static const int video_gop_size = 10;
#endif

#ifdef ALIANG_VIDEO_MAX_B_FRAMES
static const int video_max_b_frames = ALIANG_VIDEO_MAX_B_FRAMES;
#else
static const int video_max_b_frames = 1;
#endif

caiwei::media::MediaMuxer::MediaMuxer(
    int fps, int width, int height, int channels, int bit_rate, int sample_rate, PacketCallback packet_callback
) : fps(fps), width(width), height(height), packet_callback(packet_callback) {
}

caiwei::media::MediaMuxer::~MediaMuxer() {
}

bool caiwei::media::MediaMuxer::open() {
    return this->open_audio() && this->open_video();
}

bool caiwei::media::MediaMuxer::open_audio() {
    #ifdef ALIANG_AUDIO_ENCODER
    this->audio_codec = avcodec_find_encoder_by_name(ALIANG_AUDIO_ENCODER);
    #else
    this->audio_codec = avcodec_find_encoder(AV_CODEC_ID_AAC);
    #endif
    if (!this->audio_codec)
    {
        // SPDLOG_WARN("不支持音频编码器：{}", this->stream_id);
        return;
    }
    
    this->audio_codec_ctx = avcodec_alloc_context3(this->audio_codec);
    if (!this->audio_codec_ctx)
    {
        // SPDLOG_WARN("音频编码器上下文创建失败：{}", this->stream_id);
        return;
    }
    
    int audio_sample_rate = 8000;      // 标准音频采样率8000
    int audio_channels    = 1;         // 单声道
    int audio_bit_rate    = 64000;     // aac‑lc 64kbps
    this->audio_codec_ctx->sample_rate    = audio_sample_rate;
    this->audio_codec_ctx->channels       = audio_channels;
    this->audio_codec_ctx->channel_layout = av_get_default_channel_layout(audio_channels);
    this->audio_codec_ctx->sample_fmt     = AV_SAMPLE_FMT_FLTP; // aac编码器必须 FLTP 平面浮点
    this->audio_codec_ctx->bit_rate       = audio_bit_rate;
    this->audio_codec_ctx->time_base      = AVRational{1, audio_sample_rate};
    
    // aac 专属参数：帧采样点数；8000hz单声道aac一帧1024采样点
    this->audio_codec_ctx->frame_size = 1024;
    
    // 打开音频编码器
    if (avcodec_open2(this->audio_codec_ctx, this->audio_codec, nullptr) != 0)
    {
        // SPDLOG_WARN("音频编码器打开失败：{}", this->stream_id);
        return;
    }
    // SPDLOG_INFO("音频编码器打开成功");
    
    // 分配音频 AVFrame
    this->audio_frame = av_frame_alloc();
    if (!this->audio_frame)
    {
        // SPDLOG_WARN("音频帧创建失败：{}", this->stream_id);
        return;
    }
    this->audio_frame->sample_rate    = this->audio_codec_ctx->sample_rate;
    this->audio_frame->channels       = this->audio_codec_ctx->channels;
    this->audio_frame->channel_layout = this->audio_codec_ctx->channel_layout;
    this->audio_frame->format         = this->audio_codec_ctx->sample_fmt;
    this->audio_frame->nb_samples     = this->audio_codec_ctx->frame_size; // aac 1024
    
    if (av_frame_get_buffer(this->audio_frame, 0) != 0)
    {
        // SPDLOG_WARN("音频帧内存分配失败：{}", this->stream_id);
        return;
    }
    
    // 音频 AVPacket
    this->audio_packet = av_packet_alloc();
    if (!this->audio_packet)
    {
        // SPDLOG_WARN("音频包创建失败：{}", this->stream_id);
        return;
    }    
}

bool caiwei::media::MediaMuxer::open_video() {
    // SPDLOG_INFO("视频编码配置：{} - {} - {} - {}", this->stream_id, fps, width, height);
    #ifdef ALIANG_VIDEO_ENCODER
    this->video_codec = avcodec_find_encoder_by_name(ALIANG_VIDEO_ENCODER);
    #else
    this->video_codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    #endif
    if(!this->video_codec) {
        // SPDLOG_WARN("不支持编码器：{}", this->stream_id);
        return;
    }
    // SPDLOG_INFO("视频编码器配置：{} - {} - {}", this->stream_id, this->codec->name, this->codec->long_name);
    this->video_codec_ctx = avcodec_alloc_context3(this->video_codec);
    if(!this->video_codec_ctx) {
        // SPDLOG_WARN("编码器创建失败：{}", this->stream_id);
        return;
    } else {
        // SPDLOG_INFO("编码器创建成功");
    }
    // ffmpeg -h encoder=libx264
    this->video_codec_ctx->width     = this->width;
    this->video_codec_ctx->height    = this->height;
    this->video_codec_ctx->pix_fmt   = AV_PIX_FMT_YUV420P;
    this->video_codec_ctx->bit_rate  = video_bit_rate;
    this->video_codec_ctx->time_base = AVRational{ 1, this->fps };
    this->video_codec_ctx->framerate = AVRational{ this->fps, 1 };
    this->video_codec_ctx->gop_size  = video_gop_size;
    this->video_codec_ctx->max_b_frames = video_max_b_frames;
    if (this->video_codec->id == AV_CODEC_ID_H264) {
        // https://trac.ffmpeg.org/wiki/Encode/H.264
        #ifdef ALIANG_VIDEO_H264
        av_opt_set(this->video_codec_ctx->priv_data, "tune",    ALIANG_VIDEO_TUNE,    0);
        av_opt_set(this->video_codec_ctx->priv_data, "preset",  ALIANG_VIDEO_PRESET,  0);
        av_opt_set(this->video_codec_ctx->priv_data, "profile", ALIANG_VIDEO_PROFILE, 0);
        #else
        av_opt_set(this->video_codec_ctx->priv_data, "tune",    "zerolatency", 0);
        av_opt_set(this->video_codec_ctx->priv_data, "preset",  "fast",        0);
        av_opt_set(this->video_codec_ctx->priv_data, "profile", "baseline",    0);
        // 高级画质
        // av_opt_set(this->video_codec_ctx->priv_data, "tune",    "film", 0);
        // av_opt_set(this->video_codec_ctx->priv_data, "preset",  "slow", 0);
        // av_opt_set(this->video_codec_ctx->priv_data, "profile", "high", 0);
        #endif
    }
    if(avcodec_open2(this->video_codec_ctx, this->video_codec, NULL) != 0) {
        // SPDLOG_WARN("编码器打开失败：{}", this->stream_id);
        return;
    } else {
        // SPDLOG_INFO("编码器打开成功");
    }
    this->video_frame = av_frame_alloc();
    if(!this->video_frame) {
        // SPDLOG_WARN("视频帧创建失败：{}", this->stream_id);
        return;
    } else {
        // SPDLOG_INFO("视频帧创建成功");
    }
    this->video_frame->width  = this->video_codec_ctx->width;
    this->video_frame->height = this->video_codec_ctx->height;
    this->video_frame->format = this->video_codec_ctx->pix_fmt;
    if(av_frame_get_buffer(this->video_frame, 0) != 0) {
        // SPDLOG_WARN("视频帧内存分配失败：{}", this->stream_id);
        return;
    } else {
        // SPDLOG_INFO("编码器内存分配成功");
    }
    this->video_packet = av_packet_alloc();
    if(!this->video_packet) {
        // SPDLOG_WARN("视频包创建失败：{}", this->stream_id);
        return;
    } else {
        // SPDLOG_INFO("视频包创建成功");
    }
}

bool caiwei::media::MediaMuxer::stop() {
    return true;
}

bool caiwei::media::MediaMuxer::stop_audio() {
    if(audio_codec_ctx) avcodec_close(audio_codec_ctx);
    avcodec_free_context(&audio_codec_ctx);
    av_frame_free(&audio_frame);
    av_packet_free(&audio_packet);
}

bool caiwei::media::MediaMuxer::stop_video() {
    if(this->video_frame) {
        // SPDLOG_INFO("释放视频帧：{}", this->stream_id);
        av_frame_unref(this->video_frame);
        av_frame_free(&this->video_frame);
        this->video_frame = nullptr;
    }
    if(this->video_packet) {
        // SPDLOG_INFO("释放视频包：{}", this->stream_id);
        av_packet_unref(this->video_packet);
        av_packet_free(&this->video_packet);
        this->video_packet = nullptr;
    }
    if(this->video_codec_ctx) {
        // SPDLOG_INFO("释放视频编码器：{}", this->stream_id);
        avcodec_free_context(&this->video_codec_ctx);
        this->video_codec_ctx = nullptr;
    }
}

bool caiwei::media::MediaMuxer::on_audio(const AudioFrame& frame) {
    return this->on_audio(frame.msec, frame.frames, frame.data_length, frame.data.data());
}

bool caiwei::media::MediaMuxer::on_video(const VideoFrame& frame) {
    return this->on_video(frame.width, frame.height, frame.msec, frame.frames, frame.data_length, frame.data.data());
}

bool caiwei::media::MediaMuxer::on_audio(const double /* msec */, const double /* frames */, const uint32_t data_length, const uint8_t* data) {
    if (!this->audio_codec_ctx || !this->audio_frame)
    {
        return false;
    }

    //==== 1.重采样：输入S16 → 编码器需要 FLTP ====
    // 初始化swr，只初始化一次
    if (!m_swr_ctx)
    {
        // 输入：8000 单声道 S16；输出：编码器参数
        m_swr_ctx = swr_alloc_set_opts(nullptr,
            av_get_default_channel_layout(audio_codec_ctx->channels),
            audio_codec_ctx->sample_fmt,
            audio_codec_ctx->sample_rate,

            av_get_default_channel_layout(1),          // 输入声道
            AV_SAMPLE_FMT_S16,                         // 输入PCM格式
            8000,                                       // 输入采样率
            0, nullptr);

        if(swr_init(m_swr_ctx) < 0)
        {
            // SPDLOG_ERROR("swr init fail");
            return false;
        }
    }

    // 输出最多可以得到多少采样点
    int out_samples_max = swr_get_out_samples(m_swr_ctx, in_samples);
    if (out_samples_max <= 0)
        return false;

    // 重采样输出buffer，按需重新分配
    if(m_swr_out_nb_samples < out_samples_max)
    {
        av_freep(&m_swr_out_buf[0]);
        av_freep(&m_swr_out_buf);
        av_samples_alloc_array_and_samples(&m_swr_out_buf, nullptr,
            audio_codec_ctx->channels, out_samples_max,
            audio_codec_ctx->sample_fmt, 0);
        m_swr_out_nb_samples = out_samples_max;
    }

    // 执行重采样
    int converted = swr_convert(m_swr_ctx,
        m_swr_out_buf, out_samples_max,
        (const uint8_t**)&in_data, in_samples);

    if(converted <= 0)
        return false;

    //==== 2.把重采样出来的FLTP数据，不断填充audio_frame，凑足 frame_size(1024)才编码 ====
    int pos = 0;
    int frame_sz = audio_codec_ctx->frame_size; // aac=1024

    while (converted - pos >= frame_sz)
    {
        // audio_frame 可写
        if (av_frame_make_writable(this->audio_frame) < 0)
        {
            // SPDLOG_WARN("audio frame make writable failed");
            pos += frame_sz;
            continue;
        }

        // 从swr输出拷贝 frame_sz 个采样到 audio_frame
        av_samples_copy(this->audio_frame->data,
            &m_swr_out_buf[0][pos * av_get_bytes_per_sample(audio_codec_ctx->sample_fmt)],
            0, 0,
            frame_sz,
            audio_codec_ctx->channels,
            audio_codec_ctx->sample_fmt);
        // int64_t rtp_audio_ts = av_rescale_q(packet->pts, audio_codec_ctx->time_base, AVRational{1,8000});
        this->audio_frame->nb_samples = frame_sz;
        this->audio_frame->pts = audio_pts;
        audio_pts += frame_sz; // pts累加采样点数

        //==== send_frame 编码 ====
        int ret = avcodec_send_frame(this->audio_codec_ctx, this->audio_frame);
        if (ret != 0)
        {
            // SPDLOG_WARN("audio send frame ret={}", ret);
            pos += frame_sz;
            continue;
        }

        // 接收aac包，回调出去
        while (avcodec_receive_packet(this->audio_codec_ctx, this->audio_packet) == 0)
        {
            this->packet_callback(caiwei::media::MediaType::AUDIO, this->audio_packet);
            av_packet_unref(this->audio_packet);
        }
        pos += frame_sz;
    }

    return true;
}

bool caiwei::media::MediaMuxer::on_video(const int width, const int height, const double /* msec */, const double /* frames */, const uint32_t data_length, const uint8_t* data) {
    // 宽高变化：重建frame + 重建sws上下文
    if(width != this->video_frame->width || height != this->video_frame->height) {
        // SPDLOG_INFO("视频高宽发生变化：{} - {} - {} - {} - {}", this->stream_id, width, height, this->video_frame->width, this->video_frame->height);

        av_frame_unref(this->video_frame);
        this->video_frame->width  = width;
        this->video_frame->height = height;
        this->video_frame->format = this->video_codec_ctx->pix_fmt; // AV_PIX_FMT_YUV420P
        int ret = av_frame_get_buffer(this->video_frame, 0);
        if(ret < 0) {
            // SPDLOG_WARN("视频帧内存分配失败");
            return false;
        }

        // 宽高变动，销毁旧sws
        sws_freeContext(m_sws_ctx);
        m_sws_ctx = nullptr;
        m_sws_w = 0;
        m_sws_h = 0;
    }

    // 初始化/复用 sws_context RGB24 → YUV420P
    if(!m_sws_ctx || m_sws_w != width || m_sws_h != height)
    {
        m_sws_ctx = sws_getContext(
            width, height, AV_PIX_FMT_RGB24,
            width, height, AV_PIX_FMT_YUV420P,
            SWS_BILINEAR,
            nullptr, nullptr, nullptr);

        if(!m_sws_ctx)
        {
            // SPDLOG_WARN("sws_getContext 创建失败");
            return false;
        }
        m_sws_w = width;
        m_sws_h = height;
    }

    // 确保YUV frame可写入
    if(av_frame_make_writable(this->video_frame) != 0) {
        // SPDLOG_WARN("标记视频帧可写入失败");
        return false;
    }

    // 构造源data、stride：RGB24交错，linesize = width*3
    const uint8_t* src_data[1] = { data };
    int src_stride[1] = { width * 3 };

    // ========== RGB24 → YUV420P 转换 ==========
    sws_scale(m_sws_ctx,
        src_data,
        src_stride,
        0,
        height,
        this->video_frame->data,
        this->video_frame->linesize);

    // 设置pts，90000是RTP时间基
    // av_rescale_q(packet->pts, this->video_codec_ctx->time_base, AVRational{1,90000});
    this->video_frame->pts = this->video_frames++ * 90000 / this->fps;

    // 送入编码器
    int ret = avcodec_send_frame(this->video_codec_ctx, this->video_frame);
    if (ret != 0) {
        char err[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, err, sizeof err);
        // SPDLOG_WARN("编码失败：{} {}", this->stream_id, err);
        return false;
    }

    while(avcodec_receive_packet(this->video_codec_ctx, this->video_packet) == 0) {
        this->packet_callback(caiwei::media::MediaType::VIDEO, this->video_packet);
        av_packet_unref(this->video_packet);
    }
    return true;
}
