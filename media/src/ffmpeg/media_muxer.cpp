#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/media.hpp"

extern "C" {

#include "libavutil/opt.h"
#include "libavutil/audio_fifo.h"
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavformat/avformat.h"
#include "libavdevice/avdevice.h"
#include "libswresample/swresample.h"

}

static SwrContext* init_audio_swr(caiwei::media::AudioInfo& in, caiwei::media::AudioInfo& out);
static SwsContext* init_video_sws(caiwei::media::VideoInfo& in, caiwei::media::VideoInfo& out);

static void sws_free(SwsContext** sws);

caiwei::media::MediaMuxer::MediaMuxer(
    AudioInfo in_audio_info,
    VideoInfo in_video_info,
    AudioInfo out_audio_info,
    VideoInfo out_video_info,
    PacketCallback packet_callback
) : in_audio_info(in_audio_info),
    in_video_info(in_video_info),
    out_audio_info(out_audio_info),
    out_video_info(out_video_info),
    packet_callback(packet_callback) {
}

caiwei::media::MediaMuxer::~MediaMuxer() {
    this->stop();
}

bool caiwei::media::MediaMuxer::open() {
    return this->open_audio() && this->open_video();
}

bool caiwei::media::MediaMuxer::open_audio() {
    int ret = -1;
    #ifdef CAIWEI_AUDIO_ENCODER
    this->audio_codec = avcodec_find_encoder_by_name(CAIWEI_AUDIO_ENCODER);
    #else
    this->audio_codec = avcodec_find_encoder(AV_CODEC_ID_AAC);
    #endif
    if (!this->audio_codec) {
        LOG_WARN("不支持音频编码器");
        return false;
    }
    this->audio_codec_ctx = avcodec_alloc_context3(this->audio_codec);
    if (!this->audio_codec_ctx) {
        LOG_WARN("音频编码器上下文创建失败");
        return false;
    }
    AVChannelLayout ch_layout;
    av_channel_layout_default(&ch_layout, this->out_audio_info.channel);
    this->audio_codec_ctx->bit_rate    = 128'000;
    this->audio_codec_ctx->time_base   = AVRational{ 1, this->out_audio_info.sample_rate };
    this->audio_codec_ctx->ch_layout   = ch_layout;
    this->audio_codec_ctx->sample_fmt  = (AVSampleFormat) this->out_audio_info.format;
    this->audio_codec_ctx->sample_rate = this->out_audio_info.sample_rate;
    ret = avcodec_open2(this->audio_codec_ctx, this->audio_codec, nullptr);
    if (ret != 0) {
        LOG_WARN("音频编码器打开失败: %d", ret);
        return false;
    }
    LOG_INFO("音频编码器打开成功: %s - %d - %d", this->audio_codec->name, this->out_audio_info.channel, this->out_audio_info.sample_rate);
    this->audio_frame = av_frame_alloc();
    if (!this->audio_frame) {
        LOG_WARN("音频帧创建失败");
        return false;
    }
    this->audio_frame->format      = this->audio_codec_ctx->sample_fmt;
    this->audio_frame->ch_layout   = this->audio_codec_ctx->ch_layout;
    this->audio_frame->nb_samples  = this->audio_codec_ctx->frame_size == 0 ? this->audio_codec_ctx->sample_rate : this->audio_codec_ctx->frame_size;
    this->audio_frame->sample_rate = this->audio_codec_ctx->sample_rate;
    ret = av_frame_get_buffer(this->audio_frame, 0);
    if (ret != 0) {
        LOG_WARN("音频帧内存分配失败: %d", ret);
        return false;
    }
    this->audio_packet = av_packet_alloc();
    if (!this->audio_packet) {
        LOG_WARN("音频包创建失败");
        return false;
    }
    this->swr_ctx = init_audio_swr(this->in_audio_info, this->out_audio_info);
    if (!this->swr_ctx) {
        LOG_WARN("音频重采样上下文创建失败");
        return false;
    }
    ret = av_samples_alloc(
        this->audio_ch_buffer,
        nullptr,
        this->audio_codec_ctx->ch_layout.nb_channels,
        // TODO 测试
        this->audio_codec_ctx->frame_size * 8,
        this->audio_codec_ctx->sample_fmt,
        0
    );
    if (ret < 0) {
        LOG_WARN("音频通道缓存内存分配失败: %d", ret);
        return false;
    }
    this->audio_fifo = av_audio_fifo_alloc(
        this->audio_codec_ctx->sample_fmt,
        this->audio_codec_ctx->ch_layout.nb_channels,
        // TODO 测试
        this->audio_codec_ctx->sample_rate * this->audio_codec_ctx->ch_layout.nb_channels
    );
    if (!this->audio_fifo) {
        LOG_WARN("音频缓存创建失败");
        return false;
    }
    return true;
}

bool caiwei::media::MediaMuxer::open_video() {
    int ret = -1;
    #ifdef CAIWEI_VIDEO_ENCODER
    this->video_codec = avcodec_find_encoder_by_name(CAIWEI_VIDEO_ENCODER);
    #else
    this->video_codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    #endif
    if(!this->video_codec) {
        LOG_WARN("不支持视频编码器");
        return false;
    }
    this->video_codec_ctx = avcodec_alloc_context3(this->video_codec);
    if(!this->video_codec_ctx) {
        LOG_WARN("视频编码器上下文创建失败");
        return false;
    }
    this->video_codec_ctx->width        = this->out_video_info.width;
    this->video_codec_ctx->height       = this->out_video_info.height;
    this->video_codec_ctx->pix_fmt      = (AVPixelFormat) this->out_video_info.format;
    this->video_codec_ctx->bit_rate     = 512'000;
    this->video_codec_ctx->gop_size     = this->out_video_info.fps; // 网络流定期I帧最多间隔
//  this->video_codec_ctx->keyint_min   = this->out_video_info.fps; // 网络流定期I帧最少间隔
    this->video_codec_ctx->time_base    = AVRational{ 1, this->out_video_info.fps };
    this->video_codec_ctx->framerate    = AVRational{ this->out_video_info.fps, 1 };
    this->video_codec_ctx->max_b_frames = 0;
    if (this->video_codec->id == AV_CODEC_ID_H264) {
        // https://trac.ffmpeg.org/wiki/Encode/H.264
        av_opt_set(this->video_codec_ctx->priv_data, "tune",    "zerolatency", 0);
        av_opt_set(this->video_codec_ctx->priv_data, "preset",  "fast",        0);
        av_opt_set(this->video_codec_ctx->priv_data, "profile", "baseline",    0);
    }
    ret = avcodec_open2(this->video_codec_ctx, this->video_codec, nullptr);
    if(ret != 0) {
        LOG_WARN("视频编码器打开失败: %d", ret);
        return false;
    }
    LOG_INFO("视频编码器打开成功: %s - %d - %d - %d", this->video_codec->name, this->out_video_info.fps, this->out_video_info.width, this->out_video_info.format);
    this->video_frame = av_frame_alloc();
    if(!this->video_frame) {
        LOG_WARN("视频帧创建失败");
        return false;
    }
    this->video_frame->width  = this->video_codec_ctx->width;
    this->video_frame->height = this->video_codec_ctx->height;
    this->video_frame->format = this->video_codec_ctx->pix_fmt;
    ret = av_frame_get_buffer(this->video_frame, 0);
    if (ret != 0) {
        LOG_WARN("视频帧内存分配失败: %d", ret);
        return false;
    }
    this->video_packet = av_packet_alloc();
    if (!this->video_packet) {
        LOG_WARN("视频包创建失败");
        return false;
    }
    this->sws_ctx = init_video_sws(this->in_video_info, this->out_video_info);
    if (!this->sws_ctx) {
        LOG_WARN("视频重采样上下文创建失败");
        return false;
    }
    return true;
}

bool caiwei::media::MediaMuxer::stop() {
    // TODO 释放顺序验证
    return this->stop_audio() && this->stop_video();
}

bool caiwei::media::MediaMuxer::stop_audio() {
    if (this->audio_fifo) {
        LOG_INFO("释放音频缓存");
        av_audio_fifo_free(this->audio_fifo);
        this->audio_fifo = nullptr;
    }
    if (this->audio_ch_buffer[0]) {
        LOG_INFO("释放音频通道缓存");
        av_freep(&this->audio_ch_buffer[0]);
        // TODO 验证
    }
    if (this->swr_ctx) {
        LOG_INFO("释放音频重采样上下文");
        swr_free(&this->swr_ctx);
        this->swr_ctx = nullptr;
    }
    if (this->audio_frame) {
        LOG_INFO("释放音频帧");
        av_frame_unref(this->audio_frame);
        av_frame_free(&this->audio_frame);
        this->audio_frame = nullptr;
    }
    if (this->audio_packet) {
        LOG_INFO("释放音频包");
        av_packet_unref(this->audio_packet);
        av_packet_free(&this->audio_packet);
        this->audio_packet = nullptr;
    }
    if (this->audio_codec_ctx) {
        LOG_INFO("释放音频编码器");
        avcodec_free_context(&this->audio_codec_ctx);
        this->audio_codec_ctx = nullptr;
    }
    return true;
}

bool caiwei::media::MediaMuxer::stop_video() {
    if (this->sws_ctx) {
        LOG_INFO("释放视频重采样上下文");
        sws_free(&this->sws_ctx);
        this->sws_ctx = nullptr;
    }
    if (this->video_frame) {
        LOG_INFO("释放视频帧");
        av_frame_unref(this->video_frame);
        av_frame_free(&this->video_frame);
        this->video_frame = nullptr;
    }
    if (this->video_packet) {
        LOG_INFO("释放视频包");
        av_packet_unref(this->video_packet);
        av_packet_free(&this->video_packet);
        this->video_packet = nullptr;
    }
    if (this->video_codec_ctx) {
        LOG_INFO("释放视频编码器");
        avcodec_free_context(&this->video_codec_ctx);
        this->video_codec_ctx = nullptr;
    }
    return true;
}

bool caiwei::media::MediaMuxer::on_audio(const AudioFrame& frame) {
    return this->on_audio(frame.samples, frame.msec, frame.frames, frame.data_length, frame.data.data());
}

bool caiwei::media::MediaMuxer::on_video(const VideoFrame& frame) {
    return this->on_video(frame.width, frame.height, frame.msec, frame.frames, frame.data_length, frame.data.data());
}

bool caiwei::media::MediaMuxer::on_audio(const int nb_samples, const size_t msec, const size_t frames, const uint32_t data_length, const uint8_t* data) {
    if (!this->audio_frame || !this->audio_packet || !this->audio_codec_ctx) {
        return false;
    }
    int ret = -1;
    int out_samples = swr_get_out_samples(this->swr_ctx, nb_samples);
        out_samples = swr_convert(this->swr_ctx, this->audio_ch_buffer, out_samples, (const uint8_t**) &data, nb_samples);
    if(out_samples < 0) {
        LOG_WARN("音频重采样失败: %d", out_samples);
        return false;
    }
    av_audio_fifo_write(this->audio_fifo, (void**) this->audio_ch_buffer, out_samples);
    int frame_size = this->audio_codec_ctx->frame_size;
    while (av_audio_fifo_size(this->audio_fifo) >= frame_size) {
        ret = av_frame_make_writable(this->audio_frame);
        if (ret != 0) {
            LOG_WARN("音频帧标记可写入失败: %d", ret);
            return false;
        }
        int read_cnt = av_audio_fifo_read(this->audio_fifo, (void**) this->audio_frame->data, frame_size);
        if(read_cnt != frame_size) {
            LOG_WARN("fifo读取样本不足");
            break;
        }
        this->audio_frame->pts = this->audio_pts;
        this->audio_frame->nb_samples = frame_size;
        this->audio_pts += frame_size;
        ret = avcodec_send_frame(this->audio_codec_ctx, this->audio_frame);
        if (ret != 0) {
            LOG_WARN("音频帧编码失败: %d", ret);
            break;
        }
        while ((ret = avcodec_receive_packet(this->audio_codec_ctx, this->audio_packet)) == 0) {
            this->packet_callback(caiwei::media::MediaType::AUDIO, this->audio_packet);
            av_packet_unref(this->audio_packet);
        }
    }
    return true;
}

bool caiwei::media::MediaMuxer::on_video(const int width, const int height, const size_t msec, const size_t frames, const uint32_t data_length, const uint8_t* data) {
    if (!this->video_frame || !this->video_packet || !this->video_codec_ctx) {
        return false;
    }
    int ret = -1;
    if(width != this->video_frame->width || height != this->video_frame->height) {
        av_frame_unref(this->video_frame);
        this->video_frame->width   = width;
        this->video_frame->height  = height;
        this->video_frame->format  = this->video_codec_ctx->pix_fmt;
        this->in_video_info.width  = width;
        this->in_video_info.height = height;
        ret = av_frame_get_buffer(this->video_frame, 0);
        if(ret != 0) {
            LOG_WARN("视频帧内存分配失败: %d", ret);
            return false;
        }
        sws_free(&this->sws_ctx);
        this->sws_ctx = init_video_sws(this->in_video_info, this->out_video_info);
        if (!this->sws_ctx) {
            LOG_WARN("视频重采样上下文创建失败");
            return false;
        }
    }
    ret = av_frame_make_writable(this->video_frame);
    if(ret != 0) {
        LOG_WARN("视频帧标记可写入失败: %d", ret);
        return false;
    }
    const uint8_t* src_data   = data;
    int            src_stride = width * 3;
    int video_height = sws_scale(sws_ctx, &src_data, &src_stride, 0, height, this->video_frame->data, this->video_frame->linesize);
    if (video_height < 0) {
        LOG_WARN("视频重采样失败: %d", video_height);
        return false;
    }
    this->video_frame->pts = this->video_pts;
    this->video_pts++;
    ret = avcodec_send_frame(this->video_codec_ctx, this->video_frame);
    if (ret != 0) {
        LOG_WARN("视频帧编码失败: %d", ret);
        return false;
    }
    while ((ret = avcodec_receive_packet(this->video_codec_ctx, this->video_packet)) == 0) {
        this->packet_callback(caiwei::media::MediaType::VIDEO, this->video_packet);
        av_packet_unref(this->video_packet);
    }
    return true;
}

static SwrContext* init_audio_swr(caiwei::media::AudioInfo& in_audio_info, caiwei::media::AudioInfo& out_audio_info) {
    SwrContext* swr = swr_alloc();
    if(swr == nullptr) {
        LOG_WARN("打开音频重采样失败");
        return nullptr;
    }
    AVChannelLayout in_ch_layout;
    AVChannelLayout out_ch_layout;
    av_channel_layout_default(&in_ch_layout,  in_audio_info.channel);
    av_channel_layout_default(&out_ch_layout, in_audio_info.channel);
    int ret = swr_alloc_set_opts2(
        &swr,
        &out_ch_layout, (AVSampleFormat) out_audio_info.format, out_audio_info.sample_rate,
        &in_ch_layout,  (AVSampleFormat) in_audio_info.format,  in_audio_info.sample_rate,
        0, nullptr
    );
    if(ret != 0) {
        swr_free(&swr);
        LOG_WARN("打开音频重采样失败: %d", ret);
        return nullptr;
    }
    ret = swr_init(swr);
    if(ret != 0) {
        swr_free(&swr);
        LOG_WARN("打开音频重采样失败: %d", ret);
        return nullptr;
    }
    return swr;
}

static SwsContext* init_video_sws(caiwei::media::VideoInfo& in_video_info, caiwei::media::VideoInfo& out_video_info) {
    SwsContext* sws = sws_getContext(
        in_video_info.width,  in_video_info.height,  (AVPixelFormat) in_video_info.format,
        out_video_info.width, out_video_info.height, (AVPixelFormat) out_video_info.format,
//      SWS_BILINEAR,
        SWS_FAST_BILINEAR,
        nullptr, nullptr, nullptr
    );
    if(sws == nullptr) {
        LOG_INFO("打开视频重采样失败");
        return nullptr;
    }
    return sws;
}

static void sws_free(SwsContext** sws) {
    if(*sws) {
        sws_freeContext(*sws);
        *sws = nullptr;
    }
}
