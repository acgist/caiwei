#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/media.hpp"

extern "C" {

#include "libavutil/opt.h"
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavformat/avformat.h"
#include "libavdevice/avdevice.h"
#include "libswresample/swresample.h"

}

#if FF_API_AVIO_WRITE_NONCONST
static int write_fragmented(void* opaque, uint8_t* buf, int buf_size);
#else
static int write_fragmented(void* opaque, const uint8_t* buf, int buf_size);
#endif

caiwei::media::MediaFormat::MediaFormat(FormatCallback format_callback) : format_callback(format_callback) {
}

caiwei::media::MediaFormat::~MediaFormat() {
    this->stop();
}

bool caiwei::media::MediaFormat::open(const MediaMuxer& muxer) {
    this->audio_info = muxer.out_audio_info;
    this->video_info = muxer.out_video_info;
    if (!muxer.audio_codec_ctx || !muxer.video_codec_ctx) {
        LOG_WARN("缺少音视频编码器参数");
        return false;
    }
    this->format_ctx = avformat_alloc_context();
    if(!this->format_ctx) {
        LOG_WARN("格式化上下文创建失败");
        return false;
    }
    this->format_ctx->oformat = av_guess_format("mp4", NULL, NULL);
    if(!this->format_ctx->oformat) {
        LOG_WARN("输出格式创建失败");
        return false;
    }
    this->buffer.resize(1024 * 1024);
    this->format_ctx->pb    = avio_alloc_context(this->buffer.data(), this->buffer.size(), 1, this, NULL, &write_fragmented, NULL);
    this->format_ctx->flags = AVFMT_FLAG_CUSTOM_IO;
    if(!this->format_ctx->pb) {
        LOG_WARN("输出句柄创建失败");
        return false;
    }
    {
        this->audio_stream = avformat_new_stream(this->format_ctx, NULL);
        if(!this->audio_stream) {
            LOG_WARN("音频输出文件流创建失败");
            return false;
        }
        if(avcodec_parameters_from_context(this->audio_stream->codecpar, muxer.audio_codec_ctx) < 0) {
            LOG_WARN("音频输出参数拷贝失败");
            return false;
        }
    }
    {
        this->video_stream = avformat_new_stream(this->format_ctx, NULL);
        if(!this->video_stream) {
            LOG_WARN("视频输出文件流创建失败");
            return false;
        }
        if(avcodec_parameters_from_context(this->video_stream->codecpar, muxer.video_codec_ctx) < 0) {
            LOG_WARN("视频输出参数拷贝失败");
            return false;
        }
    }
    return true;
}

bool caiwei::media::MediaFormat::stop() {
    this->send_trailer();
    if(this->format_ctx) {
        LOG_INFO("释放格式化上下文");
        avformat_close_input(&this->format_ctx);
        this->format_ctx = nullptr;
    }
    return true;
}

bool caiwei::media::MediaFormat::send(MediaType type, AVPacket* packet) {
    if(!this->format_ctx) {
        return false;
    }
    // if(type == MediaType::VIDEO && (packet->flags & AV_PKT_FLAG_KEY) == AV_PKT_FLAG_KEY) {
        // I帧
        if(this->need_header) {
            this->send_header();
            this->need_header = false;
        }
    // } else {
    //     // B帧 P帧
    // }
    if(!this->need_header) {
        if (type == MediaType::AUDIO) {
            packet->pts = av_rescale_q(packet->pts, AVRational{ 1, this->audio_info.sample_rate }, this->audio_stream->time_base);
            packet->dts = packet->pts;
            packet->stream_index = this->audio_stream->index;
        } else if (type == MediaType::VIDEO) {
            packet->pts = av_rescale_q(packet->pts, AVRational{ 1, this->video_info.fps }, this->video_stream->time_base);
            packet->dts = packet->pts;
            packet->duration = av_rescale_q(1, AVRational{ 1, this->video_info.fps }, this->video_stream->time_base);
            packet->stream_index = this->video_stream->index;
        } else {
            LOG_WARN("未知的媒体类型");
            return false;
        }
        if(av_interleaved_write_frame(this->format_ctx, packet) < 0) {
            LOG_WARN("发送媒体包失败");
        }
    }
    return true;
}

void caiwei::media::MediaFormat::send_header() {
    if(!this->format_ctx) {
        return;
    }
    AVDictionary* options = nullptr;
    // frag_keyframe+empty_moov
    // frag_keyframe+empty_moov+default_base_moof
    // frag_keyframe+empty_moov+delay_moov+default_base_moof
    // Win   h264_mf可以使用delay_moov | empty_moov
    // Linux libx264只能使用delay_moov否则空头可能解析失败
    #ifdef CAIWEI_VIDEO_MOVFLAGS
    av_dict_set(&options, "movflags", ALIANG_VIDEO_MOVFLAGS, 0);
    #else
    // 文件
    // av_dict_set(&options, "movflags", "faststart", 0);
    // 网络
    // TODO 测试 linux 不要 empty_moov
    av_dict_set(&options, "movflags", "frag_keyframe+empty_moov+delay_moov+default_base_moof", 0);
    #endif
    if(avformat_write_header(this->format_ctx, &options) < 0) {
        LOG_WARN("视频写出头部失败");
    }
    av_dict_free(&options);
}

void caiwei::media::MediaFormat::send_trailer() {
    if(!this->format_ctx) {
        return;
    }
    av_interleaved_write_frame(this->format_ctx, nullptr);
    // 写出文件尾部
    if(av_write_trailer(this->format_ctx) < 0) {
        LOG_WARN("视频写出尾部失败");
    }
}

#if FF_API_AVIO_WRITE_NONCONST
static int write_fragmented(void* opaque, uint8_t* buf, int buf_size) {
#else
static int write_fragmented(void* opaque, const uint8_t* buf, int buf_size) {
#endif
    caiwei::media::MediaFormat* format = static_cast<caiwei::media::MediaFormat*>(opaque);
    if(format->format_callback) {
        format->format_callback(buf_size, buf);
    } else {
        // 没有回调
    }
    return 0;
}