#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/media_tool.hpp"

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
    delete this->audio_time_base;
    delete this->video_time_base;
}

bool caiwei::media::MediaFormat::open(
    const AVCodecContext* audio_codec_ctx,
    const AVCodecContext* video_codec_ctx,
    AVRational* audio_time_base,
    AVRational* video_time_base
) {
    // TODO 检查是否已经open
    if (!audio_codec_ctx || !video_codec_ctx) {
        CW_LOG_W("缺少音视频编码器参数");
        return false;
    }
    if (audio_time_base != nullptr) {
        this->audio_time_base = new AVRational{ audio_time_base->num, audio_time_base->den };
    } else {
        this->audio_time_base = new AVRational{ audio_codec_ctx->time_base.num, audio_codec_ctx->time_base.den };
    }
    if (video_time_base != nullptr) {
        this->video_time_base = new AVRational{ video_time_base->num, video_time_base->den };
    } else {
        this->video_time_base = new AVRational{ video_codec_ctx->time_base.num, video_codec_ctx->time_base.den };
    }
    this->format_ctx = avformat_alloc_context();
    if(!this->format_ctx) {
        CW_LOG_W("格式化上下文创建失败");
        return false;
    }
    // 写出网络
    this->format_ctx->oformat = av_guess_format("mp4", NULL, NULL);
    if(!this->format_ctx->oformat) {
        CW_LOG_W("输出格式创建失败");
        return false;
    }
    this->buffer.resize(1024 * 1024);
    this->format_ctx->pb    = avio_alloc_context(this->buffer.data(), this->buffer.size(), 1, this, NULL, &write_fragmented, NULL);
    this->format_ctx->flags = AVFMT_FLAG_CUSTOM_IO;
    if(!this->format_ctx->pb) {
        CW_LOG_W("输出句柄创建失败");
        return false;
    }
    // 写出文件
    // if (avformat_alloc_output_context2(&this->format_ctx, nullptr, nullptr, "output.mp4") < 0) {
    //     CW_LOG_W("输出格式创建失败");
    //     return false;
    // }
    {
        this->audio_stream = avformat_new_stream(this->format_ctx, NULL);
        if(!this->audio_stream) {
            CW_LOG_W("音频输出文件流创建失败");
            return false;
        }
        if(avcodec_parameters_from_context(this->audio_stream->codecpar, audio_codec_ctx) < 0) {
            CW_LOG_W("音频输出参数拷贝失败");
            return false;
        }
    }
    {
        this->video_stream = avformat_new_stream(this->format_ctx, NULL);
        if(!this->video_stream) {
            CW_LOG_W("视频输出文件流创建失败");
            return false;
        }
        if(avcodec_parameters_from_context(this->video_stream->codecpar, video_codec_ctx) < 0) {
            CW_LOG_W("视频输出参数拷贝失败");
            return false;
        }
    }
    // 写出文件
    // if (avio_open2(&this->format_ctx->pb, "output.mp4", AVIO_FLAG_WRITE, nullptr, nullptr) < 0) {
    //     CW_LOG_W("输出句柄创建失败");
    //     return false;
    // }
    if (!this->send_header()) {
        return false;
    }
    return true;
}

bool caiwei::media::MediaFormat::stop() {
    this->send_trailer();
    if(this->format_ctx) {
        CW_LOG_I("释放格式化上下文");
        avformat_close_input(&this->format_ctx);
    }
    return true;
}

bool caiwei::media::MediaFormat::send(MediaType type, AVPacket* packet) {
    if(!this->format_ctx) {
        return false;
    }
    if (type == MediaType::AUDIO) {
        av_packet_rescale_ts(packet, *this->audio_time_base, this->audio_stream->time_base);
        packet->stream_index = this->audio_stream->index;
    } else if (type == MediaType::VIDEO) {
        av_packet_rescale_ts(packet, *this->video_time_base, this->video_stream->time_base);
        packet->stream_index = this->video_stream->index;
    } else {
        CW_LOG_W("未知的媒体类型");
        return false;
    }
    if(av_interleaved_write_frame(this->format_ctx, packet) < 0) {
        CW_LOG_W("发送媒体包失败");
    }
    return this->callable;
}

bool caiwei::media::MediaFormat::send_header() {
    if(!this->format_ctx) {
        return false;
    }
    AVDictionary* options = nullptr;
    // frag_keyframe+empty_moov
    // frag_keyframe+empty_moov+default_base_moof
    // frag_keyframe+empty_moov+delay_moov+default_base_moof
    // Win   h264_mf可以使用delay_moov | empty_moov
    // Linux libx264只能使用delay_moov否则空头可能解析失败
    // 写出文件
    // av_dict_set(&options, "movflags", "faststart", 0);
    // 写出网络
    // TODO 测试 linux 不要 empty_moov
    av_dict_set(&options, "movflags", "frag_keyframe+empty_moov+delay_moov+default_base_moof", 0);
    if(avformat_write_header(this->format_ctx, &options) < 0) {
        this->write_header = false;
        CW_LOG_W("视频写出头部失败");
        av_dict_free(&options);
        return false;
    } else {
        this->write_header = true;
        av_dict_free(&options);
        return true;
    }
}

bool caiwei::media::MediaFormat::send_trailer() {
    if(!this->format_ctx) {
        return false;
    }
    // TODO 发送过数据才能关闭
    av_interleaved_write_frame(this->format_ctx, nullptr);
    // 写出文件尾部
    if (this->write_header) {
        if(av_write_trailer(this->format_ctx) < 0) {
            CW_LOG_W("视频写出尾部失败");
            return false;
        } else {
            return true;
        }
    } else {
        return true;
    }
}

#if FF_API_AVIO_WRITE_NONCONST
static int write_fragmented(void* opaque, uint8_t* buf, int buf_size) {
#else
static int write_fragmented(void* opaque, const uint8_t* buf, int buf_size) {
#endif
    caiwei::media::MediaFormat* format = static_cast<caiwei::media::MediaFormat*>(opaque);
    if(format->format_callback) {
        format->callable = format->format_callback(buf_size, buf);
    } else {
        format->callable = false;
    }
    return 0;
}