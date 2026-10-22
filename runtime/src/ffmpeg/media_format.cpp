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

caiwei::media::MediaFormat::MediaFormat(std::string filename) : filename(filename) {
}

caiwei::media::MediaFormat::MediaFormat(FormatCallback format_callback) : format_callback(format_callback) {
}

caiwei::media::MediaFormat::~MediaFormat() {
    this->stop();
}

bool caiwei::media::MediaFormat::open(
    const AVCodecContext* audio_codec_ctx,
    const AVCodecContext* video_codec_ctx,
    const AVRational    * audio_time_base,
    const AVRational    * video_time_base
) {
    this->stop();
    if (!audio_codec_ctx || !video_codec_ctx) {
        CW_LOG_W("缺少音视频编码器参数");
        return false;
    }
    int ret = AVERROR_UNKNOWN;
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
    if (this->format_ctx == nullptr) {
        CW_LOG_W("格式化上下文创建失败");
        return false;
    }
    if (!this->filename.empty()) {
        ret = avformat_alloc_output_context2(&this->format_ctx, nullptr, nullptr, this->filename.c_str());
        if (ret < 0) {
            CW_LOG_W("输出格式创建失败: %d", ret);
            return false;
        }
        ret = avio_open2(&this->format_ctx->pb, this->filename.c_str(), AVIO_FLAG_WRITE, nullptr, nullptr);
        if (ret < 0) {
            CW_LOG_W("输出句柄创建失败: %d", ret);
            return false;
        }
    } else if (this->format_callback != nullptr) {
        this->format_ctx->oformat = av_guess_format("mp4", nullptr, nullptr);
        if (this->format_ctx->oformat == nullptr) {
            CW_LOG_W("输出格式创建失败");
            return false;
        }
        this->buffer.resize(1024 * 1024);
        this->format_ctx->pb    = avio_alloc_context(this->buffer.data(), this->buffer.size(), 1, this, nullptr, &write_fragmented, nullptr);
        this->format_ctx->flags = AVFMT_FLAG_CUSTOM_IO;
        if (this->format_ctx->pb == nullptr) {
            CW_LOG_W("输出句柄创建失败");
            return false;
        }
    } else {
        CW_LOG_W("输出类型错误");
        return false;
    }
    {
        this->audio_stream = avformat_new_stream(this->format_ctx, nullptr);
        if (this->audio_stream == nullptr) {
            CW_LOG_W("音频输出文件流创建失败");
            return false;
        }
        ret = avcodec_parameters_from_context(this->audio_stream->codecpar, audio_codec_ctx);
        if (ret < 0) {
            CW_LOG_W("音频输出参数拷贝失败: %d", ret);
            return false;
        }
        CW_LOG_D("创建音频输出流: %d", this->audio_stream->index);
    }
    {
        this->video_stream = avformat_new_stream(this->format_ctx, nullptr);
        if (this->video_stream == nullptr) {
            CW_LOG_W("视频输出文件流创建失败");
            return false;
        }
        ret = avcodec_parameters_from_context(this->video_stream->codecpar, video_codec_ctx);
        if (ret < 0) {
            CW_LOG_W("视频输出参数拷贝失败: %d", ret);
            return false;
        }
        CW_LOG_D("创建视频输出流: %d", this->video_stream->index);
    }
    return this->send_header();
}

bool caiwei::media::MediaFormat::stop() {
    this->send_trailer();
    if (this->format_ctx) {
        CW_LOG_I("释放格式化上下文");
        avformat_close_input(&this->format_ctx);
    }
    if (this->audio_time_base) {
        delete this->audio_time_base;
        this->audio_time_base = nullptr;
    }
    if (this->video_time_base) {
        delete this->video_time_base;
        this->video_time_base = nullptr;
    }
    return true;
}

bool caiwei::media::MediaFormat::send(MediaType type, AVPacket* packet) {
    if (this->format_ctx == nullptr) {
        CW_LOG_W("格式化上下文未初始化");
        return false;
    }
    if (type == MediaType::AUDIO) {
        av_packet_rescale_ts(packet, *this->audio_time_base, this->audio_stream->time_base);
        packet->stream_index = this->audio_stream->index;
    } else if (type == MediaType::VIDEO) {
        av_packet_rescale_ts(packet, *this->video_time_base, this->video_stream->time_base);
        packet->stream_index = this->video_stream->index;
    } else {
        CW_LOG_W("未知媒体类型");
        return false;
    }
    int ret = av_interleaved_write_frame(this->format_ctx, packet);
    if (ret != 0) {
        CW_LOG_W("发送媒体数据失败: %d", ret);
    }
    return this->callable;
}

bool caiwei::media::MediaFormat::send_header() {
    if (this->format_ctx == nullptr) {
        CW_LOG_W("格式化上下文未初始化");
        return false;
    }
    AVDictionary* options = nullptr;
    if (!this->filename.empty()) {
        av_dict_set(&options, "movflags", "faststart", 0);
    } else if (this->format_callback != nullptr) {
        av_dict_set(&options, "movflags", "frag_keyframe+empty_moov+delay_moov+default_base_moof", 0);
    } else {
        CW_LOG_W("输出类型错误");
        return false;
    }
    int ret = avformat_write_header(this->format_ctx, &options);
    this->write_header = ret == 0;
    if (this->write_header) {
        CW_LOG_D("视频写出头部成功");
    } else {
        CW_LOG_W("视频写出头部失败");
    }
    av_dict_free(&options);
    return this->write_header;
}

bool caiwei::media::MediaFormat::send_trailer() {
    if (this->format_ctx == nullptr) {
        CW_LOG_W("格式化上下文未初始化");
        return false;
    }
    int ret = av_interleaved_write_frame(this->format_ctx, nullptr);
    if (ret != 0) {
        CW_LOG_W("发送媒体数据失败: %d", ret);
    }
    if (this->write_header) {
        ret = av_write_trailer(this->format_ctx);
        if (ret != 0) {
            CW_LOG_W("视频写出尾部失败");
            return false;
        } else {
            CW_LOG_D("视频写出尾部成功");
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
    if (format->format_callback) {
        format->callable = format->format_callback(buf_size, buf);
    } else {
        format->callable = false;
    }
    return 0;
}
