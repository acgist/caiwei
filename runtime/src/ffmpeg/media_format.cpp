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

#define AVIO_FILE_FORMAT "mp4"
#define AVIO_BUFFER_SIZE (1024 * 1024)

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
    if (audio_codec_ctx == nullptr && video_codec_ctx == nullptr) {
        CW_LOG_W("缺少音视频编码器参数");
        return false;
    }
    int ret = AVERROR_UNKNOWN;
    if (audio_time_base != nullptr && audio_codec_ctx != nullptr) {
        this->audio_time_base = new AVRational{ audio_time_base->num, audio_time_base->den };
    } else if (audio_codec_ctx != nullptr) {
        this->audio_time_base = new AVRational{ audio_codec_ctx->time_base.num, audio_codec_ctx->time_base.den };
    } else {
        // -
    }
    if (video_time_base != nullptr && video_codec_ctx != nullptr) {
        this->video_time_base = new AVRational{ video_time_base->num, video_time_base->den };
    } else if (video_codec_ctx != nullptr) {
        this->video_time_base = new AVRational{ video_codec_ctx->time_base.num, video_codec_ctx->time_base.den };
    } else {
        // -
    }
    if (this->mode_file()) {
        ret = avformat_alloc_output_context2(&this->format_ctx, nullptr, nullptr, this->filename.c_str());
        if (ret < 0) {
            CW_LOG_W("输出上下文创建失败: %d", ret);
            this->stop();
            return false;
        }
        ret = avio_open2(&this->format_ctx->pb, this->filename.c_str(), AVIO_FLAG_WRITE, nullptr, nullptr);
        if (ret < 0) {
            CW_LOG_W("输出句柄打开失败: %d", ret);
            this->stop();
            return false;
        }
    } else if (this->mode_callback()) {
        this->format_ctx = avformat_alloc_context();
        if (this->format_ctx == nullptr) {
            CW_LOG_W("输出上下文创建失败");
            this->stop();
            return false;
        }
        this->format_ctx->oformat = av_guess_format(AVIO_FILE_FORMAT, nullptr, nullptr);
        if (this->format_ctx->oformat == nullptr) {
            CW_LOG_W("输出格式创建失败");
            this->stop();
            return false;
        }
        this->buffer = static_cast<uint8_t*>(av_malloc(AVIO_BUFFER_SIZE));
        if (this->buffer == nullptr) {
            CW_LOG_W("输出缓冲区创建失败");
            this->stop();
            return false;
        }
        this->format_ctx->pb    = avio_alloc_context(this->buffer, AVIO_BUFFER_SIZE, 1, this, nullptr, &write_fragmented, nullptr);
        this->format_ctx->flags = AVFMT_FLAG_CUSTOM_IO;
        if (this->format_ctx->pb == nullptr) {
            CW_LOG_W("输出句柄创建失败");
            this->stop();
            return false;
        }
    } else {
        CW_LOG_W("输出类型错误");
        this->stop();
        return false;
    }
    if (audio_codec_ctx == nullptr) {
        CW_LOG_D("忽略音频输出");
    } else {
        this->audio_stream = avformat_new_stream(this->format_ctx, nullptr);
        if (this->audio_stream == nullptr) {
            CW_LOG_W("音频输出文件流创建失败");
            this->stop();
            return false;
        }
        ret = avcodec_parameters_from_context(this->audio_stream->codecpar, audio_codec_ctx);
        if (ret < 0) {
            CW_LOG_W("音频输出参数拷贝失败: %d", ret);
            this->stop();
            return false;
        }
        CW_LOG_D("创建音频输出流: %d", this->audio_stream->index);
    }
    if (video_codec_ctx == nullptr) {
        CW_LOG_D("忽略视频输出");
    } else {
        this->video_stream = avformat_new_stream(this->format_ctx, nullptr);
        if (this->video_stream == nullptr) {
            CW_LOG_W("视频输出文件流创建失败");
            this->stop();
            return false;
        }
        ret = avcodec_parameters_from_context(this->video_stream->codecpar, video_codec_ctx);
        if (ret < 0) {
            CW_LOG_W("视频输出参数拷贝失败: %d", ret);
            this->stop();
            return false;
        }
        CW_LOG_D("创建视频输出流: %d", this->video_stream->index);
    }
    if (!this->send_header()) {
        this->stop();
        return false;
    }
    return true;
}

bool caiwei::media::MediaFormat::stop() {
    this->send_trailer();
    if (this->format_ctx) {
        if (this->format_ctx->pb != nullptr) {
            CW_LOG_I("释放格式化上下文IO");
            if (this->mode_file()) {
                avio_closep(&this->format_ctx->pb);
            } else if (this->mode_callback()) {
                avio_context_free(&this->format_ctx->pb);
            } else {
                CW_LOG_W("输出类型错误");
            }
        }
        CW_LOG_I("释放格式化上下文");
        avformat_close_input(&this->format_ctx);
        this->audio_stream = nullptr;
        this->video_stream = nullptr;
    }
    if (this->buffer != nullptr) {
        av_freep(&this->buffer);
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
    if (!this->write_header) {
        CW_LOG_W("视频写出头部失败");
        return false;
    }
    if (type == MediaType::AUDIO && this->audio_time_base != nullptr && this->audio_stream != nullptr) {
        av_packet_rescale_ts(packet, *this->audio_time_base, this->audio_stream->time_base);
        packet->stream_index = this->audio_stream->index;
    } else if (type == MediaType::VIDEO && this->video_time_base != nullptr && this->video_stream != nullptr) {
        av_packet_rescale_ts(packet, *this->video_time_base, this->video_stream->time_base);
        packet->stream_index = this->video_stream->index;
    } else {
        CW_LOG_W("媒体没有输出媒体");
        return false;
    }
    int ret = av_interleaved_write_frame(this->format_ctx, packet);
    if (ret != 0) {
        CW_LOG_W("发送媒体数据失败: %d", ret);
        return false;
    }
    return this->callable;
}

bool caiwei::media::MediaFormat::send_header() {
    if (this->format_ctx == nullptr) {
        return false;
    }
    AVDictionary* options = nullptr;
    if (this->mode_file()) {
        av_dict_set(&options, "movflags", "faststart", 0);
    } else if (this->mode_callback()) {
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
        return false;
    }
    if (!this->write_header) {
        return false;
    }
    int ret = av_interleaved_write_frame(this->format_ctx, nullptr);
    if (ret != 0) {
        CW_LOG_W("发送媒体数据失败: %d", ret);
    }
    ret = av_write_trailer(this->format_ctx);
    if (ret != 0) {
        CW_LOG_I("视频写出尾部失败: %d", ret);
        return false;
    } else {
        CW_LOG_D("视频写出尾部成功");
        return true;
    }
}

bool caiwei::media::MediaFormat::mode_file() {
    return !this->filename.empty();
}

bool caiwei::media::MediaFormat::mode_callback() {
    return this->format_callback != nullptr;
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
