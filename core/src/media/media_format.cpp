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

// 视频回调
#if FF_API_AVIO_WRITE_NONCONST
static int write_fragmented(void* opaque, uint8_t* buf, int buf_size);
#else
static int write_fragmented(void* opaque, const uint8_t* buf, int buf_size);
#endif

caiwei::media::MediaFormat::MediaFormat(FormatCallback format_callback) : format_callback(format_callback) {

}

caiwei::media::MediaFormat::~MediaFormat() {

}

bool caiwei::media::MediaFormat::open(const MediaMuxer& muxer) {
    if(!muxer.video_codec_ctx) {
        // SPDLOG_WARN("缺少视频编码器参数：{} - {}", this->id, this->stream_id);
        return false;
    }
    this->format_ctx = avformat_alloc_context();
    if(!this->format_ctx) {
        // SPDLOG_WARN("视频格式化上下文创建失败：{} - {}", this->id, this->stream_id);
        return false;
    } else {
        // SPDLOG_INFO("视频格式化上下文创建成功");
    }
    // TODO 宏定义
    this->format_ctx->oformat = av_guess_format("mp4", NULL, NULL);
    if(!this->format_ctx->oformat) {
        // SPDLOG_WARN("视频输出格式创建失败：{} - {}", this->id, this->stream_id);
        return false;
    } else {
        // SPDLOG_INFO("视频输出格式创建成功");
    }
    this->buffer.resize(4 * 1024 * 1024);
    this->format_ctx->pb    = avio_alloc_context(this->buffer.data(), this->buffer.size(), 1, this, NULL, &write_fragmented, NULL);
    this->format_ctx->flags = AVFMT_FLAG_CUSTOM_IO;
    if(!this->format_ctx->pb) {
        // SPDLOG_WARN("视频输出句柄创建失败：{} - {}", this->id, this->stream_id);
        return false;
    } else {
        // SPDLOG_INFO("视频输出句柄创建成功");
    }
    {
        // TODO audio
    }
    {
        this->video_stream = avformat_new_stream(this->format_ctx, NULL);
        if(!this->video_stream) {
            // SPDLOG_WARN("视频输出文件流创建失败：{} - {}", this->id, this->stream_id);
            return false;
        } else {
            // SPDLOG_INFO("视频输出文件流创建成功");
        }
        if(avcodec_parameters_from_context(this->video_stream->codecpar, muxer.video_codec_ctx) < 0) {
            // SPDLOG_WARN("视频输出参数拷贝失败：{} - {}", this->id, this->stream_id);
            return false;
        } else {
            // SPDLOG_INFO("视频输出参数拷贝成功");
        }
    }
    return true;
}

bool caiwei::media::MediaFormat::stop() {
    // SPDLOG_INFO("释放订阅者：{} - {}", this->id, this->stream_id);
    if(this->format_ctx) {
        // SPDLOG_INFO("释放视频格式化上下文：{} - {}", this->id, this->stream_id);
        avformat_close_input(&this->format_ctx);
        this->format_ctx = nullptr;
    }
}

bool caiwei::media::MediaFormat::send(AVPacket* packet) {
    if(!this->format_ctx) {
        return;
    }
    if((packet->flags & AV_PKT_FLAG_KEY) == AV_PKT_FLAG_KEY) {
        // I帧
        if(this->need_header) {
            this->send_header();
            this->need_header = false;
        }
    } else {
        // B帧 P帧
    }
    if(this->need_header) {
        if(av_write_frame(this->format_ctx, packet) < 0) {
            // SPDLOG_WARN("发送视频包失败");
        }
    }
}

void caiwei::media::MediaFormat::send_header() {
    if(!this->format_ctx) {
        return;
    }
    AVDictionary* options = nullptr;
    // frag_keyframe+empty_moov
    // frag_keyframe+empty_moov+default_base_moof
    // frag_keyframe+delay_moov+default_base_moof+faststart
    // Win   h264_mf可以使用delay_moov | empty_moov
    // Linux libx264只能使用delay_moov否则空头可能解析失败
    #ifdef ALIANG_VIDEO_MOVFLAGS
    av_dict_set(&options, "movflags", ALIANG_VIDEO_MOVFLAGS, 0);
    #else
    av_dict_set(&options, "movflags", "frag_keyframe+delay_moov+default_base_moof+faststart", 0);
    #endif
    if(avformat_write_header(this->format_ctx, &options) < 0) {
        // SPDLOG_WARN("视频写出头部失败：{} - {}", this->id, this->stream_id);
    }
    av_dict_free(&options);
}

void caiwei::media::MediaFormat::send_trailer() {
    if(!this->format_ctx) {
        return;
    }
    // 写出文件尾部
    if(av_write_trailer(this->format_ctx) < 0) {
        // SPDLOG_WARN("视频写出尾部失败：{} - {}", this->id, this->stream_id);
    }
}

#if FF_API_AVIO_WRITE_NONCONST
static int write_fragmented(void* opaque, uint8_t* buf, int buf_size) {
#else
static int write_fragmented(void* opaque, const uint8_t* buf, int buf_size) {
#endif
    caiwei::media::MediaFormat* format = static_cast<caiwei::media::MediaFormat*>(opaque);
    if(format->format_callback) {
        // auto base64_data{ base64_encode(buf, buf_size) };
        // if(subscriber->callback(false, aliang::media::NOTIFY_TYPE_VIDEO, base64_data.data(), base64_data.size())) {
        //     // 发送成功
        // } else {
        //     SPDLOG_DEBUG("回调关闭清空回调：{} - {}", subscriber->stream_id, subscriber->id);
        //     subscriber->callback = nullptr;
        // }
    } else {
        // 没有回调
    }
    return 0;
}