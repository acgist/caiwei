#include "caiwei/log.hpp"
#include "caiwei/session.hpp"
#include "caiwei/media_tool.hpp"

#include "base64/base64.h"

extern "C" {
#include "libavcodec/avcodec.h"
}

caiwei::session::VideoPlaySession::VideoPlaySession(bool copy, std::string type, std::string url, Callback callback)
  : StatefulSession(std::move(callback)),
    copy(copy),
    type(std::move(type)),
    url(std::move(url)) {
}

std::future<bool> caiwei::session::VideoPlaySession::get() {
    std::promise<bool> promise;
    if (this->copy) {
        promise.set_value(this->media_copy());
    } else {
        promise.set_value(this->media_codec());
    }
    return promise.get_future();
}

bool caiwei::session::VideoPlaySession::media_copy() {
    caiwei::media::MediaFormat media_format([this](uint32_t length, const uint8_t* data) {
        if (this->callback) {
            auto output = base64_encode(data, length);
            if (this->callback(output.data(), output.length())) {
                return true;
            } else {
                CW_LOG_I("视频数据回调失败释放资源: %s - %s", this->type.c_str(), this->url.c_str());
                return false;
            }
        }
        return false;
    });
    caiwei::media::MediaDemuxer media_demuxer(this->type, this->url);
    CW_LOG_I("开始播放视频: %s = %s", this->type.c_str(), this->url.c_str());
    bool ret = media_demuxer.open(
        caiwei::media::AudioInfo(1, 16000, AV_SAMPLE_FMT_S16),
        caiwei::media::VideoInfo(640, 0, AV_PIX_FMT_RGB24),
        [](caiwei::media::AudioFrame& frame) {
            return true;
        },
        [](caiwei::media::VideoFrame& frame) {
            return true;
        },
        &media_format
    );
    media_demuxer.stop();
    media_format.stop();
    return ret;
}

bool caiwei::session::VideoPlaySession::media_codec() {
    caiwei::media::MediaFormat media_format([this](uint32_t length, const uint8_t* data) {
        if (this->callback) {
            auto output = base64_encode(data, length);
            if (this->callback(output.data(), output.length())) {
                return true;
            } else {
                CW_LOG_I("视频数据回调失败释放资源: %s - %s", this->type.c_str(), this->url.c_str());
                return false;
            }
        }
        return false;
    });
    caiwei::media::MediaMuxer media_muxer(
        caiwei::media::AudioInfo(1, 16000, AV_SAMPLE_FMT_S16),
        caiwei::media::AudioInfo(2, 48000, AV_SAMPLE_FMT_FLTP),
        caiwei::media::VideoInfo(    640, 360, AV_PIX_FMT_RGB24),
        caiwei::media::VideoInfo(24, 640, 360, AV_PIX_FMT_YUV420P),
        [&media_format](caiwei::media::MediaType type, AVPacket* packet) {
            return media_format.send(type, packet);
        }
    );
    caiwei::media::MediaDemuxer media_demuxer(this->type, this->url);
    CW_LOG_I("开始播放视频: %s = %s", this->type.c_str(), this->url.c_str());
    if (!media_muxer.open()) {
        CW_LOG_W("视频编码器上下文创建失败");
        return false;
    }
    if (!media_format.open(media_muxer.audio_codec_ctx, media_muxer.video_codec_ctx)) {
        CW_LOG_W("视频格式化上下文创建失败");
        return false;
    }
    if (!media_demuxer.open(
        caiwei::media::AudioInfo(1, 16000, AV_SAMPLE_FMT_S16),
        caiwei::media::VideoInfo(640, 0, AV_PIX_FMT_RGB24),
        [&media_muxer](caiwei::media::AudioFrame& frame) {
            return media_muxer.on_audio(frame);
        },
        [&media_muxer](caiwei::media::VideoFrame& frame) {
            return media_muxer.on_video(frame);
        }
    )) {
        CW_LOG_W("视频解码器上下文创建失败");
        return false;
    }
    media_demuxer.stop();
    media_format.stop();
    media_muxer.stop();
    return true;
}
