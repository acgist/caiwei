#include "caiwei/log.hpp"
#include "caiwei/media_tool.hpp"
#include "caiwei/session.hpp"

extern "C" {
#include "libavutil/base64.h"
}

inline static bool
callback_adapter(std::string& output, uint32_t length, const uint8_t* data, caiwei::session::Callback& callback) {
    if (callback == nullptr) {
        return false;
    }
    if (length == 0 || data == nullptr) {
        return callback(nullptr, 0);
    }
    const auto len = AV_BASE64_SIZE(length);
    if (output.size() < len) {
        output.resize(len);
    }
    char* ret = av_base64_encode(output.data(), len, data, length);
    if (ret == nullptr) {
        CW_LOG_W("媒体数据编码失败");
        return false;
    }
    if (callback(ret, len - 1)) {
        return true;
    } else {
        CW_LOG_W("媒体数据回调失败");
        return false;
    }
}

caiwei::session::VideoPlaySession::VideoPlaySession(std::string url, std::string type, bool codec, Callback callback) :
    StatefulSession(std::move(callback)),
    url(std::move(url)),
    type(std::move(type)),
    codec(codec) {
}

bool caiwei::session::VideoPlaySession::get() {
    if (this->callback == nullptr) {
        return false;
    }
    if (this->codec) {
        return this->media_codec();
    } else {
        return this->media_forward();
    }
}

bool caiwei::session::VideoPlaySession::media_codec() {
    std::string output;
    output.resize(1024 * 1024);
    caiwei::media::MediaFormat media_format([this, &output](uint32_t length, const uint8_t* data) {
        return callback_adapter(output, length, data, this->callback);
    });
    caiwei::media::MediaMuxer media_muxer(
        caiwei::media::AUDIO_INFO_ENCODE,
        caiwei::media::VIDEO_INFO_ENCODE,
        [&media_format](caiwei::media::MediaType type, AVPacket* packet) {
            return media_format.send(type, packet);
        },
        &media_format
    );
    caiwei::media::MediaDemuxer media_demuxer(this->type, this->url, &media_muxer);
    CW_LOG_I("开始播放媒体: %s = %s", this->type.c_str(), this->url.c_str());
    bool ret = media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        [&media_muxer](caiwei::media::AudioFrame& frame) {
            return media_muxer.on_audio(frame);
        },
        [&media_muxer](caiwei::media::VideoFrame& frame) {
            return media_muxer.on_video(frame);
        }
    );
    if (ret) {
        CW_LOG_I("媒体播放完成: %s = %s", this->type.c_str(), this->url.c_str());
    } else {
        CW_LOG_W("媒体播放失败: %s = %s", this->type.c_str(), this->url.c_str());
    }
    return ret;
}

bool caiwei::session::VideoPlaySession::media_forward() {
    std::string output;
    output.resize(1024 * 1024);
    caiwei::media::MediaFormat media_format([this, &output](uint32_t length, const uint8_t* data) {
        return callback_adapter(output, length, data, this->callback);
    });
    caiwei::media::MediaDemuxer media_demuxer(this->type, this->url, nullptr, &media_format);
    CW_LOG_I("开始播放媒体: %s = %s", this->type.c_str(), this->url.c_str());
    bool ret = media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        nullptr,
        nullptr,
        [&media_format](caiwei::media::MediaType type, AVPacket* packet) {
            return media_format.send(type, packet);
        }
    );
    if (ret) {
        CW_LOG_I("媒体播放完成: %s = %s", this->type.c_str(), this->url.c_str());
    } else {
        CW_LOG_W("媒体播放失败: %s = %s", this->type.c_str(), this->url.c_str());
    }
    return ret;
}
