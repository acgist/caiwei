#include "caiwei/log.hpp"
#include "caiwei/media.hpp"
#include "caiwei/player.hpp"
#include "caiwei/session.hpp"

#include "base64/base64.h"

extern "C" {

#include "libavcodec/avcodec.h"

}

caiwei::session::VideoPlaySession::VideoPlaySession(std::string type, std::string url, Callback callback)
  : StatefulSession(std::move(callback)),
    type(std::move(type)),
    url(std::move(url)) {
}

std::future<bool> caiwei::session::VideoPlaySession::get() {
    std::promise<bool> promise;
    caiwei::media::MediaDemuxer* demuxer = nullptr;
    caiwei::media::MediaFormat media_format([this, &demuxer](uint32_t length, const uint8_t* data) {
        if (this->callback) {
            auto output = base64_encode(data, length);
            bool ret = this->callback("data", output.data(), output.length());
            if(!ret) {
                CW_LOG_I("视频数据回调失败释放资源: %s - %s", this->type.c_str(), this->url.c_str());
                if (demuxer) {
                    demuxer->stop();
                }
            }
            return ret;
        }
        return true;
    });
    caiwei::media::MediaMuxer media_muxer(
        caiwei::media::AudioInfo(1, 16000, AV_SAMPLE_FMT_S16),
        caiwei::media::VideoInfo(640, 360, AV_PIX_FMT_RGB24),
        caiwei::media::AudioInfo(2, 48000, AV_SAMPLE_FMT_FLTP),
        caiwei::media::VideoInfo(25, 640, 360, AV_PIX_FMT_YUV420P),
        [&media_format](caiwei::media::MediaType type, AVPacket* packet) {
            media_format.send(type, packet);
            return true;
        }
    );
    media_muxer.open();
    media_format.open(media_muxer);
    caiwei::media::MediaDemuxer media_demuxer(this->type, this->url, [&media_muxer](const caiwei::media::AudioFrame& frame) {
        media_muxer.on_audio(frame);
        return true;
    }, [&media_muxer](const caiwei::media::VideoFrame& frame) {
        media_muxer.on_video(frame);
        return true;
    });
    demuxer = &media_demuxer;
    CW_LOG_I("开始播放视频: %s = %s", this->type.c_str(), this->url.c_str());
    bool ret = media_demuxer.open(caiwei::media::AudioInfo(1, 16000, AV_SAMPLE_FMT_S16), caiwei::media::VideoInfo(640, 0, AV_PIX_FMT_RGB24));
    media_demuxer.stop();
    media_format.stop();
    media_muxer.stop();
    promise.set_value(ret);
    return promise.get_future();
}
