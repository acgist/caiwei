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

const static AVPixelFormat video_pix_format = AV_PIX_FMT_RGB24;

const static AVSampleFormat  audio_format           = AV_SAMPLE_FMT_S16;
const static AVChannelLayout audio_layout           = AV_CHANNEL_LAYOUT_MONO;
const static int             audio_nb_channels      = 1;
const static int             audio_sample_rate      = 16000;
const static int             audio_bytes_per_sample = av_get_bytes_per_sample(audio_format);

caiwei::media::Frame::Frame(uint32_t size) {
    this->data.resize(size);
}

SwrContext* caiwei::media::init_audio_swr(AVCodecContext* ctx, AVFrame*) {
    SwrContext* swr = swr_alloc();
    if(swr == nullptr) {
        // SPDLOG_INFO("打开音频重采样失败");
        return nullptr;
    }
    int ret = swr_alloc_set_opts2(
        &swr,
        &audio_layout,   audio_format,    audio_sample_rate,
        &ctx->ch_layout, ctx->sample_fmt, ctx->sample_rate,
        0, nullptr
    );
    if(ret != 0) {
        swr_free(&swr);
        // SPDLOG_INFO("打开音频重采样失败：{}", ret);
        return nullptr;
    }
    ret = swr_init(swr);
    if(ret != 0) {
        swr_free(&swr);
        // SPDLOG_INFO("打开音频重采样失败：{}", ret);
        return nullptr;
    }
    return swr;
}

SwsContext* caiwei::media::init_video_sws(AVCodecContext* ctx, AVFrame* frame) {
    int  width  = ctx->width   != 0               ? ctx->width   : frame->width;
    int  height = ctx->height  != 0               ? ctx->height  : frame->height;
    auto format = ctx->pix_fmt != AV_PIX_FMT_NONE ? ctx->pix_fmt : (AVPixelFormat) frame->format;
    SwsContext* sws = sws_getContext(
        width, height, format,
        width, height, video_pix_format,
//      SWS_BILINEAR,
        SWS_FAST_BILINEAR,
        nullptr, nullptr, nullptr
    );
    if(sws == nullptr) {
        // SPDLOG_INFO("打开视频重采样失败");
        return nullptr;
    }
    return sws;
}

void caiwei::media::sws_free(SwsContext** sws) {
    if(*sws) {
        sws_freeContext(*sws);
        *sws = nullptr;
    }
}

