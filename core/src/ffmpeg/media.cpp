#include "caiwei/log.hpp"
#include "caiwei/media.hpp"

extern "C" {

#include "libavformat/avformat.h"
#include "libavdevice/avdevice.h"
    
}

static void print_all_decoder();
static void print_all_encoder();

void caiwei::media::init() {
    avformat_network_init();
    avdevice_register_all();
    print_all_decoder();
    print_all_encoder();
}

void caiwei::media::stop() {
    avformat_network_deinit();
}

static void print_all_decoder() {
    void* iter = nullptr;
    const AVCodec* codec = nullptr;
    while ((codec = av_codec_iterate(&iter))) {
        if (av_codec_is_decoder(codec)) {
            if (codec->type == AVMEDIA_TYPE_AUDIO) {
                LOG_DEBUG("audio decoder: %-16s = %s", codec->name, codec->long_name);
            } else if (codec->type == AVMEDIA_TYPE_VIDEO) {
                LOG_DEBUG("video decoder: %-16s = %s", codec->name, codec->long_name);
            } else {
                // -
            }
        }
    }
}

static void print_all_encoder() {
    void* iter = nullptr;
    const AVCodec* codec = nullptr;
    while ((codec = av_codec_iterate(&iter))) {
        if (av_codec_is_encoder(codec)) {
            if (codec->type == AVMEDIA_TYPE_AUDIO) {
                LOG_DEBUG("audio encoder: %-16s = %s", codec->name, codec->long_name);
            } else if (codec->type == AVMEDIA_TYPE_VIDEO) {
                LOG_DEBUG("video encoder: %-16s = %s", codec->name, codec->long_name);
            } else {
                // -
            }
        }
    }
}