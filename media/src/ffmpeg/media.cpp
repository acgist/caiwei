#include "caiwei/log.hpp"
#include "caiwei/media.hpp"

#include <filesystem>

extern "C" {

#include "libavformat/avformat.h"
#include "libavdevice/avdevice.h"
    
}

static void print_all_decoder();
static void print_all_encoder();
static void delete_old_sdp_file();

void caiwei::media::init() {
    avformat_network_init();
    avdevice_register_all();
    print_all_decoder();
    print_all_encoder();
    delete_old_sdp_file();
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
                CW_LOG_D("audio decoder: %-16s = %s", codec->name, codec->long_name);
            } else if (codec->type == AVMEDIA_TYPE_VIDEO) {
                CW_LOG_D("video decoder: %-16s = %s", codec->name, codec->long_name);
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
                CW_LOG_D("audio encoder: %-16s = %s", codec->name, codec->long_name);
            } else if (codec->type == AVMEDIA_TYPE_VIDEO) {
                CW_LOG_D("video encoder: %-16s = %s", codec->name, codec->long_name);
            } else {
                // -
            }
        }
    }
}

static void delete_old_sdp_file() {
    std::filesystem::path sdp_parent = "./sdp/";
    if (!std::filesystem::exists(sdp_parent)) {
        return;
    }
    for (auto& entry : std::filesystem::directory_iterator(sdp_parent)) {
        if (entry.is_regular_file() && entry.path().extension() == ".sdp") {
            CW_LOG_I("删除旧的SDP文件: %s", entry.path().string().c_str());
            std::filesystem::remove(entry.path());
        }
    }
}
