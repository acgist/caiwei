#include <filesystem>

#include "caiwei/log.hpp"
#include "caiwei/media_tool.hpp"

extern "C" {
#include "libavformat/avio.h"
#include "libavformat/avformat.h"
#include "libavdevice/avdevice.h"
}

#if CAIWEI_DEBUG
static void print_all_decoder();
static void print_all_encoder();
static void print_all_protocols();
#endif
static void delete_old_sdp_file();

void caiwei::media::init() {
    avformat_network_init();
    avdevice_register_all();
    #if CAIWEI_DEBUG
    print_all_decoder();
    print_all_encoder();
    print_all_protocols();
    #endif
    delete_old_sdp_file();
}

void caiwei::media::stop() {
    avformat_network_deinit();
}

#if CAIWEI_DEBUG
static void print_all_decoder() {
    void* iter = nullptr;
    const AVCodec* codec = nullptr;
    while ((codec = av_codec_iterate(&iter))) {
        if (av_codec_is_decoder(codec)) {
            if (codec->type == AVMEDIA_TYPE_AUDIO) {
                CW_LOG_D("ffmpeg audio decoder: %-16s = %s", codec->name, codec->long_name);
            } else if (codec->type == AVMEDIA_TYPE_VIDEO) {
                CW_LOG_D("ffmpeg video decoder: %-16s = %s", codec->name, codec->long_name);
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
                CW_LOG_D("ffmpeg audio encoder: %-16s = %s", codec->name, codec->long_name);
            } else if (codec->type == AVMEDIA_TYPE_VIDEO) {
                CW_LOG_D("ffmpeg video encoder: %-16s = %s", codec->name, codec->long_name);
            } else {
                // -
            }
        }
    }
}

static void print_all_protocols() {
    void* iter = nullptr;
    const char* name = nullptr;
    while ((name = avio_enum_protocols(&iter, 0)) != nullptr) {
        CW_LOG_D("ffmpeg protocol: %s", name);
    }
}
#endif

static void delete_old_sdp_file() {
    if (!std::filesystem::exists(caiwei::media::sdp_file_path)) {
        return;
    }
    for (auto& entry : std::filesystem::directory_iterator(caiwei::media::sdp_file_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".sdp") {
            if (std::filesystem::remove(entry.path())) {
                CW_LOG_I("删除旧的SDP文件成功: %s", entry.path().string().c_str());
            } else {
                CW_LOG_W("删除旧的SDP文件失败: %s", entry.path().string().c_str());
            }
        }
    }
}
