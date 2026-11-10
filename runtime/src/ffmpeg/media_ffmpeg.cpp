#include <atomic>
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

static std::atomic<bool> ffmpeg_init(false);

void caiwei::media::init() {
    if (!ffmpeg_init) {
        int ret = avformat_network_init();
        if (ret == 0) {
            CW_LOG_I("FFmpeg网络加载成功: %d", ret);
        } else {
            CW_LOG_W("FFmpeg网络加载失败: %d", ret);
        }
        avdevice_register_all();
        ffmpeg_init = true;
    }
    #if CAIWEI_DEBUG
    print_all_decoder();
    print_all_encoder();
    print_all_protocols();
    #endif
    delete_old_sdp_file();
}

void caiwei::media::stop() {
    if (ffmpeg_init) {
        int ret = avformat_network_deinit();
        if (ret == 0) {
            CW_LOG_I("FFmpeg网络卸载成功: %d", ret);
        } else {
            CW_LOG_W("FFmpeg网络卸载失败: %d", ret);
        }
        ffmpeg_init = false;
    }
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
    std::error_code error_code;
    if (!std::filesystem::is_directory(caiwei::media::sdp_file_path, error_code)) {
        CW_LOG_W("遍历旧的SDP文件失败: %s", error_code.message().c_str());
        return;
    }
    auto iter = std::filesystem::directory_iterator(caiwei::media::sdp_file_path, error_code);
    if (error_code) {
        CW_LOG_W("遍历旧的SDP文件失败: %s", error_code.message().c_str());
        return;
    }
    for (auto& entry : iter) {
        if (!entry.is_regular_file(error_code)) {
            if (error_code) {
                CW_LOG_W("删除旧的SDP文件失败: %s = %s", entry.path().string().c_str(), error_code.message().c_str());
            } else {
                CW_LOG_W("删除旧的SDP文件失败: %s", entry.path().string().c_str());
            }
            continue;
        }
        if (entry.path().extension() == ".sdp") {
            if (std::filesystem::remove(entry.path(), error_code)) {
                CW_LOG_I("删除旧的SDP文件成功: %s", entry.path().string().c_str());
            } else {
                CW_LOG_W("删除旧的SDP文件失败: %s = %s", entry.path().string().c_str(), error_code.message().c_str());
            }
        }
    }
}
