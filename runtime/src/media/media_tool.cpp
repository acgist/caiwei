#include "caiwei/log.hpp"
#include "caiwei/media_tool.hpp"

extern "C" {
#include "libavutil/pixfmt.h"
#include "libavutil/samplefmt.h"
}

caiwei::media::AudioInfo::AudioInfo(
    int channels, int sample_rate, int format
) : channels(channels), sample_rate(sample_rate), format(format) {
    if (channels < AUDIO_MIN_CHANNELS || channels > AUDIO_MAX_CHANNELS) {
        CW_LOG_W("音频通道数错误: %d", channels);
    }
    if (sample_rate < AUDIO_MIN_SAMPLE_RATE || sample_rate > AUDIO_MAX_SAMPLE_RATE) {
        CW_LOG_W("音频采样率错误: %d", sample_rate);
    }
    if (format < AV_SAMPLE_FMT_U8 || format >= AV_SAMPLE_FMT_NB) {
        CW_LOG_W("音频格式错误: %d", format);
    } else {
        this->bytes_per_sample = av_get_bytes_per_sample((AVSampleFormat) format);
    }
}

caiwei::media::VideoInfo::VideoInfo(
    int width, int height, int format
) : fps(0), width(width), height(height), format(format) {
    // 高度可以为零自动计算
    if (width < VIDEO_MIN_WIDTH || width > VIDEO_MAX_WIDTH || height < VIDEO_MIN_HEIGHT || height > VIDEO_MAX_HEIGHT) {
        CW_LOG_W("视频尺寸错误: %d x %d", width, height);
    }
    if (format < AV_PIX_FMT_YUV420P || format >= AV_PIX_FMT_NB) {
        CW_LOG_W("视频格式错误: %d", format);
    }
}

caiwei::media::VideoInfo::VideoInfo(
    int fps, int width, int height, int format
) : fps(fps), width(width), height(height), format(format) {
    if (fps < VIDEO_MIN_FPS || fps > VIDEO_MAX_FPS) {
        CW_LOG_W("视频帧率错误: %d", fps);
    }
    // 高度可以为零自动计算
    if (width < VIDEO_MIN_WIDTH || width > VIDEO_MAX_WIDTH || height < VIDEO_MIN_HEIGHT || height > VIDEO_MAX_HEIGHT) {
        CW_LOG_W("视频尺寸错误: %d x %d", width, height);
    }
    if (format < AV_PIX_FMT_YUV420P || format >= AV_PIX_FMT_NB) {
        CW_LOG_W("视频格式错误: %d", format);
    }
}
