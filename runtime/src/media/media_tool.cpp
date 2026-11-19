#include "nlohmann/json.hpp"

#include <vector>
#include <algorithm>

#include "caiwei/log.hpp"
#include "caiwei/media_tool.hpp"

#if ENABLE_CAIWEI_BACKEND_RKNN
#include "rga/im2d.h"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STBIR_DEFAULT_FILTER_DOWNSAMPLE STBIR_FILTER_BOX
#define STB_IMAGE_RESIZE2_IMPLEMENTATION
#include "stb/stb_image_resize2.h"

#if ENABLE_CAIWEI_TEST
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#endif

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

namespace caiwei {
namespace media  {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Cls,                 class_id, score);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Box, x1, y1, x2, y2, class_id, score);

}
}

std::string caiwei::media::to_json(const caiwei::media::Cls& cls) {
    return nlohmann::json(cls).dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

std::string caiwei::media::to_json(const caiwei::media::Box& box) {
    return nlohmann::json(box).dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

std::string caiwei::media::to_json(const std::vector<Cls>& cls) {
    nlohmann::json ret = cls;
    return ret.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

std::string caiwei::media::to_json(const std::vector<Box>& box) {
    nlohmann::json ret = box;
    return ret.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

bool caiwei::media::resize(
    const uint8_t* __restrict src,
          uint8_t* __restrict dst,
    const int src_w,
    const int src_h,
    const int dst_w,
    const int dst_h,
    const int channels
) {
    if (src_h == dst_h && src_w == dst_w) {
        std::copy_n(src, static_cast<size_t>(src_w) * src_h * channels, dst);
        return true;
    }
    if (channels == 3) {
        if (stbir_resize_uint8_linear(src, src_w, src_h, 0, dst, dst_w, dst_h, 0, STBIR_RGB) == nullptr) {
            return false;
        }
    } else if (channels == 4) {
        if (stbir_resize_uint8_linear(src, src_w, src_h, 0, dst, dst_w, dst_h, 0, STBIR_RGBA) == nullptr) {
            return false;
        }
    } else {
        CW_LOG_W("不支持的图片通道: %d", channels);
        return false;
    }
    return true;
}

#if ENABLE_CAIWEI_BACKEND_RKNN
bool caiwei::media::resize(
    const uint8_t* src,
          uint8_t* dst,
    const int src_w,
    const int src_h,
    const int w,
    const int h,
    const int dst_w,
    const int dst_h,
    const int pad_w,
    const int pad_h,
    const int channels
) {
    RgaSURF_FORMAT image_type;
    if (channels == 3) {
        image_type = RK_FORMAT_RGB_888;
    } else if (channels == 4) {
        image_type = RK_FORMAT_RGBA_8888;
    } else {
        CW_LOG_W("不支持的图片通道: %d", channels);
        return false;
    }
    if (src_h == dst_h && src_w == dst_w && pad_w == 0 && pad_h == 0) {
        std::copy_n(src, static_cast<size_t>(src_w) * src_h * channels, dst);
        return true;
    }
    if (src_w % 16 != 0) {
        CW_LOG_D("图片宽度必须是16的倍数: %d", src_w);
        thread_local std::vector<uint8_t> v;
        v.resize(static_cast<size_t>(dst_w) * dst_h * channels);
        if (!caiwei::media::resize(src, v.data(), src_w, src_h, dst_w, dst_h, channels)) {
            return false;
        }
        if (!caiwei::media::padding(v.data(), dst, dst_w, dst_h, pad_w, pad_h, w, h, channels)) {
            return false;
        }
        return true;
    }
    rga_buffer_t src_img = wrapbuffer_virtualaddr((void*) src, src_w, src_h, image_type);
    rga_buffer_t dst_img = wrapbuffer_virtualaddr(dst,         w,     h,     image_type);
    rga_buffer_t pat_img = wrapbuffer_virtualaddr(nullptr,     0,     0,     0         );
    im_rect src_rect = { 0,     0,     src_w, src_h };
    im_rect dst_rect = { pad_w, pad_h, dst_w, dst_h };
    im_rect pat_rect = { 0,     0,     0,     0     };
    const auto status = improcess(src_img, dst_img, pat_img, src_rect, dst_rect, pat_rect, IM_SYNC);
    if (status != IM_STATUS_SUCCESS) {
        CW_LOG_W("图片处理失败: %d", status);
        return false;
    } else {
        return true;
    }
}
#endif
