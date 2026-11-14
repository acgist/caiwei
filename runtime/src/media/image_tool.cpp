#include "caiwei/log.hpp"
#include "caiwei/image_tool.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STBIR_DEFAULT_FILTER_DOWNSAMPLE STBIR_FILTER_BOX
#define STB_IMAGE_RESIZE2_IMPLEMENTATION
#include "stb/stb_image_resize2.h"

#if ENABLE_CAIWEI_TEST
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#endif

#if ENABLE_CAIWEI_BACKEND_RKNN
#include "rga/im2d.h"
#endif

#if ENABLE_CAIWEI_BACKEND_RKNN
bool caiwei::image::resize(
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
        if (!caiwei::image::resize(src, v.data(), src_w, src_h, dst_w, dst_h, channels)) {
            return false;
        }
        if (!caiwei::image::padding(v.data(), dst, dst_w, dst_h, pad_w, pad_h, w, h, channels)) {
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
