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

#if defined(ENABLE_CAIWEI_RUNTIME_RKNN2) || defined(ENABLE_CAIWEI_RUNTIME_RKNN3)
#include "rga/im2d.hpp"
#endif

#if defined(ENABLE_CAIWEI_RUNTIME_RKNN2) || defined(ENABLE_CAIWEI_RUNTIME_RKNN3)
bool caiwei::image::resize(
    const uint8_t* src,
          uint8_t* dst,
    const int width,
    const int height,
    const int w,
    const int h,
    const int dst_w,
    const int dst_h,
    const int pad_w,
    const int pad_h
) {
    rga_buffer_t src_img = wrapbuffer_virtualaddr((void*) src, width, height, RK_FORMAT_RGB_888);
    rga_buffer_t dst_img = wrapbuffer_virtualaddr(dst,         w,     h,      RK_FORMAT_RGB_888);
    rga_buffer_t pat_img = wrapbuffer_virtualaddr(nullptr,     0,     0,      0                );
    im_rect src_rect = { 0,     0,     width, height };
    im_rect dst_rect = { pad_w, pad_h, dst_w, dst_h  };
    im_rect pat_rect = { 0,     0,     0,     0      };
    const auto status = improcess(src_img, dst_img, pat_img, src_rect, dst_rect, pat_rect, IM_SYNC);
    if (status != IM_STATUS_SUCCESS) {
        CW_LOG_W("图片处理失败: %d", status);
        return false;
    } else {
        return true;
    }
}
#endif
