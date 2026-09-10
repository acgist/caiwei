#include "test.hpp"

#include "caiwei/type.hpp"
#include "caiwei/image_tool.hpp"

#include <cassert>
#include <numeric>

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
#include "rga/im2d.h"
#endif

extern "C" {
#include "libswscale/swscale.h"
}

struct Box {
    float x1; // 左上角x百分比
    float y1; // 左上角y百分比
    float x2; // 右下角x百分比
    float y2; // 右下角y百分比
    int   class_id; // 类别ID
    float score;    // 置信度
};

static int resize_w = 640;
static int resize_h = 640;

#define IMAGE_PATH "./acgist.jpg"
// #define IMAGE_PATH "./caiwei.jpg"
// #define IMAGE_PATH "./caiwei_h.jpg"
// #define IMAGE_PATH "./caiwei_w.jpg"

[[maybe_unused]]
void test_load() {
    CAIWEI_FOR_EACH(100)
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    stbi_image_free(data);
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_crop() {
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    std::vector<uint8_t> dst(256 * 256 * channels);
    caiwei::image::crop(data, width, height, dst.data(), 256, 256, 256, 256, channels);
    stbi_write_jpg("./image_crop.jpg", 256, 256, channels, dst.data(), 80);
    CAIWEI_FOR_EACH(1000)
    caiwei::image::crop(data, width, height, dst.data(), 256, 256, 256, 256, channels);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

[[maybe_unused]]
void test_resize() {
    // TODO: < 1 ms
    const int w = resize_w;
    const int h = resize_h;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    caiwei::image::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale);
    std::vector<uint8_t> dst(dst_w * dst_h * channels);
    caiwei::image::resize(data, dst.data(), width, height, dst_w, dst_h);
    stbi_write_jpg("./image_resize.jpg", dst_w, dst_h, channels, dst.data(), 80);
    CAIWEI_FOR_EACH(1000)
    caiwei::image::resize(data, dst.data(), width, height, dst_w, dst_h);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
[[maybe_unused]]
void test_resize_rga() {
    const int w = resize_w;
    const int h = resize_h;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    caiwei::image::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale);
    std::vector<uint8_t> dst(dst_w * dst_h * channels);
    rga_buffer_t src_img = wrapbuffer_virtualaddr(data, width, height, RK_FORMAT_RGB_888);
    rga_buffer_t dst_img = wrapbuffer_virtualaddr(dst.data(), dst_w, dst_h, RK_FORMAT_RGB_888);
    imresize(src_img, dst_img, scale, scale);
    stbi_write_jpg("./image_resize_rga.jpg", dst_w, dst_h, channels, dst.data(), 80);
    CAIWEI_FOR_EACH(1000)
    imresize(src_img, dst_img, scale, scale);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}
#endif

[[maybe_unused]]
void test_resize_ffmpeg() {
    const int w = resize_w;
    const int h = resize_h;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    caiwei::image::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale);
    SwsContext* sws = sws_getContext(
        width, height, AV_PIX_FMT_RGB24,
        dst_w, dst_h,  AV_PIX_FMT_RGB24,
//      SWS_BILINEAR,
        SWS_FAST_BILINEAR,
        nullptr, nullptr, nullptr
    );
    std::vector<uint8_t> dst(dst_w * dst_h * channels);
    int src_stride = width * 3;
    int dst_stride = dst_w * 3;
    auto dst_data = dst.data();
    {
        int ret = sws_scale(sws, &data, &src_stride, 0, height, &dst_data, &dst_stride);
        stbi_write_jpg("./image_ffmpeg_resize.jpg", dst_w, dst_h, channels, dst.data(), 80);
    }
    CAIWEI_FOR_EACH(1000)
    int ret = sws_scale(sws, &data, &src_stride, 0, height, &dst_data, &dst_stride);
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_resize_pad() {
    // TODO: < 1 ms
    const int w = resize_w;
    const int h = resize_h;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    caiwei::image::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale);
    std::vector<uint8_t> dst(dst_w * dst_h * channels);
    std::vector<uint8_t> pad(    w *     h * channels, caiwei::image::DEFAULT_PADDING);
    caiwei::image::resize (data,       dst.data(), width, height, dst_w, dst_h);
    caiwei::image::padding(dst.data(), pad.data(), dst_w, dst_h,  pad_w, pad_h, w, h);
    stbi_write_jpg("./image_resize_pad.jpg", w, h, channels, pad.data(), 80);
    CAIWEI_FOR_EACH(1000)
    caiwei::image::resize (data,       dst.data(), width, height, dst_w, dst_h);
    caiwei::image::padding(dst.data(), pad.data(), dst_w, dst_h,  pad_w, pad_h, w, h);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
[[maybe_unused]]
void test_resize_pad_rga() {
    const int w = resize_w;
    const int h = resize_h;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    caiwei::image::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale);
    std::vector<uint8_t> pad(w * h * channels, caiwei::image::DEFAULT_PADDING);
    rga_buffer_t src_img = wrapbuffer_virtualaddr(data, width, height, RK_FORMAT_RGB_888);
    rga_buffer_t dst_img = wrapbuffer_virtualaddr(pad.data(), w, h, RK_FORMAT_RGB_888);
    rga_buffer_t pat_img = wrapbuffer_virtualaddr(nullptr, 0, 0, 0);
    im_rect src_rect = { 0, 0, width, height };
    im_rect dst_rect = { pad_w, pad_h, dst_w, dst_h };
    im_rect pat_rect = { 0, 0, 0, 0 };
    improcess(src_img, dst_img, pat_img, src_rect, dst_rect, pat_rect, IM_SYNC);
    stbi_write_jpg("./image_resize_pad_rga.jpg", w, h, channels, pad.data(), 80);
    CAIWEI_FOR_EACH(1000)
    improcess(src_img, dst_img, {}, src_rect, dst_rect, {}, IM_SYNC);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}
#endif

[[maybe_unused]]
void test_nms_boxes() {
    // x1y1x2y2
    std::vector<Box> boxes = {
        {  2,  2,   8,   8, 0, 0.80F },
        { 10, 10, 120, 120, 0, 0.80F },
        { 20, 20, 140, 140, 0, 0.90F },
        { 20, 20, 140, 140, 1, 0.85F },
        {  2 / 100.0F,  2 / 100.0F,   8 / 100.0F,   8 / 100.0F, 0, 0.80F },
        { 10 / 100.0F, 10 / 100.0F, 120 / 100.0F, 120 / 100.0F, 0, 0.80F },
        { 20 / 100.0F, 20 / 100.0F, 140 / 100.0F, 140 / 100.0F, 0, 0.90F },
        { 20 / 100.0F, 20 / 100.0F, 140 / 100.0F, 140 / 100.0F, 1, 0.85F },
    };
    auto ret = caiwei::image::nms_boxes(boxes, 0.5F);
    assert(ret.size() == 6);
    assert(ret[0] == 2);
    assert(ret[1] == 6);
    assert(ret[2] == 3);
    assert(ret[3] == 7);
    assert(ret[4] == 0);
    assert(ret[5] == 4);
    CAIWEI_FOR_EACH(1000)
    caiwei::image::nms_boxes(boxes, 0.5F);
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_min_max_loc() {
    float scores[128];
    std::iota(scores, scores + 128, 1.0F);
    scores[100] = 10000.0F;
    int   max_index;
    float max_score;
    caiwei::image::max_loc(scores, 128, max_score, max_index);
    assert(max_score == 10000.0F);
    assert(max_index == 100);
    CAIWEI_FOR_EACH(1000)
    caiwei::image::max_loc(scores, 128, max_score, max_index);
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_transpose() {
    uint8_t* src = new uint8_t[84 * 8400];
    uint8_t* dst = new uint8_t[84 * 8400];
    uint8_t* ret = new uint8_t[84 * 8400];
    std::iota(src, src + 84 * 8400, 1);
    caiwei::image::transpose(src, dst, 84, 8400);
    assert(!std::equal(dst, dst + 84 * 8400, src));
    caiwei::image::transpose(dst, ret, 8400, 84);
    assert(std::equal(ret, ret + 84 * 8400, src));
    CAIWEI_FOR_EACH(1000)
    caiwei::image::transpose(src, dst, 84, 8400);
    caiwei::image::transpose(dst, ret, 8400, 84);
    CAIWEI_FOR_EACH_END
    delete[] src;
    delete[] dst;
    delete[] ret;
}

[[maybe_unused]]
void test_hwc_to_chw() {
    // TODO: < 2 ms
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    std::vector<float> hwc(width * height * channels);
    std::vector<float> chw(width * height * channels);
    caiwei::type::i8_to_f32(data, width * height * channels, hwc.data(), 255.0F);
    caiwei::image::hwc_to_chw(hwc.data(), chw.data(), height, width, channels);
    caiwei::image::chw_to_hwc(chw.data(), hwc.data(), height, width, channels);
    stbi_write_hdr("./image_blob_from_image.hdr", width, height, channels, hwc.data());
    CAIWEI_FOR_EACH(1000)
    caiwei::image::hwc_to_chw(hwc.data(), chw.data(), height, width, channels);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

int main() {
    caiwei::test::init_test();
    test_load();
    test_crop();
    test_resize();
    #ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
    test_resize_rga();
    #endif
    test_resize_ffmpeg();
    test_resize_pad();
    #ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
    test_resize_pad_rga();
    #endif
    test_nms_boxes();
    test_min_max_loc();
    test_transpose();
    test_hwc_to_chw();
    caiwei::test::stop_test();
    return 0;
}
