#include "test.hpp"

#include "caiwei/type.hpp"

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
#include "rga/im2d.h"
#endif

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

extern "C" {
#include "libswscale/swscale.h"
}

struct Box {
    // clang-format off
    float x1;
    float y1;
    float x2;
    float y2;
    int   class_id;
    float score;
    // clang-format on
};

static int resize_w = 640;
static int resize_h = 640;

[[maybe_unused]]
void test_resize(const char* image_path) {
    const int w = resize_w;
    const int h = resize_h;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    int width, height, channels;
    auto data = stbi_load(image_path, &width, &height, &channels, STBI_rgb);
    CAIWEI_ASSERT(data != nullptr);
    channels = caiwei::media::IMAGE_CHANNELS;
    CAIWEI_ASSERT(caiwei::media::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale));
    std::vector<uint8_t> dst(dst_w * dst_h * channels);
    CAIWEI_ASSERT(caiwei::media::resize(data, dst.data(), width, height, dst_w, dst_h, channels));
    CAIWEI_ASSERT(stbi_write_jpg("./image_resize.jpg", dst_w, dst_h, channels, dst.data(), 80) != 0);
    CAIWEI_FOR_EACH(1000)
    caiwei::media::resize(data, dst.data(), width, height, dst_w, dst_h, channels);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
[[maybe_unused]]
void test_resize_rga(const char* image_path) {
    const int w = resize_w;
    const int h = resize_h;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    int width, height, channels;
    auto data = stbi_load(image_path, &width, &height, &channels, STBI_rgb);
    CAIWEI_ASSERT(data != nullptr);
    channels = caiwei::media::IMAGE_CHANNELS;
    CAIWEI_ASSERT(caiwei::media::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale));
    std::vector<uint8_t> dst(dst_w * dst_h * channels);
    // clang-format off
    rga_buffer_t src_img = wrapbuffer_virtualaddr(data,       width, height, RK_FORMAT_RGB_888);
    rga_buffer_t dst_img = wrapbuffer_virtualaddr(dst.data(), dst_w, dst_h,  RK_FORMAT_RGB_888);
    // clang-format on
    CAIWEI_ASSERT(imresize(src_img, dst_img, scale, scale) == IM_STATUS_SUCCESS);
    CAIWEI_ASSERT(stbi_write_jpg("./image_resize_rga.jpg", dst_w, dst_h, channels, dst.data(), 80) != 0);
    CAIWEI_FOR_EACH(1000)
    imresize(src_img, dst_img, scale, scale);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}
#endif

[[maybe_unused]]
void test_resize_ffmpeg(const char* image_path) {
    const int w = resize_w;
    const int h = resize_h;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    int width, height, channels;
    auto data = stbi_load(image_path, &width, &height, &channels, STBI_rgb);
    CAIWEI_ASSERT(data != nullptr);
    channels = caiwei::media::IMAGE_CHANNELS;
    CAIWEI_ASSERT(caiwei::media::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale));
    // clang-format off
    SwsContext* sws = sws_getContext(
        width, height, AV_PIX_FMT_RGB24,
        dst_w, dst_h,  AV_PIX_FMT_RGB24,
//      SWS_BILINEAR,
        SWS_FAST_BILINEAR,
        nullptr, nullptr, nullptr
    );
    // clang-format on
    CAIWEI_ASSERT(sws != nullptr);
    std::vector<uint8_t> dst(dst_w * dst_h * channels);
    int src_stride = width * channels;
    int dst_stride = dst_w * channels;
    auto dst_data = dst.data();
    CAIWEI_ASSERT(sws_scale(sws, &data, &src_stride, 0, height, &dst_data, &dst_stride) == dst_h);
    CAIWEI_ASSERT(stbi_write_jpg("./image_resize_ffmpeg.jpg", dst_w, dst_h, channels, dst.data(), 80) != 0);
    CAIWEI_FOR_EACH(1000)
    sws_scale(sws, &data, &src_stride, 0, height, &dst_data, &dst_stride);
    CAIWEI_FOR_EACH_END
    sws_freeContext(sws);
    sws = nullptr;
}

[[maybe_unused]]
void test_resize_pad(const char* image_path) {
    const int w = resize_w;
    const int h = resize_h;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    int width, height, channels;
    auto data = stbi_load(image_path, &width, &height, &channels, STBI_rgb);
    CAIWEI_ASSERT(data != nullptr);
    channels = caiwei::media::IMAGE_CHANNELS;
    CAIWEI_ASSERT(caiwei::media::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale));
    // clang-format off
    std::vector<uint8_t> dst(dst_w * dst_h * channels);
    std::vector<uint8_t> pad(    w *     h * channels, caiwei::media::DEFAULT_PADDING);
    // clang-format on
    CAIWEI_ASSERT(caiwei::media::resize(data, dst.data(), width, height, dst_w, dst_h, channels));
    CAIWEI_ASSERT(caiwei::media::padding(dst.data(), pad.data(), dst_w, dst_h, pad_w, pad_h, w, h, channels));
    CAIWEI_ASSERT(stbi_write_jpg("./image_resize_pad.jpg", w, h, channels, pad.data(), 80) != 0);
    CAIWEI_FOR_EACH(1000)
    caiwei::media::resize(data, dst.data(), width, height, dst_w, dst_h, channels);
    caiwei::media::padding(dst.data(), pad.data(), dst_w, dst_h, pad_w, pad_h, w, h, channels);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
[[maybe_unused]]
void test_resize_pad_rga(const char* image_path) {
    const int w = resize_w;
    const int h = resize_h;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    int width, height, channels;
    auto data = stbi_load(image_path, &width, &height, &channels, STBI_rgb);
    CAIWEI_ASSERT(data != nullptr);
    channels = caiwei::media::IMAGE_CHANNELS;
    CAIWEI_ASSERT(caiwei::media::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale));
    std::vector<uint8_t> pad(w * h * channels, caiwei::media::DEFAULT_PADDING);
    // clang-format off
    rga_buffer_t src_img = wrapbuffer_virtualaddr(data,       width, height, RK_FORMAT_RGB_888);
    rga_buffer_t dst_img = wrapbuffer_virtualaddr(pad.data(), w,     h,      RK_FORMAT_RGB_888);
    rga_buffer_t pat_img = wrapbuffer_virtualaddr(nullptr,    0,     0,      0                );
    im_rect src_rect = { 0,     0,     width, height };
    im_rect dst_rect = { pad_w, pad_h, dst_w, dst_h  };
    im_rect pat_rect = { 0,     0,     0,     0      };
    // clang-format on
    CAIWEI_ASSERT(improcess(src_img, dst_img, pat_img, src_rect, dst_rect, pat_rect, IM_SYNC) == IM_STATUS_SUCCESS);
    CAIWEI_ASSERT(stbi_write_jpg("./image_resize_pad_rga.jpg", w, h, channels, pad.data(), 80) != 0);
    CAIWEI_FOR_EACH(1000)
    improcess(src_img, dst_img, pat_img, src_rect, dst_rect, pat_rect, IM_SYNC);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}
#endif

[[maybe_unused]]
void test_nms_boxes() {
    // clang-format off
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
    // clang-format on
    auto ret = caiwei::media::nms_boxes(boxes, 0.5F);
    CAIWEI_ASSERT(ret.size() == 6);
    CAIWEI_ASSERT(ret[0] == 2);
    CAIWEI_ASSERT(ret[1] == 6);
    CAIWEI_ASSERT(ret[2] == 3);
    CAIWEI_ASSERT(ret[3] == 7);
    CAIWEI_ASSERT(ret[4] == 0);
    CAIWEI_ASSERT(ret[5] == 4);
    CAIWEI_FOR_EACH(1000)
    caiwei::media::nms_boxes(boxes, 0.5F);
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_transpose() {
    std::vector<uint8_t> src_vector(84 * 8400);
    std::vector<uint8_t> dst_vector(84 * 8400);
    std::vector<uint8_t> ret_vector(84 * 8400);
    uint8_t* src = src_vector.data();
    uint8_t* dst = dst_vector.data();
    uint8_t* ret = ret_vector.data();
    std::iota(src, src + 84 * 8400, 1);
    caiwei::media::transpose(src, dst, 84, 8400);
    CAIWEI_ASSERT(!std::equal(dst, dst + 84 * 8400, src));
    caiwei::media::transpose(dst, ret, 8400, 84);
    CAIWEI_ASSERT(std::equal(ret, ret + 84 * 8400, src));
    CAIWEI_FOR_EACH(1000)
    caiwei::media::transpose(src, dst, 84, 8400);
    caiwei::media::transpose(dst, ret, 8400, 84);
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_hwc_to_chw(const char* image_path) {
    // clang-format off
    std::vector<float> hwc_vec = {
        10, 20, 30,
        11, 21, 31,
        12, 22, 32,
        13, 23, 33
    };
    // clang-format on
    std::vector<float> chw_vec(hwc_vec.size());
    // clang-format off
    std::vector<float> chw_true = {
        10, 11, 12, 13,
        20, 21, 22, 23,
        30, 31, 32, 33
    };
    // clang-format on
    caiwei::media::hwc_to_chw(hwc_vec.data(), chw_vec.data(), 1, 4, 3);
    CAIWEI_ASSERT((chw_vec == chw_true));
    int width, height, channels;
    auto data = stbi_load(image_path, &width, &height, &channels, STBI_rgb);
    CAIWEI_ASSERT(data != nullptr);
    channels = caiwei::media::IMAGE_CHANNELS;
    // clang-format off
    std::vector<float>  hwc   (width * height * channels);
    std::vector<float>  chw   (width * height * channels);
    std::vector<int8_t> chw_i8(width * height * channels);
    // clang-format on
    caiwei::type::u8_to_f32(data, width * height * channels, hwc.data(), 255.0F);
    caiwei::media::hwc_to_chw(hwc.data(), chw.data(), height, width, channels);
    CAIWEI_ASSERT(stbi_write_hdr("./image_chw.hdr", width, height, channels, chw.data()) != 0);
    CAIWEI_ASSERT(stbi_write_hdr("./image_hwc.hdr", width, height, channels, hwc.data()) != 0);
    {
        CAIWEI_FOR_EACH(1000)
        caiwei::media::hwc_to_chw(hwc.data(), chw.data(), height, width, channels);
        CAIWEI_FOR_EACH_END
    }
    {
        CAIWEI_FOR_EACH(1000)
        caiwei::media::hwc_to_chw_u8_i8(data, chw_i8.data(), height, width, channels);
        CAIWEI_FOR_EACH_END
    }
    {
        CAIWEI_FOR_EACH(1000)
        caiwei::media::hwc_to_chw_u8_f32(data, chw.data(), height, width, channels, 255.0F);
        CAIWEI_FOR_EACH_END
    }
    stbi_image_free(data);
}

[[maybe_unused]]
void test_image(const char* image_path) {
    const int w = resize_w;
    const int h = resize_h;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    int width, height, channels;
    auto data = stbi_load(image_path, &width, &height, &channels, STBI_rgb);
    CAIWEI_ASSERT(data != nullptr);
    channels = caiwei::media::IMAGE_CHANNELS;
    CAIWEI_ASSERT(caiwei::media::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale));
    // clang-format off
    std::vector<uint8_t> dst(dst_w * dst_h * channels);
    std::vector<uint8_t> pad(w     * h     * channels);
    std::vector<float>   hwc(w     * h     * channels);
    std::vector<float>   chw(w     * h     * channels);
    // clang-format on
    {
        CAIWEI_FOR_EACH(1000)
        caiwei::media::resize(data, dst.data(), width, height, dst_w, dst_h, channels);
        caiwei::media::padding(dst.data(), pad.data(), dst_w, dst_h, pad_w, pad_h, w, h, channels);
        caiwei::type::u8_to_f32(pad.data(), w * h * channels, hwc.data(), 255.0F);
        caiwei::media::hwc_to_chw(hwc.data(), chw.data(), h, w, channels);
        CAIWEI_FOR_EACH_END
    }
    {
        CAIWEI_FOR_EACH(1000)
        caiwei::media::resize(data, dst.data(), width, height, dst_w, dst_h, channels);
        caiwei::media::padding(dst.data(), pad.data(), dst_w, dst_h, pad_w, pad_h, w, h, channels);
        caiwei::media::hwc_to_chw_u8_f32(pad.data(), chw.data(), h, w, channels, 255.0F);
        CAIWEI_FOR_EACH_END
    }
    stbi_image_free(data);
}

int main(int argc, char* argv[]) {
    const char* image_path;
    if (argc > 1) {
        image_path = argv[1];
    } else {
        image_path = "./acgist.jpg";
    }
    caiwei::test::init_test();
    test_resize(image_path);
#ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
    test_resize_rga(image_path);
#endif
    test_resize_ffmpeg(image_path);
    test_resize_pad(image_path);
#ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
    test_resize_pad_rga(image_path);
#endif
    test_nms_boxes();
    test_transpose();
    test_hwc_to_chw(image_path);
    test_image(image_path);
    caiwei::test::stop_test();
    return 0;
}
