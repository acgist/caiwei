#include "test_media.hpp"

#include "caiwei/transform.hpp"

#include <numeric>

#include "opencv2/opencv.hpp"
#include "opencv2/core/utils/logger.hpp"

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

#define IMAGE_PATH "./acgist.jpg"
// #define IMAGE_PATH "./caiwei_h.jpg"
// #define IMAGE_PATH "./caiwei_w.jpg"
// #define IMAGE_PATH "./caiwei.jpg"

[[maybe_unused]]
void test_load() {
    {
        int width, height, channels;
        auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
        cv::Mat mat(height, width, CV_8UC3, data);
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        cv::imshow("loc", mat);
        cv::waitKey(0);
        stbi_image_free(data);
    }
    {
        auto mat = cv::imread(IMAGE_PATH);;
        cv::imshow("ocv", mat);
        cv::waitKey(0);
    }
}

[[maybe_unused]]
void test_loc_load() {
    // 1280 =  7 ms
    // 1920 = 18 ms
    CAIWEI_FOR_EACH(1000)
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    stbi_image_free(data);
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_ocv_load() {
    // 1280 =  5 ms
    // 1920 = 11 ms
    CAIWEI_FOR_EACH(1000)
    cv::imread(IMAGE_PATH);
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_crop() {
    {
        int width, height, channels;
        auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
        cv::Mat mat(height, width, CV_8UC3, data);
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        std::vector<uint8_t> dst(256 * 256 * channels);
        caiwei::transform::crop(data, width, height, dst.data(), 256, 256, 256, 256, channels);
        mat = cv::Mat(256, 256, CV_8UC3, dst.data()).clone();
        cv::imshow("loc", mat);
        cv::waitKey(0);
        stbi_image_free(data);
    }
    {
        auto mat = cv::imread(IMAGE_PATH);
        mat = mat(cv::Rect(256, 256, 256, 256));
        cv::imshow("ocv", mat);
        cv::waitKey(0);
    }
}

[[maybe_unused]]
void test_loc_crop() {
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    std::vector<uint8_t> dst(256 * 256 * channels);
    CAIWEI_FOR_EACH(10'000)
    caiwei::transform::crop(data, width, height, dst.data(), 256, 256, 256, 256, channels);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

[[maybe_unused]]
void test_ocv_crop() {
    auto mat = cv::imread(IMAGE_PATH);
    CAIWEI_FOR_EACH(10'000)
    auto crop = mat(cv::Rect(256, 256, 256, 256)).clone();
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_draw() {
    {
        int width, height, channels;
        auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
        caiwei::transform::draw(data, width, height, 256, 256, 256, 256);
        cv::Mat mat(height, width, CV_8UC3, data);
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        cv::imshow("loc", mat);
        cv::waitKey(0);
        stbi_image_free(data);
    }
    {
        auto mat = cv::imread(IMAGE_PATH);
        cv::rectangle(mat, cv::Point(256, 256), cv::Point(512, 512), cv::Scalar{ 0, 0, 0 });
        cv::imshow("ocv", mat);
        cv::waitKey(0);
    }
}

[[maybe_unused]]
void test_loc_draw() {
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    CAIWEI_FOR_EACH(10'000)
    caiwei::transform::draw(data, width, height, 256, 256, 256, 256);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

[[maybe_unused]]
void test_ocv_draw() {
    auto mat = cv::imread(IMAGE_PATH);
    CAIWEI_FOR_EACH(10'000)
    cv::rectangle(mat, cv::Point(256, 256), cv::Point(512, 512), cv::Scalar{ 0, 0, 0 });
    CAIWEI_FOR_EACH_END
}

static int resize_w = 640;
static int resize_h = 1280;

[[maybe_unused]]
void test_resize() {
    const int w = resize_w;
    const int h = resize_h;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    {
        int width, height, channels;
        auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
        caiwei::transform::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale);
        std::vector<uint8_t> dst(dst_w * dst_h * channels);
        caiwei::transform::resize(data, dst.data(), width, height, dst_w, dst_h);
        cv::Mat mat(dst_h, dst_w, CV_8UC3, dst.data());
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        cv::imshow("loc", mat);
        cv::waitKey(0);
        stbi_image_free(data);
    }
    {
        auto mat = cv::imread(IMAGE_PATH);
        cv::Mat dst;
        cv::resize(mat, dst, cv::Size(dst_w, dst_h));
        cv::imshow("ocv", dst);
        cv::waitKey(0);
    }
}

[[maybe_unused]]
void test_loc_resize() {
    // 1280 = 1 ms
    // 1920 = 3 ms
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    const int w = resize_w;
    const int h = resize_h;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    caiwei::transform::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale);
    CAIWEI_FOR_EACH(1000)
    std::vector<uint8_t> dst(dst_w * dst_h * channels);
    caiwei::transform::resize(data, dst.data(), width, height, dst_w, dst_h);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

[[maybe_unused]]
void test_ocv_resize() {
    auto mat = cv::imread(IMAGE_PATH);
    const int w = resize_w;
    const int h = resize_h;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    caiwei::transform::resize(mat.cols, mat.rows, w, h, dst_w, dst_h, pad_w, pad_h, scale);
    CAIWEI_FOR_EACH(1000)
    cv::Mat dst;
    cv::resize(mat, dst, cv::Size(dst_w, dst_h));
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_ffmpeg_resize() {
    // 1280 = 2 ms
    // 1920 = 1 ms
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    const int w = resize_w;
    const int h = resize_h;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    caiwei::transform::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale);
    SwsContext* sws = sws_getContext(
        width, height, AV_PIX_FMT_RGB24,
        dst_w, dst_h,  AV_PIX_FMT_RGB24,
        SWS_FAST_BILINEAR,
        nullptr, nullptr, nullptr
    );
    CAIWEI_FOR_EACH(1000)
    int src_stride = width * 3;
    int dst_stride = dst_w * 3;
    std::vector<uint8_t> dst(dst_w * dst_h * channels);
    auto dst_data = dst.data();
    int ret = sws_scale(sws, &data, &src_stride, 0, height, &dst_data, &dst_stride);
    // if (caiwei_index == 0) {
    //     cv::Mat mat(dst_h, dst_w, CV_8UC3, dst.data());
    //     cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    //     cv::imshow("ffmpeg", mat);
    //     cv::waitKey(0);
    // }
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_resize_pad() {
    int width, height, channels;
    const int w = resize_w;
    const int h = resize_h;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    {
        auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
        caiwei::transform::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale);
        std::vector<uint8_t> dst(dst_w * dst_h * channels);
        std::vector<uint8_t> pad(w     * h     * channels);
        caiwei::transform::resize (data,       dst.data(), width, height, dst_w, dst_h);
        caiwei::transform::padding(dst.data(), pad.data(), dst_w, dst_h,  pad_w, pad_h, w, h);
        cv::Mat mat(h, w, CV_8UC3, pad.data());
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        cv::imshow("loc", mat);
        cv::waitKey(0);
        stbi_image_free(data);
    }
    {
        auto mat = cv::imread(IMAGE_PATH);
        cv::Mat dst;
        cv::resize(mat, dst, cv::Size(dst_w, dst_h));
        cv::Mat result = cv::Mat::zeros(h, w, CV_8UC3);
        dst.copyTo(result(cv::Rect(pad_w, pad_h, dst_w, dst_h)));
        cv::imshow("ocv", result);
        cv::waitKey(0);
    }
}

[[maybe_unused]]
void test_loc_resize_pad() {
    // 1280 = 2 ms
    // 1920 = 3 ms
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    const int w = resize_w;
    const int h = resize_h;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    caiwei::transform::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale);
    CAIWEI_FOR_EACH(1000)
    std::vector<uint8_t> dst(dst_w * dst_h * channels);
    std::vector<uint8_t> pad(w     * h     * channels);
    caiwei::transform::resize (data,       dst.data(), width, height, dst_w, dst_h);
    caiwei::transform::padding(dst.data(), pad.data(), dst_w, dst_h,  pad_w, pad_h, w, h);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

[[maybe_unused]]
void test_ocv_resize_pad() {
    auto mat = cv::imread(IMAGE_PATH);
    const int w = resize_w;
    const int h = resize_h;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    caiwei::transform::resize(mat.cols, mat.rows, w, h, dst_w, dst_h, pad_w, pad_h, scale);
    CAIWEI_FOR_EACH(1000)
    cv::Mat dst;
    cv::resize(mat, dst, cv::Size(dst_w, dst_h));
    cv::Mat result = cv::Mat::zeros(h, w, CV_8UC3);
    dst.copyTo(result(cv::Rect(pad_w, pad_h, dst_w, dst_h)));
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_nms_boxes() {
    {
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
        auto ret = caiwei::transform::nms_boxes(boxes, 0.5F);
        assert(ret.size() == 6);
        assert(ret[0].x1 == boxes[2].x1);
        assert(ret[1].x1 == boxes[6].x1);
        assert(ret[2].x1 == boxes[3].x1);
        assert(ret[3].x1 == boxes[7].x1);
        assert(ret[4].x1 == boxes[0].x1);
        assert(ret[5].x1 == boxes[4].x1);
    }
    {
        // xywh
        std::vector<int>        ret;
        std::vector<int>        classes{ 0, 0, 0, 1, 0, 0, 0, 1 };
        std::vector<float>      scores { 0.8F, 0.8F, 0.9F, 0.85F, 0.8F, 0.8F, 0.9F, 0.85F };
        std::vector<cv::Rect2d> boxes{
            cv::Rect2d{  2,  2,   6,   6 },
            cv::Rect2d{ 10, 10, 110, 110 },
            cv::Rect2d{ 20, 20, 120, 120 },
            cv::Rect2d{ 20, 20, 120, 120 },
            cv::Rect2d{  2 / 100.0,  2 / 100.0,   6 / 100.0,   6 / 100.0 },
            cv::Rect2d{ 10 / 100.0, 10 / 100.0, 110 / 100.0, 110 / 100.0 },
            cv::Rect2d{ 20 / 100.0, 20 / 100.0, 120 / 100.0, 120 / 100.0 },
            cv::Rect2d{ 20 / 100.0, 20 / 100.0, 120 / 100.0, 120 / 100.0 },
        };
        // cv::dnn::NMSBoxes(boxes, scores, 0, 0.5F, ret);
        cv::dnn::NMSBoxesBatched(boxes, scores, classes, 0, 0.5F, ret);
        assert(ret.size() == 6);
        assert(boxes[ret[0]].x == boxes[2].x);
        assert(boxes[ret[1]].x == boxes[6].x);
        assert(boxes[ret[2]].x == boxes[3].x);
        assert(boxes[ret[3]].x == boxes[7].x);
        assert(boxes[ret[4]].x == boxes[0].x);
        assert(boxes[ret[5]].x == boxes[4].x);
    }
}

[[maybe_unused]]
void test_loc_nms_boxes() {
    CAIWEI_FOR_EACH(10'000)
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
    caiwei::transform::nms_boxes(std::move(boxes), 0.5F);
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_ocv_nms_boxes() {
    CAIWEI_FOR_EACH(10'000)
    std::vector<int>        ret;
    std::vector<int>        classes{ 0, 0, 0, 1, 0, 0, 0, 1 };
    std::vector<float>      scores { 0.8F, 0.8F, 0.9F, 0.85F, 0.8F, 0.8F, 0.9F, 0.85F };
    std::vector<cv::Rect2d> boxes{
        cv::Rect2d{  2,  2,   6,   6 },
        cv::Rect2d{ 10, 10, 110, 110 },
        cv::Rect2d{ 20, 20, 120, 120 },
        cv::Rect2d{ 20, 20, 120, 120 },
        cv::Rect2d{  2 / 100.0,  2 / 100.0,   6 / 100.0,   6 / 100.0 },
        cv::Rect2d{ 10 / 100.0, 10 / 100.0, 110 / 100.0, 110 / 100.0 },
        cv::Rect2d{ 20 / 100.0, 20 / 100.0, 120 / 100.0, 120 / 100.0 },
        cv::Rect2d{ 20 / 100.0, 20 / 100.0, 120 / 100.0, 120 / 100.0 },
    };
    // cv::dnn::NMSBoxes(boxes, scores, 0, 0.5F, ret);
    cv::dnn::NMSBoxesBatched(boxes, scores, classes, 0, 0.5F, ret);
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_min_max_loc() {
    float scores[128];
    std::iota(scores, scores + 128, 1.0F);
    scores[100] = 10000.0F;
    {
        int   max_index;
        float max_score;
        caiwei::transform::max_loc(scores, 128, max_score, max_index);
        assert(max_score == 10000.0F);
        assert(max_index == 100);
    }
    {
        double    max_score;
        cv::Point max_index;
        cv::Mat mat(1, 128, CV_32FC1, scores);
        cv::minMaxLoc(mat, NULL, &max_score, NULL, &max_index);
        assert(max_score == 10000.0F);
        assert(max_index.x == 100);
    }
}

[[maybe_unused]]
void test_loc_min_max_loc() {
    float scores[128];
    std::iota(scores, scores + 128, 1.0F);
    int   max_index;
    float max_score;
    CAIWEI_FOR_EACH(10'000)
    caiwei::transform::max_loc(scores, 128, max_score, max_index);
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_ocv_min_max_loc() {
    double    max_score;
    cv::Point max_class;
    float scores[128];
    std::iota(scores, scores + 128, 1.0F);
    cv::Mat mat(1, 128, CV_32FC1, scores);
    CAIWEI_FOR_EACH(10'000)
    cv::minMaxLoc(mat, NULL, &max_score, NULL, &max_class);
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_transpose() {
    uint8_t src[128];
    uint8_t dts[128];
    std::iota(src, src + 128, 1);
    caiwei::transform::transpose(src, dts, 2, 64);
    caiwei::transform::transpose(dts, src, 64, 2);
    cv::Mat mat = cv::Mat(64, 2, CV_8UC1, dts).clone();
    bool ret = std::equal(dts, dts + 128, mat.data);
    assert(ret);
    mat = mat.t();
    ret = std::equal(src, src + 128, mat.data);
    assert(ret);
    mat = mat.t();
    ret = std::equal(dts, dts + 128, mat.data);
    assert(ret);
}

[[maybe_unused]]
void test_loc_transpose() {
    uint8_t src[84 * 8400];
    uint8_t dts[84 * 8400];
    std::iota(src, src + 84 * 8400, 1);
    CAIWEI_FOR_EACH(10'000)
    caiwei::transform::transpose(src, dts, 84, 8400);
    caiwei::transform::transpose(src, dts, 8400, 84);
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_ocv_transpose() {
    uint8_t src[84 * 8400];
    uint8_t dts[84 * 8400];
    std::iota(src, src + 84 * 8400, 1);
    cv::Mat mat = cv::Mat(8400, 84, CV_8UC1, dts);
    CAIWEI_FOR_EACH(10'000)
    mat.t().t();
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_convert_to() {
    {
        int width, height, channels;
        auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
        std::vector<float> dst(width * height * channels);
        caiwei::transform::i8_to_f32(data, width * height * channels, dst.data(), 255.0F);
        cv::Mat mat = cv::Mat(height, width, CV_32FC3, dst.data()).clone();
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        cv::imshow("loc", mat);
        cv::waitKey(0);
        stbi_image_free(data);
    }
    {
        auto mat = cv::imread(IMAGE_PATH);
        cv::Mat dst;
        mat.convertTo(dst, CV_32FC3, 1 / 255.0F);
        cv::imshow("ocv", dst);
        cv::waitKey(0);
    }
}

[[maybe_unused]]
void test_loc_convert_to() {
    // 1280 = 3 ms
    // 1920 = 8 ms
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    CAIWEI_FOR_EACH(1000)
    std::vector<float> dst(width * height * channels);
    caiwei::transform::i8_to_f32(data, width * height * channels, dst.data());
    caiwei::transform::i8_to_f32(data, width * height * channels, dst.data(), 255.0F);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

[[maybe_unused]]
void test_ocv_convert_to() {
    // 1280 = 2 ms
    // 1920 = 7 ms
    auto mat = cv::imread(IMAGE_PATH);
    CAIWEI_FOR_EACH(1000)
    cv::Mat dst;
    mat.convertTo(dst, CV_32FC3);
    mat.convertTo(dst, CV_32FC3, 1 / 255.0F);
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_blob_from_image() {
    {
        int width, height, channels;
        auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
        std::vector<float> hwc(width * height * channels);
        std::vector<float> chw(width * height * channels);
        caiwei::transform::i8_to_f32(data, width * height * channels, hwc.data(), 255.0F);
        caiwei::transform::hwc_to_chw(hwc.data(), chw.data(), height, width, channels);
        caiwei::transform::chw_to_hwc(chw.data(), hwc.data(), height, width, channels);
        cv::Mat hwc_mat(height, width, CV_32FC3, hwc.data());
        cv::cvtColor(hwc_mat, hwc_mat, cv::COLOR_RGB2BGR);
        cv::imshow("loc", hwc_mat);
        cv::waitKey(0);
        stbi_image_free(data);
    }
    {
        auto mat = cv::imread(IMAGE_PATH);
        cv::Mat nchw_mat;
        cv::dnn::blobFromImage(mat, nchw_mat, 1.0F / 255.0F);
        std::vector<float> hwc(mat.cols * mat.rows * mat.channels());
        caiwei::transform::chw_to_hwc((float*) nchw_mat.data, hwc.data(), mat.cols, mat.rows, mat.channels());
        cv::Mat hwc_mat(mat.size(), CV_32FC3, hwc.data());
        cv::imshow("ocv", hwc_mat);
        cv::waitKey(0);
    }
}

[[maybe_unused]]
void test_loc_blob_from_image() {
    // 1280 = 4 ms
    // 1920 = 9 ms
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    CAIWEI_FOR_EACH(1000)
    std::vector<float> dst(width * height * channels);
    caiwei::transform::i8_to_f32(data, width * height * channels, dst.data(), 255.0F);
    caiwei::transform::hwc_to_chw(dst.data(), dst.data(), height, width, channels);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

[[maybe_unused]]
void test_ocv_blob_from_image() {
    // 1280 =  7 ms
    // 1920 = 17 ms
    auto mat = cv::imread(IMAGE_PATH);
    CAIWEI_FOR_EACH(1000)
    cv::Mat dst;
    cv::dnn::blobFromImage(mat, dst, 1.0F / 255.0F);
    CAIWEI_FOR_EACH_END
}

int main() {
    init_test();
    cv::utils::logging::setLogLevel(::cv::utils::logging::LOG_LEVEL_ERROR);
    // test_load();
    test_loc_load();
    test_ocv_load();
    // test_crop();
    test_loc_crop();
    test_ocv_crop();
    // test_draw();
    test_loc_draw();
    test_ocv_draw();
    // test_resize();
    test_loc_resize();
    test_ocv_resize();
    test_ffmpeg_resize();
    // test_resize_pad();
    test_loc_resize_pad();
    test_ocv_resize_pad();
    // test_nms_boxes();
    test_loc_nms_boxes();
    test_ocv_nms_boxes();
    // test_min_max_loc();
    test_loc_min_max_loc();
    test_ocv_min_max_loc();
    // test_transpose();
    test_loc_transpose();
    test_ocv_transpose();
    // test_convert_to();
    test_loc_convert_to();
    test_ocv_convert_to();
    // test_blob_from_image();
    test_loc_blob_from_image();
    test_ocv_blob_from_image();
    stop_test();
    return 0;
}
