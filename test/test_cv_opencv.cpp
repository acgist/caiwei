#include "test_media.hpp"

#include "caiwei/transform.hpp"

#include <numeric>

#include "opencv2/opencv.hpp"
#include "opencv2/core/utils/logger.hpp"

#define IMAGE_PATH "D:/download/h.jpg"
// #define IMAGE_PATH "D:/download/w.jpg"
// #define IMAGE_PATH "D:/download/caiwei.jpg"

[[maybe_unused]]
void test_load() {
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    cv::Mat mat(height, width, CV_8UC3, data);
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    cv::imshow("image", mat);
    cv::waitKey(0);
    stbi_image_free(data);
}

[[maybe_unused]]
void test_loc_load() {
    // 100 = 1318 ms
    CAIWEI_FOR_EACH(100)
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    stbi_image_free(data);
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_ocv_load() {
    // 100 = 833 ms
    CAIWEI_FOR_EACH(100)
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
        caiwei::transform::crop(data, width, height, dst.data(), 512, 512, 256, 256, channels);
        mat = cv::Mat(256, 256, CV_8UC3, dst.data()).clone();
        cv::imshow("loc", mat);
        cv::waitKey(0);
        stbi_image_free(data);
    }
    {
        auto mat = cv::imread(IMAGE_PATH);
        mat = mat(cv::Rect(512, 512, 256, 256));
        cv::imshow("ocv", mat);
        cv::waitKey(0);
    }
}

[[maybe_unused]]
void test_loc_crop() {
    // 10'000 = 37 ms
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    std::vector<uint8_t> dst(256 * 256 * channels);
    CAIWEI_FOR_EACH(10'000)
    caiwei::transform::crop(data, width, height, dst.data(), 512, 512, 256, 256, channels);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

[[maybe_unused]]
void test_ocv_crop() {
    // 10'000 = 42 ms
    auto mat = cv::imread(IMAGE_PATH);
    CAIWEI_FOR_EACH(10'000)
    auto crop = mat(cv::Rect(512, 512, 256, 256)).clone();
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_draw() {
    {
        int width, height, channels;
        auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
        caiwei::transform::draw(data, width, height, 512, 512, 256, 256);
        cv::Mat mat(height, width, CV_8UC3, data);
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        cv::imshow("loc", mat);
        cv::waitKey(0);
        stbi_image_free(data);
    }
    {
        auto mat = cv::imread(IMAGE_PATH);
        cv::rectangle(mat, cv::Point(512, 512), cv::Point(768, 768), cv::Scalar{ 0, 0, 0 });
        cv::imshow("ocv", mat);
        cv::waitKey(0);
    }
}

[[maybe_unused]]
void test_loc_draw() {
    // 10'000 = 2 ms
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    CAIWEI_FOR_EACH(10'000)
    caiwei::transform::draw(data, width, height, 512, 512, 256, 256);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

[[maybe_unused]]
void test_ocv_draw() {
    // 10'000 = 18 ms
    auto mat = cv::imread(IMAGE_PATH);
    CAIWEI_FOR_EACH(10'000)
    cv::rectangle(mat, cv::Point(512, 512), cv::Point(768, 768), cv::Scalar{ 0, 0, 0 });
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_resize() {
    int w = 640;
    int h = 640;
    float scale;
    {
        int width, height, channels;
        auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
        caiwei::transform::resize(width, height, w, h, scale);
        std::vector<uint8_t> dst(w * h * channels);
        caiwei::transform::resize(data, width, height, dst.data(), w, h);
        cv::Mat mat(h, w, CV_8UC3, dst.data());
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        cv::imshow("loc", mat);
        cv::waitKey(0);
        stbi_image_free(data);
    }
    {
        auto mat = cv::imread(IMAGE_PATH);
        cv::Mat dst;
        cv::resize(mat, dst, cv::Size(w, h));
        cv::imshow("ocv", dst);
        cv::waitKey(0);
    }
}

[[maybe_unused]]
void test_loc_resize() {
    // 100 = 375 ms
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    int w = 640;
    int h = 640;
    float scale;
    caiwei::transform::resize(width, height, w, h, scale);
    CAIWEI_FOR_EACH(100)
    std::vector<uint8_t> dst(w * h * channels);
    caiwei::transform::resize(data, width, height, dst.data(), w, h);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

[[maybe_unused]]
void test_ocv_resize() {
    // 100 = 41 ms
    auto mat = cv::imread(IMAGE_PATH);
    int w = 640;
    int h = 640;
    float scale;
    caiwei::transform::resize(mat.cols, mat.rows, w, h, scale);
    CAIWEI_FOR_EACH(100)
    cv::Mat dst;
    cv::resize(mat, dst, cv::Size(w, h));
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_resize_pad() {
    int width, height, channels;
    int w = 640;
    int h = 640;
    float scale;
    {
        auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
        caiwei::transform::resize(width, height, w, h, scale);
        int max = std::max(w, h);
        int pad_t, pad_l;
        std::vector<uint8_t> dst(w * h * channels);
        std::vector<uint8_t> pad(max * max * channels);
        caiwei::transform::resize(data, width, height, dst.data(), pad.data(), w, h, pad_t, pad_l);
        cv::Mat mat(max, max, CV_8UC3, pad.data());
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        cv::imshow("loc", mat);
        cv::waitKey(0);
        stbi_image_free(data);
    }
    {
        auto mat = cv::imread(IMAGE_PATH);
        cv::Mat dst;
        int max = std::max(w, h);
        cv::resize(mat, dst, cv::Size(w, h));
        cv::Mat result = cv::Mat::zeros(max, max, CV_8UC3);
        dst.copyTo(result(cv::Rect((max - w) / 2, (max - h) / 2, w, h)));
        cv::imshow("ocv", result);
        cv::waitKey(0);
    }
}

[[maybe_unused]]
void test_loc_resize_pad() {
    // 100 = 317 ms
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    int w = 640;
    int h = 640;
    float scale;
    caiwei::transform::resize(width, height, w, h, scale);
    int max = std::max(w, h);
    int pad_t, pad_l;
    CAIWEI_FOR_EACH(100)
    std::vector<uint8_t> dst(w * h * channels);
    std::vector<uint8_t> pad(max * max * channels);
    caiwei::transform::resize(data, width, height, dst.data(), pad.data(), w, h, pad_t, pad_l);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

[[maybe_unused]]
void test_ocv_resize_pad() {
    // 100 = 66 ms
    auto mat = cv::imread(IMAGE_PATH);
    int w = 640;
    int h = 640;
    float scale;
    caiwei::transform::resize(mat.cols, mat.rows, w, h, scale);
    int max = std::max(w, h);
    CAIWEI_FOR_EACH(100)
    cv::Mat dst;
    cv::resize(mat, dst, cv::Size(w, h));
    cv::Mat result = cv::Mat::zeros(max, max, CV_8UC3);
    dst.copyTo(result(cv::Rect((max - w) / 2, (max - h) / 2, w, h)));
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
    uint8_t src[128];
    uint8_t dts[128];
    std::iota(src, src + sizeof(src) / sizeof(uint8_t), 1);
    CAIWEI_FOR_EACH(10'000)
    caiwei::transform::transpose(src, dts, 2, 64);
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_ocv_transpose() {
    uint8_t src[128];
    uint8_t dts[128];
    std::iota(src, src + sizeof(src) / sizeof(uint8_t), 1);
    cv::Mat mat = cv::Mat(64, 2, CV_8UC1, dts).clone();
    CAIWEI_FOR_EACH(10'000)
    mat.t();
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_loc_convert_to() {
    // 100 = 800 ms
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    CAIWEI_FOR_EACH(100)
    std::vector<float> dst(width * height * channels);
    caiwei::transform::i8_to_f32(data, width * height * channels, dst.data());
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

[[maybe_unused]]
void test_ocv_convert_to() {
    // 100 = 793 ms
    auto mat = cv::imread(IMAGE_PATH);
    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
    CAIWEI_FOR_EACH(100)
    cv::Mat dst;
    mat.convertTo(dst, CV_32FC3);
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_min_max_loc() {
    float scores[128];
    std::iota(scores, scores + sizeof(scores) / sizeof(float), 1.0F);
    scores[100] = 10000.0F;
    {
        int   max_index;
        float max_score;
        caiwei::transform::max_loc(scores, sizeof(scores) / sizeof(float), max_score, max_index);
        LOG_INFO("max_score: %f, max_index: %d", max_score, max_index);
    }
    {
        double    max_score;
        cv::Point max_index;
        cv::Mat mat(1, sizeof(scores) / sizeof(float), CV_32FC1, scores);
        cv::minMaxLoc(mat, NULL, &max_score, NULL, &max_index);
        LOG_INFO("max_score: %f, max_index: %d", max_score, max_index.x);
    }
}

[[maybe_unused]]
void test_loc_min_max_loc() {
    float scores[128];
    std::iota(scores, scores + sizeof(scores) / sizeof(float), 1.0F);
    int   max_index;
    float max_score;
    CAIWEI_FOR_EACH(10'000)
    caiwei::transform::max_loc(scores, sizeof(scores) / sizeof(float), max_score, max_index);
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_ocv_min_max_loc() {
    double    max_score;
    cv::Point max_class;
    float scores[128];
    std::iota(scores, scores + sizeof(scores) / sizeof(float), 1.0F);
    cv::Mat mat(1, sizeof(scores) / sizeof(float), CV_32FC1, scores);
    CAIWEI_FOR_EACH(10'000)
    cv::minMaxLoc(mat, NULL, &max_score, NULL, &max_class);
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
        cv::Mat hwc_mat = cv::Mat(height, width, CV_32FC3, hwc.data()).clone();
        cv::cvtColor(hwc_mat, hwc_mat, cv::COLOR_RGB2BGR);
        cv::imshow("image", hwc_mat);
        cv::waitKey(0);
        caiwei::transform::hwc_to_chw(hwc.data(), chw.data(), height, width, channels);
        caiwei::transform::chw_to_hwc(chw.data(), hwc.data(), height, width, channels);
        hwc_mat = cv::Mat(height, width, CV_32FC3, hwc.data()).clone();
        cv::cvtColor(hwc_mat, hwc_mat, cv::COLOR_RGB2BGR);
        cv::imshow("image", hwc_mat);
        cv::waitKey(0);
        stbi_image_free(data);
    }
    {
        auto hwc_mat = cv::imread(IMAGE_PATH);
        cv::imshow("image", hwc_mat);
        cv::waitKey(0);
        cv::cvtColor(hwc_mat, hwc_mat, cv::COLOR_BGR2RGB);
        cv::Mat nchw_mat;
        cv::dnn::blobFromImage(hwc_mat, nchw_mat, 1.0F / 255.0F);
        std::vector<float> hwc(hwc_mat.cols * hwc_mat.rows * hwc_mat.channels());
        float* chw = (float*) nchw_mat.data;
        caiwei::transform::chw_to_hwc(chw, hwc.data(), hwc_mat.cols, hwc_mat.rows, hwc_mat.channels());
        hwc_mat = cv::Mat(hwc_mat.size(), CV_32FC3, hwc.data()).clone();
        cv::cvtColor(hwc_mat, hwc_mat, cv::COLOR_RGB2BGR);
        cv::imshow("image", hwc_mat);
        cv::waitKey(0);
    }
}

[[maybe_unused]]
void test_loc_blob_from_image() {
    // 100 = 1179 ms
    int width, height, channels;
    auto data = stbi_load(IMAGE_PATH, &width, &height, &channels, STBI_default);
    CAIWEI_FOR_EACH(100)
    std::vector<float> dst(width * height * channels);
    std::transform(data, data + width * height * channels, dst.data(), [](uint8_t v) {
        return static_cast<float>(v) / 255.0F;
    });
    caiwei::transform::hwc_to_chw(dst.data(), dst.data(), height, width, channels);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

[[maybe_unused]]
void test_ocv_blob_from_image() {
    // 100 = 2073 ms
    auto mat = cv::imread(IMAGE_PATH);
    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
    CAIWEI_FOR_EACH(100)
    cv::Mat dst;
    cv::dnn::blobFromImage(mat, dst, 1.0F / 255.0F);
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]]
void test_nms_boxes() {
    {
        std::vector<caiwei::media::Box> boxes = {
            {  2,  2,   8,   8, 0, 0.80F },
            { 10, 10, 120, 120, 0, 0.80F },
            { 20, 20, 140, 140, 0, 0.90F },
            { 20, 20, 140, 140, 1, 0.85F },
            {  2 / 100.0F,  2 / 100.0F,   8 / 100.0F,   8 / 100.0F, 0, 0.80F },
            { 10 / 100.0F, 10 / 100.0F, 120 / 100.0F, 120 / 100.0F, 0, 0.80F },
            { 20 / 100.0F, 20 / 100.0F, 140 / 100.0F, 140 / 100.0F, 0, 0.90F },
            { 20 / 100.0F, 20 / 100.0F, 140 / 100.0F, 140 / 100.0F, 1, 0.85F },
        };
        for(auto &v : boxes) {
            printf("====>x1=%.2f y1=%.2f x2=%.2f y2=%.2f score=%.2f cls=%d\n", v.x1, v.y1, v.x2, v.y2, v.score, v.class_id);
        }
        auto ret = caiwei::transform::nms_boxes(boxes, 0.5F);
        for(auto &v : ret) {
            printf("<====x1=%.2f y1=%.2f x2=%.2f y2=%.2f score=%.2f cls=%d\n", v.x1, v.y1, v.x2, v.y2, v.score, v.class_id);
        }
    }
    {
        std::vector<int>        ret;
        std::vector<int>        classes{ 0, 0, 0, 1, 0, 0, 0, 1 };
        std::vector<float>      scores { 0.8F, 0.8F, 0.9F, 0.85F, 0.8F, 0.8F, 0.9F, 0.85F };
        std::vector<cv::Rect2d> boxes{
            cv::Rect2d{  2,  2,   8,   8 },
            cv::Rect2d{ 10, 10, 120, 120 },
            cv::Rect2d{ 20, 20, 140, 140 },
            cv::Rect2d{ 20, 20, 140, 140 },
            cv::Rect2d{  2 / 100.0,  2 / 100.0,   8 / 100.0,   8 / 100.0 },
            cv::Rect2d{ 10 / 100.0, 10 / 100.0, 120 / 100.0, 120 / 100.0 },
            cv::Rect2d{ 20 / 100.0, 20 / 100.0, 140 / 100.0, 140 / 100.0 },
            cv::Rect2d{ 20 / 100.0, 20 / 100.0, 140 / 100.0, 140 / 100.0 },
        };
        // cv::dnn::NMSBoxes(boxes, scores, 0, 0.5F, ret);
        cv::dnn::NMSBoxesBatched(boxes, scores, classes, 0, 0.5F, ret);
        for(auto &v : ret) {
            printf("====>v=%d\n", v);
        }
    }
}

[[maybe_unused]]
void test_loc_nms_boxes() {
    CAIWEI_FOR_EACH(10000)
    std::vector<caiwei::media::Box> boxes = {
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
    CAIWEI_FOR_EACH(10000)
    std::vector<int>        ret;
    std::vector<int>        classes{ 0, 0, 0, 1, 0, 0, 0, 1 };
    std::vector<float>      scores { 0.8F, 0.8F, 0.9F, 0.85F, 0.8F, 0.8F, 0.9F, 0.85F };
    std::vector<cv::Rect2d> boxes{
        cv::Rect2d{  2,  2,   8,   8 },
        cv::Rect2d{ 10, 10, 120, 120 },
        cv::Rect2d{ 20, 20, 140, 140 },
        cv::Rect2d{ 20, 20, 140, 140 },
        cv::Rect2d{  2 / 100.0,  2 / 100.0,   8 / 100.0,   8 / 100.0 },
        cv::Rect2d{ 10 / 100.0, 10 / 100.0, 120 / 100.0, 120 / 100.0 },
        cv::Rect2d{ 20 / 100.0, 20 / 100.0, 140 / 100.0, 140 / 100.0 },
        cv::Rect2d{ 20 / 100.0, 20 / 100.0, 140 / 100.0, 140 / 100.0 },
    };
    // cv::dnn::NMSBoxes(boxes, scores, 0, 0.5F, ret);
    cv::dnn::NMSBoxesBatched(boxes, scores, classes, 0, 0.5F, ret);
    CAIWEI_FOR_EACH_END
}

int main() {
    init_test();
    cv::utils::logging::setLogLevel(::cv::utils::logging::LOG_LEVEL_ERROR);
    // test_load();
    // test_loc_load();
    // test_ocv_load();
    // test_crop();
    // test_loc_crop();
    // test_ocv_crop();
    // test_draw();
    // test_loc_draw();
    // test_ocv_draw();
    // test_resize();
    // test_loc_resize();
    // test_ocv_resize();
    // test_resize_pad();
    test_loc_resize_pad();
    test_ocv_resize_pad();
    // test_nms_boxes();
    // test_loc_nms_boxes();
    // test_ocv_nms_boxes();
    // test_min_max_loc();
    // test_loc_min_max_loc();
    // test_ocv_min_max_loc();
    // test_transpose();
    // test_loc_transpose();
    // test_ocv_transpose();
    // test_loc_convert_to();
    // test_ocv_convert_to();
    // test_blob_from_image();
    // test_loc_blob_from_image();
    // test_ocv_blob_from_image();
    stop_test();
    return 0;
}
