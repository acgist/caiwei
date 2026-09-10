#include "test.hpp"

extern "C" {
#include "libavcodec/avcodec.h"
}

[[maybe_unused]]
void det_draw() {
    int width, height, channels;
    auto data = stbi_load("./acgist.jpg", &width, &height, &channels, STBI_default);
    caiwei::media::ImageFrame frame(width * height * channels);
    std::copy_n(data, width * height * channels, frame.data.data());
    frame.width = width;
    frame.height = height;
    frame.channels = channels;
    auto ptr = caiwei::context::get_context<caiwei::context::DetContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Box>>("yolo26n-det");
    if (ptr == nullptr) {
        return;
    }
    auto result = ptr->run(frame);
    for (const auto& box : result) {
        CW_LOG_I("box: %d = %f", box.class_id, box.score);
        caiwei::test::draw_rect(
            data,
            frame.width,
            frame.height,
            box.x1 * frame.width,
            box.y1 * frame.height,
            (box.x2 - box.x1) * frame.width,
            (box.y2 - box.y1) * frame.height
        );
    }
    stbi_write_jpg("./acgist_det.jpg", width, height, channels, data, 80);
    stbi_image_free(data);
}

[[maybe_unused]]
void det_image() {
    int width, height, channels;
    auto data = stbi_load("./caiwei.jpg", &width, &height, &channels, STBI_default);
    caiwei::media::ImageFrame frame(width * height * channels);
    std::copy_n(data, width * height * channels, frame.data.data());
    frame.width = width;
    frame.height = height;
    frame.channels = channels;
    auto ptr = caiwei::context::get_context<caiwei::context::DetContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Box>>("yolo26n-det");
    if (ptr == nullptr) {
        return;
    }
    CAIWEI_FOR_EACH(100)
    auto result = ptr->run(frame);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

[[maybe_unused]]
void det_video() {
    // rtp|sdp|file|http|rtmp|rtsp|device
    auto type = "file";
    auto url  = "./caiwei.mp4";
    // auto url = R"(audio=麦克风阵列 (适用于数字麦克风的英特尔® 智音技术):video=Integrated Camera)";
    auto ptr = caiwei::context::get_context<caiwei::context::DetContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Box>>("yolo26n-det");
    if (ptr == nullptr) {
        return;
    }
    int frame_count = 0;
    std::vector<caiwei::image::Box> ret;
    caiwei::player::open_player(1, 16000, 640, 360);
    caiwei::media::MediaDemuxer media_demuxer(type, url, [](const caiwei::media::AudioFrame& frame) {
        return caiwei::player::play_audio(frame.data.data(), frame.data_length);
    }, [&ptr, &ret, &frame_count](const caiwei::media::VideoFrame& frame) {
        auto data = frame.data;
        if (frame_count++ % 4 == 0) {
            ret = ptr->run(frame);
        }
        for (const auto& box : ret) {
            caiwei::test::draw_rect(
                data.data(),
                frame.width,
                frame.height,
                box.x1 * frame.width,
                box.y1 * frame.height,
                (box.x2 - box.x1) * frame.width,
                (box.y2 - box.y1) * frame.height
            );
        }
        return caiwei::player::play_video(data.data(), frame.width * 3);
    });
    media_demuxer.open(caiwei::media::AudioInfo(1, 16000, AV_SAMPLE_FMT_S16), caiwei::media::VideoInfo(640, 0, AV_PIX_FMT_RGB24));
    media_demuxer.stop();
    caiwei::player::stop_player();
}

int main() {
    #if ENABLE_CAIWEI_RUNTIME_RKNN2
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "DET,YOLO,yolo26n-det,yolo26n-det-rk3588-f16.rknn");
    #endif
    caiwei::test::init_test();
    // det_draw();
    // det_image();
    det_video();
    caiwei::test::stop_test();
    return 0;
}
