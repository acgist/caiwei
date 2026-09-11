#include "test.hpp"

extern "C" {
#include "libavcodec/avcodec.h"
}

[[maybe_unused]]
void seg_draw() {
    int width, height, channels;
    auto data = stbi_load("./acgist.jpg", &width, &height, &channels, STBI_default);
    caiwei::media::ImageFrame frame(width * height * channels);
    std::copy_n(data, width * height * channels, frame.data.data());
    frame.width = width;
    frame.height = height;
    frame.channels = channels;
    auto ptr = caiwei::manager::get_context<caiwei::context::SegContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Seg>>("yolo26n-seg");
    if (ptr == nullptr) {
        return;
    }
    auto result = ptr->run(frame);
    for (const auto& seg : result) {
        const auto& box  = seg.box;
        const auto& mask = seg.mask;
        const int box_x = box.x1 * frame.width;
        const int box_y = box.y1 * frame.height;
        const int box_w = (box.x2 - box.x1) * frame.width;
        const int box_h = (box.y2 - box.y1) * frame.height;
        caiwei::test::draw_rect(
            data,
            frame.width,
            frame.height,
            box_x,
            box_y,
            box_w,
            box_h
        );
        std::vector<float> mask_data(box_w * box_h);
        caiwei::image::resize(mask.mask.data(), mask_data.data(), mask.w, mask.h, box_w, box_h);
        const uint8_t color[3] = { 0, 255, 0 };
        caiwei::test::draw_mask(data, frame.width, frame.height, mask_data.data(), box_x, box_y, box_w, box_h, 0.5, color);
    }
    stbi_write_jpg("./acgist_seg.jpg", width, height, channels, data, 80);
    stbi_image_free(data);
}

[[maybe_unused]]
void seg_image() {
    // CUDA    100 =  1777 ms
    // Debug   100 = 19169 ms
    // Release 100 = 10160 ms
    int width, height, channels;
    auto data = stbi_load("./acgist.jpg", &width, &height, &channels, STBI_default);
    caiwei::media::ImageFrame frame(width * height * channels);
    std::copy_n(data, width * height * channels, frame.data.data());
    frame.width = width;
    frame.height = height;
    frame.channels = channels;
    auto ptr = caiwei::manager::get_context<caiwei::context::SegContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Seg>>("yolo26n-seg");
    if (ptr == nullptr) {
        return;
    }
    CAIWEI_FOR_EACH(100)
    auto result = ptr->run(frame);
    CAIWEI_FOR_EACH_END
    stbi_image_free(data);
}

[[maybe_unused]]
void seg_video() {
    // rtp|sdp|file|http|rtmp|rtsp|device
    auto type = "file";
    auto url  = "./caiwei.mp4";
    // auto url = R"(audio=麦克风阵列 (适用于数字麦克风的英特尔® 智音技术):video=Integrated Camera)";
    auto ptr = caiwei::manager::get_context<caiwei::context::SegContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Seg>>("yolo26n-seg");
    if (ptr == nullptr) {
        return;
    }
    int frame_count = 0;
    std::vector<caiwei::image::Seg> ret;
    caiwei::player::open_player(1, 16000, 640, 360);
    caiwei::media::MediaDemuxer media_demuxer(type, url, [](const caiwei::media::AudioFrame& frame) {
        return caiwei::player::play_audio(frame.data.data(), frame.data_length);
    }, [&ptr, &ret, &frame_count](const caiwei::media::VideoFrame& frame) {
        auto data = frame.data;
        if (frame_count++ % 8 == 0) {
            ret = ptr->run(frame);
        }
        for (const auto& seg : ret) {
            const auto& box  = seg.box;
            const auto& mask = seg.mask;
            const int box_x = box.x1 * frame.width;
            const int box_y = box.y1 * frame.height;
            const int box_w = (box.x2 - box.x1) * frame.width;
            const int box_h = (box.y2 - box.y1) * frame.height;
            caiwei::test::draw_rect(
                data.data(),
                frame.width,
                frame.height,
                box_x,
                box_y,
                box_w,
                box_h
            );
            std::vector<float> mask_data(box_w * box_h);
            caiwei::image::resize(mask.mask.data(), mask_data.data(), mask.w, mask.h, box_w, box_h);
            const uint8_t color[3] = { 0, 255, 0 };
            caiwei::test::draw_mask(data.data(), frame.width, frame.height, mask_data.data(), box_x, box_y, box_w, box_h, 0.5, color);
        }
        return caiwei::player::play_video(data.data(), frame.width * 3);
    });
    media_demuxer.open(caiwei::media::AudioInfo(1, 16000, AV_SAMPLE_FMT_S16), caiwei::media::VideoInfo(640, 0, AV_PIX_FMT_RGB24));
    media_demuxer.stop();
    caiwei::player::stop_player();
}

int main() {
    #if ENABLE_CAIWEI_RUNTIME_RKNN2
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "SEG,YOLO,yolo26n-seg,yolo26n-seg-rk3588-f16.rknn");
    #endif
    caiwei::test::init_test();
    // seg_draw();
    // seg_image();
    seg_video();
    caiwei::test::stop_test();
    return 0;
}
