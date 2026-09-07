#include "test.hpp"

extern "C" {

#include "libavcodec/avcodec.h"

}

[[maybe_unused]]
void det_image() {
    int width, height, channels;
    auto data = stbi_load("./acgist.jpg", &width, &height, &channels, STBI_default);
    caiwei::media::ImageFrame frame(width * height * channels);
    std::copy_n(data, width * height * channels, frame.data.data());
    frame.width = width;
    frame.height = height;
    frame.channels = channels;
    auto ptr = caiwei::context::get_context<caiwei::context::PoseContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Pose>>("yolo26n-pose");
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
    auto ptr = caiwei::context::get_context<caiwei::context::PoseContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Pose>>("yolo26n-pose");
    int frame_count = 0;
    std::vector<caiwei::image::Pose> ret;
    caiwei::media::MediaDemuxer media_demuxer(type, url, [](const caiwei::media::AudioFrame& frame) {
        return caiwei::player::play_audio(frame.data.data(), frame.data_length);
    }, [&ptr, &ret, &frame_count](const caiwei::media::VideoFrame& frame) {
        auto data = frame.data;
        if (frame_count++ % 8 == 0) {
            ret = std::move(ptr->run(frame));
        }
        for (const auto& pose : ret) {
            const auto& box   = pose.box;
            const auto& point = pose.point;
            caiwei::image::draw_rect(
                data.data(),
                frame.width,
                frame.height,
                box.x1 * frame.width,
                box.y1 * frame.height,
                (box.x2 - box.x1) * frame.width,
                (box.y2 - box.y1) * frame.height
            );
            //  0-鼻子
            //  1-左眼  2-右眼  3-左耳  4-右耳
            //  5-左肩  6-右肩  7-左肘  8-右肘  9-左腕   10-右腕
            // 11-左髋 12-右髋 13-左膝 14-右膝 15-左脚踝 16-右脚踝
            std::vector<std::pair<int, int>> skeleton = {
                { 0,  1}, { 0,  2}, {0, 3}, {0,  4},
                { 5,  7}, { 7,  9},
                { 6,  8}, { 8, 10},
                {11, 13}, {13, 15},
                {12, 14}, {14, 16},
            };
            for (auto[ a, z ] : skeleton) {
                const auto& a_point = point[a];
                const auto& z_point = point[z];
                if (a_point.score < 0.5F || z_point.score < 0.5F) {
                    continue;
                }
                caiwei::image::draw_line(
                    data.data(),
                    frame.width,
                    frame.height,
                    a_point.x * frame.width,
                    a_point.y * frame.height,
                    z_point.x * frame.width,
                    z_point.y * frame.height,
                    128
                );
                caiwei::image::draw_point(
                    data.data(),
                    frame.width,
                    frame.height,
                    a_point.x * frame.width,
                    a_point.y * frame.height,
                    255
                );
            }
        }
        return caiwei::player::play_video(data.data(), frame.width * 3);
    });
    std::thread player([]() {
        caiwei::player::open_player(1, 16000, 640, 360);
    });
    media_demuxer.open(caiwei::media::AudioInfo(1, 16000, AV_SAMPLE_FMT_S16), caiwei::media::VideoInfo(640, 0, AV_PIX_FMT_RGB24));
    caiwei::player::stop_player();
    player.join();
    media_demuxer.stop();
}

int main() {
    init_test();
    // det_image();
    det_video();
    stop_test();
    return 0;
}
