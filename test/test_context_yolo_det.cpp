#include "test_runtime.hpp"

extern "C" {

#include "libavcodec/avcodec.h"

}

[[maybe_unused]]
void det_image() {
    // CUDA    100 =  1777 ms
    // Debug   100 = 19169 ms
    // Release 100 = 10160 ms
    int width, height, channels;
    auto data = stbi_load("./caiwei.jpg", &width, &height, &channels, STBI_default);
    caiwei::media::ImageFrame frame(width * height * channels);
    std::copy_n(data, width * height * channels, frame.data.data());
    frame.width = width;
    frame.height = height;
    frame.channels = channels;
    auto ptr = caiwei::context::get_context<caiwei::context::DetContext, caiwei::media::ImageFrame, std::vector<caiwei::yolo::Box>>("yolo26n-det");
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
    auto ptr = caiwei::context::get_context<caiwei::context::DetContext, caiwei::media::ImageFrame, std::vector<caiwei::yolo::Box>>("yolo26n-det");
    int frame_count = 0;
    std::vector<caiwei::yolo::Box> ret;
    caiwei::media::MediaDemuxer media_demuxer(type, url, [](const caiwei::media::AudioFrame& frame) {
        return caiwei::player::play_audio(frame.data.data(), frame.data_length);
    }, [&ptr, &ret, &frame_count](const caiwei::media::VideoFrame& frame) {
        auto data = frame.data;
        if (frame_count++ % 4 == 0) {
            ret = std::move(ptr->run(frame));
        }
        for (const auto& box : ret) {
            caiwei::transform::draw(
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
