#include "test.hpp"

extern "C" {
#include "libavcodec/avcodec.h"
}

#include <thread>
#include <vector>

[[maybe_unused]]
void cls_image() {
    int width, height, channels;
    auto data = stbi_load("./samoyed.jpg", &width, &height, &channels, STBI_default);
    caiwei::media::ImageFrame frame(width * height * channels);
    std::copy_n(data, width * height * channels, frame.data.data());
    stbi_image_free(data);
    frame.width = width;
    frame.height = height;
    frame.channels = channels;
    auto ptr = caiwei::manager::get_context<caiwei::context::ClsContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Cls>>("yolo26n-cls");
    if (ptr == nullptr) {
        return;
    }
    auto result = ptr->run(frame);
    for (const auto& cls : result) {
        std::printf("cls = %d, confidence = %f\n", cls.first, cls.second);
    }
    // 258 == samoyed
    assert(!result.empty());
    assert(result[0].first == 258);
    caiwei_wait_enter();
    {
        CAIWEI_FOR_EACH(100)
        auto result = ptr->run(frame);
        CAIWEI_FOR_EACH_END
    }
    caiwei_wait_enter();
    {
        CAIWEI_FOR_EACH(100)
        auto result = ptr->run(frame);
        CAIWEI_FOR_EACH_END
    }
    caiwei_wait_enter();
}

[[maybe_unused]]
void cls_image_thread() {
    int width, height, channels;
    auto data = stbi_load("./samoyed.jpg", &width, &height, &channels, STBI_default);
    caiwei::media::ImageFrame frame(width * height * channels);
    std::copy_n(data, width * height * channels, frame.data.data());
    stbi_image_free(data);
    frame.width = width;
    frame.height = height;
    frame.channels = channels;
    auto ptr = caiwei::manager::get_context<caiwei::context::ClsContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Cls>>("yolo26n-cls");
    if (ptr == nullptr) {
        return;
    }
    std::vector<std::thread> threads;
    for (int i = 0; i < 2; ++i) {
        threads.push_back(std::thread([&]() {
            CAIWEI_FOR_EACH(100)
            ptr->run(frame);
            CAIWEI_FOR_EACH_END
        }));
    }
    for (auto& thread : threads) {
        thread.join();
    }
    caiwei_wait_enter();
}

[[maybe_unused]]
void cls_video() {
    // rtp|sdp|file|http|rtmp|rtsp|device
    auto type = "file";
    auto url  = "./caiwei.mp4";
    // auto url = R"(audio=麦克风阵列 (适用于数字麦克风的英特尔® 智音技术):video=Integrated Camera)";
    auto ptr = caiwei::manager::get_context<caiwei::context::ClsContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Cls>>("yolo26n-cls");
    if (ptr == nullptr) {
        return;
    }
    caiwei::player::open_player(1, 16000, 640, 360);
    caiwei::media::MediaDemuxer media_demuxer(type, url, [](caiwei::media::AudioFrame& frame) {
        return caiwei::player::play_audio(frame.data.data(), frame.data.size());
    }, [&ptr](caiwei::media::VideoFrame& frame) {
        auto ret = ptr->run(frame);
        for (const auto& index : ret) {
            CW_LOG_D("%d = %.2f", index.first, index.second);
        }
        return caiwei::player::play_video(frame.data.data(), frame.width * 3);
    });
    media_demuxer.open(caiwei::media::AudioInfo(1, 16000, AV_SAMPLE_FMT_S16), caiwei::media::VideoInfo(640, 0, AV_PIX_FMT_RGB24));
    media_demuxer.stop();
    caiwei::player::stop_player();
}

int main(int argc, char* argv[]) {
    #if ENABLE_CAIWEI_RUNTIME_RKNN2
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "CLS,YOLO,yolo26n-cls,yolo26n-cls-rk3588-f16.rknn");
    // caiwei::env::set("CAIWEI_CONTEXT_INFO", "CLS,YOLO,yolo26n-cls,yolo26n-cls-rk3588-i8-toolkit.rknn");
    #endif
    if (argc > 1) {
        caiwei::env::set("CAIWEI_CONTEXT_INFO", argv[1]);
    }
    caiwei::test::init_test();
    cls_image();
    // cls_image_thread();
    // cls_image_thread();
    // cls_video();
    caiwei::test::stop_test();
    return 0;
}
