#include "test.hpp"

#include <thread>
#include <vector>

[[maybe_unused]]
void det_draw() {
    caiwei::media::ImageFrame frame;
    if (!frame.load_from_file("./acgist.jpg")) {
        return;
    }
    auto ptr = caiwei::manager::get_context<caiwei::context::DetContext>("yolo26n-det");
    if (ptr == nullptr) {
        return;
    }
    auto result = ptr->run(frame);
    for (const auto& box : result) {
        CW_LOG_I("box: %d = %.2f", box.class_id, box.score);
        caiwei::test::draw_rect(
            frame.data.data(),
            frame.width,
            frame.height,
            static_cast<int>(box.x1 * frame.width),
            static_cast<int>(box.y1 * frame.height),
            static_cast<int>((box.x2 - box.x1) * frame.width),
            static_cast<int>((box.y2 - box.y1) * frame.height),
            frame.channels
        );
    }
    stbi_write_jpg("./acgist_det.jpg", frame.width, frame.height, frame.channels, frame.data.data(), 80);
}

[[maybe_unused]]
void det_image() {
    caiwei::media::ImageFrame frame;
    if (!frame.load_from_file("./acgist.jpg")) {
        return;
    }
    auto ptr = caiwei::manager::get_context<caiwei::context::DetContext>("yolo26n-det");
    if (ptr == nullptr) {
        return;
    }
    caiwei::test::caiwei_wait_enter();
    {
        CAIWEI_FOR_EACH(1000)
        ptr->run(frame);
        CAIWEI_FOR_EACH_END
    }
    caiwei::test::caiwei_wait_enter();
    {
        CAIWEI_FOR_EACH(1000)
        ptr->run(frame);
        CAIWEI_FOR_EACH_END
    }
    caiwei::test::caiwei_wait_enter();
}

[[maybe_unused]]
void det_image_thread() {
    caiwei::media::ImageFrame frame;
    if (!frame.load_from_file("./acgist.jpg")) {
        return;
    }
    std::vector<std::thread> threads;
    for (int i = 0; i < 2; ++i) {
        threads.push_back(std::thread([&]() {
            auto ptr = caiwei::manager::get_context<caiwei::context::DetContext>("yolo26n-det");
            if (ptr == nullptr) {
                return;
            }
            CAIWEI_FOR_EACH(100)
            ptr->run(frame);
            CAIWEI_FOR_EACH_END
        }));
    }
    for (auto& thread : threads) {
        thread.join();
    }
    caiwei::test::caiwei_wait_enter();
}

[[maybe_unused]]
void det_video() {
    auto ptr = caiwei::manager::get_context<caiwei::context::DetContext>("yolo26n-det");
    if (ptr == nullptr) {
        return;
    }
    size_t frame_count = 0;
    std::vector<caiwei::image::Box> ret;
    caiwei::player::open_player();
    caiwei::media::MediaDemuxer media_demuxer("file", "./caiwei.mp4");
    // caiwei::media::MediaDemuxer media_demuxer("device", R"(audio=麦克风阵列 (适用于数字麦克风的英特尔® 智音技术):video=Integrated Camera)");
    media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        [](caiwei::media::AudioFrame& frame) {
            return caiwei::player::play_audio(frame.data.data(), static_cast<int>(frame.data.size()));
        },
        [&ptr, &ret, &frame_count](caiwei::media::VideoFrame& frame) {
            if (frame_count++ % 4 == 0) {
                ret = ptr->run(frame);
            }
            for (const auto& box : ret) {
                caiwei::test::draw_rect(
                    frame.data.data(),
                    frame.width,
                    frame.height,
                    static_cast<int>(box.x1 * frame.width),
                    static_cast<int>(box.y1 * frame.height),
                    static_cast<int>((box.x2 - box.x1) * frame.width),
                    static_cast<int>((box.y2 - box.y1) * frame.height),
                    frame.channels
                );
            }
            return caiwei::player::play_video(frame.data.data(), frame.width * 3);
        }
    );
    caiwei::player::stop_player();
}

int main(int argc, char* argv[]) {
    #if ENABLE_CAIWEI_RUNTIME_RKNN2
    // caiwei::env::set("CAIWEI_CONTEXT_INFO", "DET,YOLO,yolo26n-det,yolo26n-det-rk3588-i8.rknn");
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "DET,YOLO,yolo26n-det,yolo26n-det-rk3588-f16.rknn");
    #endif
    // caiwei::env::set("CAIWEI_CONTEXT_INFO", "DET,YOLO,yolo26n-det,yolo26n-det-e2e.onnx");
    if (argc > 1) {
        caiwei::env::set("CAIWEI_CONTEXT_INFO", argv[1]);
    }
    caiwei::test::init_test();
    det_draw();
    det_image();
    // det_image_thread();
    // det_image_thread();
    // det_video();
    caiwei::test::stop_test();
    return 0;
}
