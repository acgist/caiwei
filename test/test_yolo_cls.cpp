#include "test.hpp"

#include <thread>
#include <vector>

[[maybe_unused]]
void cls_image() {
    caiwei::media::ImageFrame frame;
    if (!frame.load_from_file("./samoyed.jpg")) {
        return;
    }
    auto ptr = caiwei::manager::get_context<caiwei::context::ClsContext>("yolo26n-cls");
    if (ptr == nullptr) {
        return;
    }
    auto result = ptr->run(frame);
    ALWAYS_ASSERT(!result.empty());
    for (const auto& cls : result) {
        CW_LOG_I("cls: %d = %.2f", cls.class_id, cls.score);
    }
    // 258 == samoyed
    ALWAYS_ASSERT(result[0].class_id== 258);
}

[[maybe_unused]]
void cls_image_for(const char* image_path) {
    caiwei::media::ImageFrame frame;
    if (!frame.load_from_file(image_path)) {
        return;
    }
    auto ptr = caiwei::manager::get_context<caiwei::context::ClsContext>("yolo26n-cls");
    if (ptr == nullptr) {
        return;
    }
    caiwei::test::caiwei_wait_enter();
    CW_LOG_I("图片大小: %dx%d", frame.width, frame.height);
    {
        CAIWEI_FOR_EACH(1000)
        ptr->run(frame);
        CAIWEI_FOR_EACH_END
    }
    caiwei::test::caiwei_wait_enter();
    CW_LOG_I("图片大小: %dx%d", frame.width, frame.height);
    {
        CAIWEI_FOR_EACH(1000)
        ptr->run(frame);
        CAIWEI_FOR_EACH_END
    }
    caiwei::test::caiwei_wait_enter();
}

[[maybe_unused]]
void cls_image_thread(const char* image_path) {
    caiwei::media::ImageFrame frame;
    if (!frame.load_from_file(image_path)) {
        return;
    }
    std::vector<std::jthread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.push_back(std::jthread([&]() {
            auto ptr = caiwei::manager::get_context<caiwei::context::ClsContext>("yolo26n-cls");
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
void cls_video() {
    auto ptr = caiwei::manager::get_context<caiwei::context::ClsContext>("yolo26n-cls");
    if (ptr == nullptr) {
        return;
    }
    ALWAYS_ASSERT(caiwei::player::open_player());
    caiwei::media::MediaDemuxer media_demuxer("file", "./caiwei.mp4");
    bool success = media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        [](caiwei::media::AudioFrame& frame) {
            return caiwei::player::play_audio(frame.data.data(), static_cast<int>(frame.data.size()));
        },
        [&ptr](caiwei::media::VideoFrame& frame) {
            auto ret = ptr->run(frame);
            for (const auto& index : ret) {
                CW_LOG_I("cls: %d = %.2f", index.class_id, index.score);
            }
            return caiwei::player::play_video(frame.data.data(), frame.width * 3);
        }
    );
    if (!success) {
        CW_LOG_W("视频加载失败");
    }
    caiwei::player::stop_player();
}

int main(int argc, char* argv[]) {
    #if ENABLE_CAIWEI_RUNTIME_RKNN2
    // caiwei::env::set("CAIWEI_CONTEXT_INFO", "CLS,YOLO,yolo26n-cls,yolo26n-cls-rk3588-i8.rknn");
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "CLS,YOLO,yolo26n-cls,yolo26n-cls-rk3588-f16.rknn");
    #endif
    if (argc > 1) {
        caiwei::env::set("CAIWEI_CONTEXT_INFO", argv[1]);
    }
    const char* image_path;
    if (argc > 2) {
        image_path = argv[2];
    } else {
        image_path = "./acgist.jpg";
    }
    caiwei::test::init_test();
    cls_image();
    cls_image_for(image_path);
    // cls_image_thread(image_path);
    // cls_image_thread(image_path);
    // cls_video();
    caiwei::test::stop_test();
    return 0;
}
