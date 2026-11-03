#include "test.hpp"

#include <thread>
#include <vector>

[[maybe_unused]]
void cls_image() {
    caiwei::media::ImageFrame frame;
    if (!frame.load_from_file("./samoyed.jpg")) {
        CW_LOG_W("图片加载失败");
        return;
    }
    auto ptr = caiwei::manager::get_context<caiwei::context::ClsContext>("yolo26n-cls");
    if (ptr == nullptr) {
        CW_LOG_W("模型加载失败");
        return;
    }
    auto result = ptr->run(frame);
    assert(!result.empty());
    for (const auto& cls : result) {
        CW_LOG_I("cls: %d = %.2f", cls.class_id, cls.score);
    }
    // 258 == samoyed
    assert(result[0].class_id== 258);
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
void cls_image_thread() {
    caiwei::media::ImageFrame frame;
    if (!frame.load_from_file("./samoyed.jpg")) {
        CW_LOG_W("图片加载失败");
        return;
    }
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.push_back(std::thread([&]() {
            // 独占模型非线程安全每个线程独立一个模型实例
            auto ptr = caiwei::manager::get_context<caiwei::context::ClsContext>("yolo26n-cls");
            if (ptr == nullptr) {
                CW_LOG_W("模型加载失败");
                return;
            }
            CAIWEI_FOR_EACH(100)
            auto result = ptr->run(frame);
            assert(result[0].class_id== 258);
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
        CW_LOG_W("模型加载失败");
        return;
    }
    caiwei::player::open_player();
    caiwei::media::MediaDemuxer media_demuxer("file", "./caiwei.mp4");
    // caiwei::media::MediaDemuxer media_demuxer("device", R"(audio=麦克风阵列 (适用于数字麦克风的英特尔® 智音技术):video=Integrated Camera)");
    bool success = media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        [](caiwei::media::AudioFrame& frame) {
            return caiwei::player::play_audio(frame.data.data(), static_cast<int>(frame.data.size()));
        },
        [&ptr](caiwei::media::VideoFrame& frame) {
            auto ret = ptr->run(frame);
            for (const auto& index : ret) {
                CW_LOG_I("%d = %.2f", index.class_id, index.score);
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
    caiwei::test::init_test();
    cls_image();
    // cls_image_thread();
    // cls_image_thread();
    // cls_video();
    caiwei::test::stop_test();
    return 0;
}
