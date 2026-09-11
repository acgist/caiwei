#include "test.hpp"

#include <thread>

[[maybe_unused]]
void test_manager() {
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.push_back(std::thread([i]() {
            auto ptr = caiwei::manager::get_context<caiwei::context::ClsContext, caiwei::media::ImageFrame, std::vector<std::pair<uint32_t, float>>>("yolo26n-cls");
            if (ptr) {
                CW_LOG_I("获取context: %d", i);
                std::this_thread::sleep_for(std::chrono::seconds(4));
            } else {
                CW_LOG_E("获取context失败: %d", i);
            }
            std::fflush(stdout);
        }));
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

[[maybe_unused]]
void test_manager_get() {
    CAIWEI_FOR_EACH(100'000)
    caiwei::manager::get_context<caiwei::context::ClsContext, caiwei::media::ImageFrame, std::vector<std::pair<uint32_t, float>>>("yolo26n-cls");
    CAIWEI_FOR_EACH_END
}

int main() {
    caiwei::env::set("CAIWEI_ONNXRUNTIME_MIN_POOL", "2");
    caiwei::env::set("CAIWEI_ONNXRUNTIME_MAX_POOL", "2");
    caiwei::test::init_test();
    // test_manager();
    test_manager_get();
    caiwei::test::stop_test();
    return 0;
}
