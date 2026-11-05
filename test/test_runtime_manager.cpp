#include "test.hpp"

#include <thread>

[[maybe_unused]]
void test_manager() {
    std::vector<std::jthread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.push_back(std::jthread([i]() {
            auto ptr = caiwei::manager::get_context<caiwei::context::ClsContext>("yolo26n-cls");
            if (ptr) {
                CW_LOG_I("获取context成功: %d", i);
                std::this_thread::sleep_for(std::chrono::seconds(4));
            } else {
                CW_LOG_W("获取context失败: %d", i);
            }
        }));
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

[[maybe_unused]]
void test_manager_init() {
    caiwei::runtime::init();
    caiwei::runtime::init();
    caiwei::runtime::stop();
    caiwei::runtime::init();
}

[[maybe_unused]]
void test_manager_optimize() {
    auto ptr1 = caiwei::manager::get_context<caiwei::context::ClsContext>("yolo26n-cls");
    assert(ptr1 != nullptr);
    auto ptr2 = caiwei::manager::get_context<caiwei::context::ClsContext>("yolo26n-cls");
    assert(ptr2 != nullptr);
    auto ptr3 = caiwei::manager::get_context<caiwei::context::ClsContext>("yolo26n-cls");
    assert(ptr3 != nullptr);
    auto ptr4 = caiwei::manager::get_context<caiwei::context::ClsContext>("yolo26n-cls");
    assert(ptr4 != nullptr);
    ptr4 = nullptr;
    auto ptr5 = caiwei::manager::get_context<caiwei::context::ClsContext>("yolo26n-cls");
    assert(ptr5 != nullptr);
    auto ptr6 = caiwei::manager::get_context<caiwei::context::ClsContext>("yolo26n-cls");
    assert(ptr6 == nullptr);
    ptr5 = nullptr;
    caiwei::test::caiwei_wait_enter();
}

[[maybe_unused]]
void test_log() {
    std::vector<std::jthread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.push_back(std::jthread([i]() {
            CAIWEI_FOR_EACH(10000)
            CW_LOG_I("测试日志: %d", caiwei_index);
            CAIWEI_FOR_EACH_END
        }));
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

int main() {
    caiwei::env::set("CAIWEI_ONNXRUNTIME_MIN_POOL", "2");
    caiwei::env::set("CAIWEI_ONNXRUNTIME_MAX_POOL", "4");
    caiwei::test::init_test();
    // test_manager();
    // test_manager_init();
    // test_manager_optimize();
    test_log();
    caiwei::test::stop_test();
    return 0;
}
