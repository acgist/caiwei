#include "test.hpp"

#include <thread>

[[maybe_unused]]
void test_manager() {
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.push_back(std::thread([i]() {
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

void test_manager_optimize() {
    auto ptr1 = caiwei::manager::get_context<caiwei::context::ClsContext>("yolo26n-cls");
    assert(ptr1 != nullptr);
    auto ptr2 = caiwei::manager::get_context<caiwei::context::ClsContext>("yolo26n-cls");
    assert(ptr2 != nullptr);
    auto ptr3 = caiwei::manager::get_context<caiwei::context::ClsContext>("yolo26n-cls");
    assert(ptr3 != nullptr);
    ptr3 = nullptr;
    auto ptr4 = caiwei::manager::get_context<caiwei::context::ClsContext>("yolo26n-cls");
    assert(ptr4 != nullptr);
    auto ptr5 = caiwei::manager::get_context<caiwei::context::ClsContext>("yolo26n-cls");
    assert(ptr5 == nullptr);
    ptr4 = nullptr;
    std::this_thread::sleep_for(std::chrono::minutes(60));
}

int main() {
    caiwei::env::set("CAIWEI_ONNXRUNTIME_MIN_POOL", "2");
    caiwei::env::set("CAIWEI_ONNXRUNTIME_MAX_POOL", "3");
    caiwei::test::init_test();
    // test_manager();
    test_manager_optimize();
    caiwei::test::stop_test();
    return 0;
}
