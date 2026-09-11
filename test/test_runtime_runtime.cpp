#include "test.hpp"

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
#include "caiwei/runtime/rknn2.hpp"

[[maybe_unused]]
void test_rknn2() {
    caiwei::context::RKNN2Context context("yolo26n-det.rknn", 3, 640, 640);
    std::vector<uint8_t> data(3 * 640 * 640);
    CAIWEI_FOR_EACH(100)
    context.run(data.data());
    CAIWEI_FOR_EACH_END
}
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
#include "caiwei/runtime/onnxruntime.hpp"

[[maybe_unused]]
void test_onnxruntime() {
    auto* env = new Ort::Env(caiwei::context::onnxruntime_log_level, "caiwei");
    caiwei::context::ONNXRuntimeContext context("yolo26n-det.onnx", 3, 640, 640, env);
    std::vector<float> data(3 * 640 * 640);
    CAIWEI_FOR_EACH(100)
    auto result = context.run(data.data());
    CAIWEI_FOR_EACH_END
}
#endif

int main() {
    caiwei::test::init_test();
    #ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
    // detect
    // RK3588 24~28 ms yolo11n-i8(瑞芯微导出版本)
    // RK3588 36~42 ms yolo11n-i8(YOLO官网导出版本)
    // RK3588 38~44 ms yolo26n-i8(YOLO官网导出版本)
    // RK3588 94~100 ms yolo26n-f16(YOLO官网导出版本)
    test_rknn2();
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
    // detect
    // CUDA: 8~10 ms
    // CUDA: 8~10 ms e2e
    test_onnxruntime();
    #endif
    caiwei::test::stop_test();
    return 0;
}
