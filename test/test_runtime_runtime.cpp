#include "test.hpp"

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
#include "caiwei/runtime/rknn2.hpp"
#endif
#ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
#include "caiwei/runtime/onnxruntime.hpp"
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
[[maybe_unused]]
void test_rknn2(int size, const std::string& path, const std::string& type) {
    size = size == 0 ? 640 : size;
    caiwei::context::RKNN2Context context(path.empty() ? "yolo26n-det.rknn" : path, 3, size, size);
    context.load_model();
    if (type == "i8") {
        std::vector<int8_t> data(3 * size * size);
        CAIWEI_FOR_EACH(1000)
        auto result = context.run(data.data(), data.size());
        CAIWEI_FOR_EACH_END
    } else if (type == "u8") {
        std::vector<uint8_t> data(3 * size * size);
        CAIWEI_FOR_EACH(1000)
        auto result = context.run(data.data(), data.size());
        CAIWEI_FOR_EACH_END
    } else {
        std::vector<float> data(3 * size * size);
        CAIWEI_FOR_EACH(1000)
        auto result = context.run(data.data(), data.size());
        CAIWEI_FOR_EACH_END
    }
}
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
[[maybe_unused]]
void test_onnxruntime(int size, const std::string& path) {
    size = size == 0 ? 640 : size;
    auto* env = new Ort::Env(caiwei::context::onnxruntime_log_level, "caiwei");
    caiwei::context::ONNXRuntimeContext context(path.empty() ? "yolo26n-det.onnx" : path, 3, size, size, env);
    context.load_model();
    std::vector<float> data(3 * size * size);
    CAIWEI_FOR_EACH(1000)
    auto result = context.run(data.data(), data.size());
    CAIWEI_FOR_EACH_END
}
#endif

int main(int argc, char* argv[]) {
    caiwei::test::init_test();
    int size = 0;
    std::string path;
    std::string type;
    if (argc > 1) {
        size = std::stoi(argv[1]);
    }
    if (argc > 2) {
        path = argv[2];
    }
    if (argc > 3) {
        type = argv[3];
    }
    #ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
    test_rknn2(size, path, type);
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
    test_onnxruntime(size, path);
    #endif
    caiwei::test::stop_test();
    return 0;
}
