#include "test.hpp"

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
#include "caiwei/runtime/rknn2.hpp"
#endif
#ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
#include "caiwei/runtime/onnxruntime.hpp"
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
[[maybe_unused]]
static void test_rknn2(int size, const std::string& path, const std::string& type) {
    const int channels = caiwei::media::IMAGE_CHANNELS;
    const int data_size = channels * size * size;
    caiwei::context::RKNN2Context context(path, channels, size, size);
    CAIWEI_ASSERT(context.load_model());
    if (type == "i8") {
        std::vector<int8_t> data(data_size);
        CAIWEI_FOR_EACH(1000)
        context.run<int8_t>(data.data(), data.size());
        CAIWEI_FOR_EACH_END
    } else if (type == "u8") {
        std::vector<uint8_t> data(data_size);
        CAIWEI_FOR_EACH(1000)
        context.run<uint8_t>(data.data(), data.size());
        CAIWEI_FOR_EACH_END
    } else {
        std::vector<uint16_t> data(data_size);
        CAIWEI_FOR_EACH(1000)
        context.run<uint16_t>(data.data(), data.size());
        CAIWEI_FOR_EACH_END
    }
}
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
[[maybe_unused]]
static void test_onnxruntime(int size, const std::string& path) {
    const int channels = caiwei::media::IMAGE_CHANNELS;
    const int data_size = channels * size * size;
    auto env = std::make_unique<Ort::Env>(caiwei::context::onnxruntime_log_level, "caiwei");
    caiwei::context::ONNXRuntimeContext context(path, channels, size, size, env.get());
    CAIWEI_ASSERT(context.load_model());
    std::vector<float> data(data_size);
    CAIWEI_FOR_EACH(1000)
    context.run(data.data(), data.size());
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
    } else {
        size = 640;
    }
    if (argc > 2) {
        path = argv[2];
    } else {
#if ENABLE_CAIWEI_RUNTIME_RKNN2
        path = "yolo26n-det.rknn";
#endif
#if ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
        path = "yolo26n-det.onnx";
#endif
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
