#include "caiwei/log.hpp"
#include "caiwei/image_tool.hpp"
#include "caiwei/runtime/onnxruntime.hpp"

caiwei::context::ClsONNXRuntimeContext::ClsONNXRuntimeContext(
    std::string path,
    int c,
    int h,
    int w,
    int class_size,
    int top_k,
    float confidence_threshold,
    Ort::Env* env,
    caiwei::runtime::Runtime* runtime
) : ClsContext(c, h, w, class_size, top_k, confidence_threshold, runtime),
    ONNXRuntimeContext(std::move(path), c, h, w, env) {
}

caiwei::context::ClsONNXRuntimeContext::~ClsONNXRuntimeContext() {
}

bool caiwei::context::ClsONNXRuntimeContext::load() {
    return this->load_model();
}

std::vector<caiwei::image::Cls> caiwei::context::ClsONNXRuntimeContext::run(const caiwei::media::ImageFrame& image) {
    auto output{ this->run(this->h, this->w, image) };
    if (output.empty()) {
        return {};
    }
    const auto& output_dims = output.front().GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    if (output_dims.size() != 2) {
        CW_LOG_W("模型输出维度错误: %zu", output_dims.size());
        return {};
    }
    const int64_t data_length = output_dims[1];
    if (data_length != this->class_size) {
        CW_LOG_W("类型数量错误: %llu != %d", data_length, this->class_size);
        return {};
    }
    const float* output_data = output.front().GetTensorData<float>();
    return caiwei::image::top_k<float, caiwei::image::Cls>(output_data, data_length, this->top_k, this->confidence_threshold);
}
