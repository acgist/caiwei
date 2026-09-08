#include "caiwei/runtime/onnxruntime.hpp"

#include "caiwei/image_tool.hpp"

caiwei::context::ClsONNXRuntimeContext::ClsONNXRuntimeContext(std::string path, int w, int h, int top_k, int class_size, float confidence_threshold, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime)
 : ClsContext(w, h, top_k, class_size, confidence_threshold, runtime),
   ONNXRuntimeContext(path, runtime->env) {
    this->input_node_dims.push_back(1);
    this->input_node_dims.push_back(3);
    this->input_node_dims.push_back(this->h);
    this->input_node_dims.push_back(this->w);
    this->input_data_length = 3 * this->h * this->w;
}

caiwei::context::ClsONNXRuntimeContext::~ClsONNXRuntimeContext() {
}

std::vector<std::pair<uint32_t, float>> caiwei::context::ClsONNXRuntimeContext::run(const caiwei::media::ImageFrame& image) {
    auto output{ this->run(this->h, this->w, image) };
    float* output_data = output.front().GetTensorMutableData<float>();
    const auto& output_dims = output.front().GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    const int64_t data_length = output_dims[1];
    return caiwei::image::top_k(output_data, data_length, this->top_k, this->confidence_threshold);
}
