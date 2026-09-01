#include "onnxruntime.hpp"

#include <filesystem>

template <>
std::shared_ptr<caiwei::context::ClsContext> caiwei::context::get_context(caiwei::context::Type type, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) {
    int w = caiwei::env::get_int("CAIWEI_CLS_W");
    int h = caiwei::env::get_int("CAIWEI_CLS_H");
    int top_k = caiwei::env::get_int("CAIWEI_CLS_TOP_K");
    int class_size = caiwei::env::get_int("CAIWEI_CLS_CLASS_SIZE");
    float confidence_threshold = caiwei::env::get_float("CAIWEI_CLS_CONFIDENCE_THRESHOLD");
    std::string path = caiwei::env::get_string("CAIWEI_CLS_PATH");
    if (!std::filesystem::exists(path)) {
        CW_LOG_W("ONNXRtuntime模型无效: %s", path.c_str());
        return nullptr;
    }
    return std::make_shared<ClsONNXRuntimeContext>(path, w, h, top_k, class_size, confidence_threshold, runtime);
}

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
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    // TODO 全局变量 判断是否变化
    caiwei::transform::resize(image.width, image.height, this->w, this->h, dst_w, dst_h, pad_w, pad_h, scale);
    std::vector<uint8_t> dst(dst_w   * dst_h   * image.channels);
    std::vector<uint8_t> pad(this->w * this->h * image.channels);
    std::vector<float>   hwc(this->w * this->h * image.channels);
    std::vector<float>   chw(this->w * this->h * image.channels);
    caiwei::transform::resize(image.data.data(), dst.data(), image.width, image.height, dst_w, dst_h);
    caiwei::transform::padding(dst.data(), pad.data(), dst_w, dst_h, pad_w, pad_h, this->w, this->h);
    caiwei::transform::i8_to_f32(pad.data(), this->w * this->h * image.channels, hwc.data(), 255.0F);
    caiwei::transform::hwc_to_chw(hwc.data(), chw.data(), this->h, this->w, image.channels);
    std::vector<int64_t> dims;
    auto output{ this->run(chw.data(), dims) };
    const int64_t& data_length = dims[1];
    float* output_data = output.GetTensorMutableData<float>();
    // TODO
    return caiwei::transform::top_k(output_data, data_length, this->top_k, this->confidence_threshold);
}
