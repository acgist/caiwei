#include "caiwei/runtime/rknn2.hpp"

#include "caiwei/type.hpp"
#include "caiwei/image_tool.hpp"

caiwei::context::ClsRKNN2Context::ClsRKNN2Context(std::string path, int c, int h, int w, int top_k, int class_size, float confidence_threshold, caiwei::runtime::Runtime* runtime)
 : ClsContext(c, h, w, top_k, class_size, confidence_threshold, runtime),
   RKNN2Context(std::move(path), c, h, w) {
}

caiwei::context::ClsRKNN2Context::~ClsRKNN2Context() {
}

std::vector<std::pair<uint32_t, float>> caiwei::context::ClsRKNN2Context::run(const caiwei::media::ImageFrame& image) {
    std::lock_guard<std::mutex> lock(this->mutex);
    auto output{ this->run(this->h, this->w, image) };
    const auto& output_attr = this->output_attrs[0];
    const int64_t output_data_length = output_attr.dims[1];
    if (output_attr.type != RKNN_TENSOR_FLOAT16) {
        CW_LOG_E("不支持的输出类型: %s", get_type_string(output_attr.type));
        rknn_outputs_release(this->context, output.size(), output.data());
        return {};
    }
    std::vector<float> output_data(output_data_length);
    auto* output_buf = reinterpret_cast<uint16_t*>(output[0].buf);
    caiwei::type::fp16_to_f32(output_data.data(), output_buf, output_data_length);
    rknn_outputs_release(this->context, output.size(), output.data());
    return caiwei::image::top_k(output_data.data(), output_data_length, this->top_k, this->confidence_threshold);
}
