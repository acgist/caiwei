#include "caiwei/runtime/rknn2.hpp"

#include "caiwei/image_tool.hpp"

caiwei::context::ClsRKNN2Context::ClsRKNN2Context(std::string path, int w, int h, int top_k, int class_size, float confidence_threshold, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime)
 : ClsContext(w, h, top_k, class_size, confidence_threshold, runtime),
   RKNN2Context(path) {
    this->input_data_length = 3 * this->h * this->w;
}

caiwei::context::ClsRKNN2Context::~ClsRKNN2Context() {
}

std::vector<std::pair<uint32_t, float>> caiwei::context::ClsRKNN2Context::run(const caiwei::media::ImageFrame& image) {
    auto output{ this->run(this->h, this->w, image) };
    const auto& output_attr = this->output_attrs[0];
    uint16_t* output_data = reinterpret_cast<uint16_t*>(output[0].buf);
    const int64_t data_length = output_attr.dims[1];
    std::vector<float> data(data_length);
    fp16_to_f32(data.data(), output_data, data_length);
    rknn_outputs_release(this->context, output.size(), output.data());
    return caiwei::image::top_k(data.data(), data_length, this->top_k, this->confidence_threshold);
}
