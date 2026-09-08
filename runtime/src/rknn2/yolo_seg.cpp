#include "caiwei/runtime/rknn2.hpp"

#include "caiwei/image_tool.hpp"

caiwei::context::SegRKNN2Context::SegRKNN2Context(std::string path, int w, int h, int top_k, int class_size, float confidence_threshold, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime)
 : SegContext(w, h, top_k, class_size, confidence_threshold, runtime),
   RKNN2Context(path) {
    this->input_data_length = 3 * this->h * this->w;
}

caiwei::context::SegRKNN2Context::~SegRKNN2Context() {
}

std::vector<caiwei::image::Seg> caiwei::context::SegRKNN2Context::run(const caiwei::media::ImageFrame& image) {
    return {};
}
