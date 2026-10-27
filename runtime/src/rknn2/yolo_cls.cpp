#include "caiwei/log.hpp"
#include "caiwei/type.hpp"
#include "caiwei/image_tool.hpp"
#include "caiwei/runtime/rknn2.hpp"

caiwei::context::ClsRKNN2Context::ClsRKNN2Context(
    std::string path,
    int c,
    int h,
    int w,
    int class_size,
    int top_k,
    float confidence_threshold,
    caiwei::runtime::Runtime* runtime
) : RKNN2Context(std::move(path), c, h, w),
    ClsContext(c, h, w, class_size, top_k, confidence_threshold, runtime) {
}

caiwei::context::ClsRKNN2Context::~ClsRKNN2Context() {
}

bool caiwei::context::ClsRKNN2Context::load() {
    return this->load_model();
}

std::vector<caiwei::image::Cls> caiwei::context::ClsRKNN2Context::run(const caiwei::media::ImageFrame& image) {
    auto output{ this->run(this->h, this->w, image) };
    if (output.outputs.empty()) {
        return {};
    }
    const auto& output_attr = this->output_attrs[0];
    if (output_attr.type != RKNN_TENSOR_FLOAT16) {
        CW_LOG_W("不支持的输出类型: %s", get_type_string(output_attr.type));
        return {};
    }
    const int64_t out_dst_length = output_attr.dims[1];
    if (out_dst_length != this->out_dst.size()) {
        this->out_dst.resize(out_dst_length);
    }
    const auto* out_buf = reinterpret_cast<uint16_t*>(output.outputs[0].buf);
    caiwei::type::fp16_to_f32(this->out_dst.data(), out_buf, out_dst_length);
    return caiwei::image::top_k<float, caiwei::image::Cls>(this->out_dst.data(), out_dst_length, this->top_k, this->confidence_threshold);
}
