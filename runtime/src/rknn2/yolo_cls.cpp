#include "caiwei/log.hpp"
#include "caiwei/media_tool.hpp"
#include "caiwei/runtime/rknn2.hpp"
#include "caiwei/type.hpp"

caiwei::context::ClsRKNN2Context::ClsRKNN2Context(
    std::string path,
    int c,
    int h,
    int w,
    int class_size,
    int top_k,
    float confidence_threshold,
    caiwei::runtime::Runtime* runtime
) :
    ClsContext(c, h, w, class_size, top_k, confidence_threshold, runtime),
    RKNN2Context(std::move(path), c, h, w) {
}

caiwei::context::ClsRKNN2Context::~ClsRKNN2Context() {
}

bool caiwei::context::ClsRKNN2Context::load() {
    return this->load_model();
}

std::vector<caiwei::media::Cls> caiwei::context::ClsRKNN2Context::run(const caiwei::media::ImageFrame& image) {
    auto output {this->run(this->h, this->w, image)};
    if (output.outputs.empty()) {
        return {};
    }
    const auto& output_attr = this->output_attrs[0];
    if (output_attr.type != RKNN_TENSOR_FLOAT16) {
        CW_LOG_W("不支持的输出类型: %s", get_type_string(output_attr.type));
        return {};
    }
    const int64_t data_length = output_attr.dims[1];
    if (data_length != this->class_size) {
        CW_LOG_W("类型数量错误: %" PRId64 " != %d", data_length, this->class_size);
        return {};
    }
    if (static_cast<size_t>(data_length) != this->out_dst.size()) {
        this->out_dst.resize(data_length);
    }
    const auto* out_buf = reinterpret_cast<uint16_t*>(output.outputs[0].buf);
    auto* output_data = this->out_dst.data();
    caiwei::type::fp16_to_f32(output_data, out_buf, data_length);
    return caiwei::media::top_k<float, caiwei::media::Cls>(output_data, data_length, this->top_k, this->confidence_threshold);
}
