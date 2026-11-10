#include "caiwei/log.hpp"
#include "caiwei/image_tool.hpp"
#include "caiwei/runtime/onnxruntime.hpp"

#ifndef CAIWEI_END2END_STRIDE_LENGTH
#define CAIWEI_END2END_STRIDE_LENGTH 6
#endif

caiwei::context::DetONNXRuntimeContext::DetONNXRuntimeContext(
    std::string path,
    int c,
    int h,
    int w,
    int class_size,
    float iou_threshold,
    float confidence_threshold,
    Ort::Env* env,
    caiwei::runtime::Runtime* runtime
) : DetContext(c, h, w, class_size, iou_threshold, confidence_threshold, runtime),
    ONNXRuntimeContext(std::move(path), c, h, w, env) {
}

caiwei::context::DetONNXRuntimeContext::~DetONNXRuntimeContext() {
}

bool caiwei::context::DetONNXRuntimeContext::load() {
    return this->load_model();
}

std::vector<caiwei::image::Box> caiwei::context::DetONNXRuntimeContext::run(const caiwei::media::ImageFrame& image) {
    auto output{ this->run(this->h, this->w, image) };
    if (output.empty()) {
        return {};
    }
    const auto&  output_dims = output.front().GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    if (output_dims.size() != 3) {
        CW_LOG_W("模型输出维度错误: %zu", output_dims.size());
        return {};
    }
    const int64_t result_length = output_dims[1];
    const int64_t stride_length = output_dims[2];
    const float* output_data = output.front().GetTensorData<float>();
    std::vector<caiwei::image::Box> ret;
    if (stride_length == CAIWEI_END2END_STRIDE_LENGTH) {
        const float* data = output_data;
        for (int64_t index = 0; index < result_length; ++index) {
            int   max_class = static_cast<int>(data[5]);
            float max_score = data[4];
            if(max_score > this->confidence_threshold) {
                const float box_x1 = (data[0] - this->pad_w) / static_cast<float>(this->dst_w);
                const float box_y1 = (data[1] - this->pad_h) / static_cast<float>(this->dst_h);
                const float box_x2 = (data[2] - this->pad_w) / static_cast<float>(this->dst_w);
                const float box_y2 = (data[3] - this->pad_h) / static_cast<float>(this->dst_h);
                ret.emplace_back(caiwei::image::to_box_end2end(box_x1, box_y1, box_x2, box_y2, max_class, max_score));
            }
            data += stride_length;
        }
        return ret;
    } else if (result_length == this->class_size + 4) {
        const size_t total = static_cast<size_t>(result_length) * stride_length;
        if (total != this->out_dst.size()) {
            this->out_dst.resize(total);
        }
        caiwei::image::transpose(output_data, this->out_dst.data(), result_length, stride_length);
        const float* data = this->out_dst.data();
        for (int64_t index = 0; index < stride_length; ++index) {
            int   max_class;
            float max_score;
            const float* scores = data + 4;
            caiwei::image::max_loc(scores, this->class_size, max_score, max_class);
            if(max_score > this->confidence_threshold) {
                const float box_x = (data[0] - this->pad_w) / static_cast<float>(this->dst_w);
                const float box_y = (data[1] - this->pad_h) / static_cast<float>(this->dst_h);
                const float box_w = (data[2]              ) / static_cast<float>(this->dst_w);
                const float box_h = (data[3]              ) / static_cast<float>(this->dst_h);
                ret.emplace_back(caiwei::image::to_box(box_x, box_y, box_w, box_h, max_class, max_score));
            }
            data += result_length;
        }
        auto index = caiwei::image::nms_boxes(ret, this->iou_threshold);
        std::vector<caiwei::image::Box> result;
        result.resize(index.size());
        size_t j = 0;
        for (size_t i : index) {
            result[j++] = ret[i];
        }
        return result;
    } else {
        CW_LOG_W("不支持的维度: %" PRId64 " - %" PRId64, result_length, stride_length);
        return {};
    }
}
