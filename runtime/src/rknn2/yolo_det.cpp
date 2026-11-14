#include "caiwei/log.hpp"
#include "caiwei/type.hpp"
#include "caiwei/image_tool.hpp"
#include "caiwei/runtime/rknn2.hpp"

caiwei::context::DetRKNN2Context::DetRKNN2Context(
    std::string path,
    int c,
    int h,
    int w,
    int class_size,
    float iou_threshold,
    float confidence_threshold,
    caiwei::runtime::Runtime* runtime
) : DetContext(c, h, w, class_size, iou_threshold, confidence_threshold, runtime),
    RKNN2Context(std::move(path), c, h, w) {
}

caiwei::context::DetRKNN2Context::~DetRKNN2Context() {
}

bool caiwei::context::DetRKNN2Context::load() {
    return this->load_model();
}

std::vector<caiwei::image::Box> caiwei::context::DetRKNN2Context::run(const caiwei::media::ImageFrame& image) {
    auto output{ this->run(this->h, this->w, image) };
    if (output.outputs.empty()) {
        return {};
    }
    const auto& output_attr = this->output_attrs[0];
    const int64_t result_length = output_attr.dims[1];
    const int64_t stride_length = output_attr.dims[2];
    if (result_length != this->class_size + YOLO_BOX_SIZE) {
        CW_LOG_W("输出长度错误: %" PRId64 " - %d", result_length, this->class_size);
        return {};
    }
    std::vector<caiwei::image::Box> ret;
    if (output_attr.type == RKNN_TENSOR_INT8) {
        const size_t total = static_cast<size_t>(result_length) * stride_length;
        if (total != this->out_dst_i8.size()) {
            this->out_dst_i8.resize(total);
        }
        const auto* out_buf = reinterpret_cast<int8_t*>(output.outputs[0].buf);
        int8_t* data = this->out_dst_i8.data();
        caiwei::image::transpose(out_buf, data, result_length, stride_length);
        const int32_t zp    = output_attr.zp;
        const float   scale = output_attr.scale;
        const int8_t confidence_threshold_i8 = caiwei::type::qnt_affine_to_i8(this->confidence_threshold, zp, scale);
        for (int64_t index = 0; index < stride_length; ++index) {
            const int8_t* scores = data + YOLO_BOX_SIZE;
            auto   max_ptr      = std::max_element(scores, scores + this->class_size);
            int8_t max_score_i8 = *max_ptr;
            int    max_class    = std::distance(scores, max_ptr);
            if(max_score_i8 > confidence_threshold_i8) {
                // I8坐标归一化到[0,1]需要乘以高宽还原
                const float box_x = (caiwei::type::deqnt_affine_to_f32(data[0], zp, scale) * this->w - this->pad_w) / static_cast<float>(this->dst_w);
                const float box_y = (caiwei::type::deqnt_affine_to_f32(data[1], zp, scale) * this->h - this->pad_h) / static_cast<float>(this->dst_h);
                const float box_w = (caiwei::type::deqnt_affine_to_f32(data[2], zp, scale) * this->w              ) / static_cast<float>(this->dst_w);
                const float box_h = (caiwei::type::deqnt_affine_to_f32(data[3], zp, scale) * this->h              ) / static_cast<float>(this->dst_h);
                const float max_score = caiwei::type::deqnt_affine_to_f32(max_score_i8, zp, scale);
                ret.emplace_back(caiwei::image::to_box(box_x, box_y, box_w, box_h, max_class, max_score));
            }
            data += result_length;
        }
    } else if (output_attr.type == RKNN_TENSOR_FLOAT16) {
        const size_t total = static_cast<size_t>(result_length) * stride_length;
        const auto* out_buf_f16 = reinterpret_cast<uint16_t*>(output.outputs[0].buf);
        if (total != this->out_buf_f32.size()) {
            this->out_buf_f32.resize(total);
        }
        if (total != this->out_dst_f32.size()) {
            this->out_dst_f32.resize(total);
        }
        float* out_buf = this->out_buf_f32.data();
        float* data    = this->out_dst_f32.data();
        caiwei::type::fp16_to_f32(out_buf, out_buf_f16, total);
        caiwei::image::transpose(out_buf, data, result_length, stride_length);
        for (int64_t index = 0; index < stride_length; ++index) {
            const float* scores = data + YOLO_BOX_SIZE;
            auto  max_ptr   = std::max_element(scores, scores + this->class_size);
            float max_score = *max_ptr;
            int   max_class = std::distance(scores, max_ptr);
            if(max_score > this->confidence_threshold) {
                // F16坐标没有归一化到[0,1]直接使用
                const float box_x = (data[0] - this->pad_w) / static_cast<float>(this->dst_w);
                const float box_y = (data[1] - this->pad_h) / static_cast<float>(this->dst_h);
                const float box_w = (data[2]              ) / static_cast<float>(this->dst_w);
                const float box_h = (data[3]              ) / static_cast<float>(this->dst_h);
                ret.emplace_back(caiwei::image::to_box(box_x, box_y, box_w, box_h, max_class, max_score));
            }
            data += result_length;
        }
    } else {
        CW_LOG_W("不支持的输出类型: %s", get_type_string(output_attr.type));
        return {};
    }
    auto index = caiwei::image::nms_boxes(ret, this->iou_threshold);
    std::vector<caiwei::image::Box> result;
    result.resize(index.size());
    size_t j = 0;
    for (size_t i : index) {
        result[j++] = ret[i];
    }
    return result;
}
