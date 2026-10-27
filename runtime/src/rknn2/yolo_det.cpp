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
) : RKNN2Context(std::move(path), c, h, w),
    DetContext(c, h, w, class_size, iou_threshold, confidence_threshold, runtime) {
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
    std::vector<caiwei::image::Box> ret;
    if (output_attr.type == RKNN_TENSOR_INT8) {
        if (result_length * stride_length != this->out_dst_i8.size()) {
            this->out_dst_i8.resize(result_length * stride_length);
        }
        const auto* out_buf = reinterpret_cast<int8_t*>(output.outputs[0].buf);
        int8_t* data = this->out_dst_i8.data();
        caiwei::image::transpose(out_buf, data, result_length, stride_length);
        const int8_t confidence_threshold_i8 = caiwei::type::qnt_affine_to_i8(this->confidence_threshold, output_attr.zp, output_attr.scale);
        for (int64_t index = 0; index < stride_length; ++index) {
            int    max_class;
            int8_t max_score;
            const int8_t* scores = data + 4;
            caiwei::image::max_loc(scores, this->class_size, max_score, max_class);
            if(max_score > confidence_threshold_i8) {
                const float box_x = (caiwei::type::deqnt_affine_to_f32(data[0], output_attr.zp, output_attr.scale) * this->w - this->pad_w) / (float) this->dst_w;
                const float box_y = (caiwei::type::deqnt_affine_to_f32(data[1], output_attr.zp, output_attr.scale) * this->h - this->pad_h) / (float) this->dst_h;
                const float box_w = (caiwei::type::deqnt_affine_to_f32(data[2], output_attr.zp, output_attr.scale) * this->w              ) / (float) this->dst_w;
                const float box_h = (caiwei::type::deqnt_affine_to_f32(data[3], output_attr.zp, output_attr.scale) * this->h              ) / (float) this->dst_h;
                ret.emplace_back(
                    caiwei::image::Box(
                        std::clamp(box_x - box_w / 2.0F, 0.0F, 1.0F),
                        std::clamp(box_y - box_h / 2.0F, 0.0F, 1.0F),
                        std::clamp(box_x + box_w / 2.0F, 0.0F, 1.0F),
                        std::clamp(box_y + box_h / 2.0F, 0.0F, 1.0F),
                        max_class,
                        caiwei::type::deqnt_affine_to_f32(max_score, output_attr.zp, output_attr.scale)
                    )
                );
            }
            data += result_length;
        }
    } else if (output_attr.type == RKNN_TENSOR_FLOAT16) {
        const auto* out_buf_f16 = reinterpret_cast<uint16_t*>(output.outputs[0].buf);
        if (result_length * stride_length != this->out_buf_f32.size()) {
            this->out_buf_f32.resize(result_length * stride_length);
        }
        if (result_length * stride_length != this->out_dst_f32.size()) {
            this->out_dst_f32.resize(result_length * stride_length);
        }
        float* out_buf = this->out_buf_f32.data();
        float* data    = this->out_dst_f32.data();
        caiwei::type::fp16_to_f32(out_buf, out_buf_f16, result_length * stride_length);
        caiwei::image::transpose(out_buf, data, result_length, stride_length);
        for (int64_t index = 0; index < stride_length; ++index) {
            int   max_class;
            float max_score;
            const float* scores = data + 4;
            caiwei::image::max_loc(scores, this->class_size, max_score, max_class);
            if(max_score > this->confidence_threshold) {
                const float box_x = (data[0] - this->pad_w) / (float) this->dst_w;
                const float box_y = (data[1] - this->pad_h) / (float) this->dst_h;
                const float box_w = (data[2]              ) / (float) this->dst_w;
                const float box_h = (data[3]              ) / (float) this->dst_h;
                ret.emplace_back(
                    caiwei::image::Box(
                        std::clamp(box_x - box_w / 2.0F, 0.0F, 1.0F),
                        std::clamp(box_y - box_h / 2.0F, 0.0F, 1.0F),
                        std::clamp(box_x + box_w / 2.0F, 0.0F, 1.0F),
                        std::clamp(box_y + box_h / 2.0F, 0.0F, 1.0F),
                        max_class,
                        max_score
                    )
                );
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
