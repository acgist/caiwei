#include "caiwei/runtime/rknn2.hpp"

#include "caiwei/type.hpp"
#include "caiwei/image_tool.hpp"

caiwei::context::DetRKNN2Context::DetRKNN2Context(std::string path, int c, int h, int w, int class_size, float iou_threshold, float confidence_threshold, caiwei::runtime::Runtime* runtime)
 : DetContext(c, h, w, class_size, iou_threshold, confidence_threshold, runtime),
   RKNN2Context(std::move(path), c, h, w) {
}

caiwei::context::DetRKNN2Context::~DetRKNN2Context() {
}

std::vector<caiwei::image::Box> caiwei::context::DetRKNN2Context::run(const caiwei::media::ImageFrame& image) {
    std::lock_guard<std::mutex> lock(this->mutex);
    auto output{ this->run(this->h, this->w, image) };
    const auto& output_attr = this->output_attrs[0];
    const int64_t result_length = output_attr.dims[1];
    const int64_t stride_length = output_attr.dims[2];
    std::vector<caiwei::image::Box> ret;
    if (output_attr.type == RKNN_TENSOR_INT8) {
        auto* buf_i8 = reinterpret_cast<int8_t*>(output[0].buf);
        int8_t qnt_confidence_threshold = caiwei::type::qnt_affine_to_i8(this->confidence_threshold, output_attr.zp, output_attr.scale);
        // TODO
        std::vector<int8_t> out_dst;
        out_dst.resize(result_length * stride_length);
        caiwei::image::transpose(buf_i8, out_dst.data(), result_length, stride_length);
        int8_t* data = out_dst.data();
        for (int index = 0; index < stride_length; ++index) {
            int    max_class;
            int8_t max_score;
            int8_t* scores = data + 4;
            caiwei::image::max_loc(scores, this->class_size, max_score, max_class);
            if(max_score > qnt_confidence_threshold) {
                float box_x = (caiwei::type::deqnt_affine_to_f32(data[0], output_attr.zp, output_attr.scale) * this->w - this->pad_w) / (float) this->dst_w;
                float box_y = (caiwei::type::deqnt_affine_to_f32(data[1], output_attr.zp, output_attr.scale) * this->h - this->pad_h) / (float) this->dst_h;
                float box_w = (caiwei::type::deqnt_affine_to_f32(data[2], output_attr.zp, output_attr.scale) * this->w              ) / (float) this->dst_w;
                float box_h = (caiwei::type::deqnt_affine_to_f32(data[3], output_attr.zp, output_attr.scale) * this->h              ) / (float) this->dst_h;
                ret.push_back(
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
        rknn_outputs_release(this->context, output.size(), output.data());
    } else if (output_attr.type == RKNN_TENSOR_FLOAT16) {
        auto* buf_f16 = reinterpret_cast<uint16_t*>(output[0].buf);
        std::vector<float> buf_f32(result_length * stride_length);
        caiwei::type::fp16_to_f32(buf_f32.data(), buf_f16, result_length * stride_length);
        rknn_outputs_release(this->context, output.size(), output.data());
        float* output_data = buf_f32.data();
        // TODO
        std::vector<float> out_dst;
        out_dst.resize(result_length * stride_length);
        caiwei::image::transpose(output_data, out_dst.data(), result_length, stride_length);
        float* data = out_dst.data();
        for (int index = 0; index < stride_length; ++index) {
            int   max_class;
            float max_score;
            float* scores = data + 4;
            caiwei::image::max_loc(scores, this->class_size, max_score, max_class);
            if(max_score > this->confidence_threshold) {
                float box_x = (data[0] - this->pad_w) / (float) this->dst_w;
                float box_y = (data[1] - this->pad_h) / (float) this->dst_h;
                float box_w = (data[2]              ) / (float) this->dst_w;
                float box_h = (data[3]              ) / (float) this->dst_h;
                ret.push_back(
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
        CW_LOG_E("不支持的输出类型: %s", get_type_string(output_attr.type));
        rknn_outputs_release(this->context, output.size(), output.data());
        return {};
    }
    auto index = caiwei::image::nms_boxes(ret, this->iou_threshold);
    std::vector<caiwei::image::Box> result;
    result.resize(index.size());
    int j = 0;
    for (size_t i : index) {
        result[j] = ret[i];
        ++j;
    }
    return result;
}
