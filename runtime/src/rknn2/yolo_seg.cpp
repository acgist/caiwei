#include "caiwei/runtime/rknn2.hpp"

#include "caiwei/type.hpp"
#include "caiwei/image_tool.hpp"

caiwei::context::SegRKNN2Context::SegRKNN2Context(std::string path, int c, int h, int w, int class_size, float iou_threshold, float confidence_threshold, caiwei::runtime::Runtime* runtime)
 : SegContext(c, h, w, class_size, iou_threshold, confidence_threshold, runtime),
   RKNN2Context(std::move(path), c, h, w) {
}

caiwei::context::SegRKNN2Context::~SegRKNN2Context() {
}

std::vector<caiwei::image::Seg> caiwei::context::SegRKNN2Context::run(const caiwei::media::ImageFrame& image) {
    std::lock_guard<std::mutex> lock(this->mutex);
    auto output{ this->run(this->h, this->w, image) };
    const auto& output_attr = this->output_attrs[0];
    const auto& output_attr1 = this->output_attrs[1];
    if (output_attr.type != RKNN_TENSOR_FLOAT16) {
        CW_LOG_E("不支持的输出类型: %s", get_type_string(output_attr.type));
        rknn_outputs_release(this->context, output.size(), output.data());
        return {};
    }
    const int64_t result_length = output_attr.dims[1];
    const int64_t stride_length = output_attr.dims[2];
    // TODO
    std::vector<float> output_data(result_length * stride_length);
    std::vector<float> output_data1(result_length * stride_length);
    auto* output_buf = reinterpret_cast<uint16_t*>(output[0].buf);
    auto* output_buf1 = reinterpret_cast<uint16_t*>(output[1].buf);
    caiwei::type::fp16_to_f32(output_data.data(), output_buf, result_length * stride_length);
    caiwei::type::fp16_to_f32(output_data1.data(), output_buf1, result_length * stride_length);
    rknn_outputs_release(this->context, output.size(), output.data());
    std::vector<float> out_dst;
    out_dst.resize(result_length * stride_length);
    caiwei::image::transpose(output_data.data(), out_dst.data(), result_length, stride_length);
    float* data = out_dst.data();
    std::vector<caiwei::image::Box>  ret_box;
    std::vector<caiwei::image::Mask> ret_mask;
    const int proto_c = output_attr1.dims[1];
    const int proto_h = output_attr1.dims[2];
    const int proto_w = output_attr1.dims[3];
    const int proto_pad_h = this->pad_h / (this->h / proto_h);
    const int proto_pad_w = this->pad_w / (this->w / proto_w);
    const int mask_h = proto_h - 2 * proto_pad_h;
    const int mask_w = proto_w - 2 * proto_pad_w;
    for (int index = 0; index < stride_length; ++index) {
        int   max_class; // 最大类别
        float max_score; // 最大分数
        float* scores = data   + 4;
        float* coeffs = scores + this->class_size;
        caiwei::image::max_loc(scores, this->class_size, max_score, max_class);
        if(max_score > this->confidence_threshold) {
            float box_x = (data[0] - this->pad_w) / (float) this->dst_w;
            float box_y = (data[1] - this->pad_h) / (float) this->dst_h;
            float box_w = (data[2]              ) / (float) this->dst_w;
            float box_h = (data[3]              ) / (float) this->dst_h;
            ret_box.push_back(
                caiwei::image::Box(
                    std::clamp(box_x - box_w / 2.0F, 0.0F, 1.0F),
                    std::clamp(box_y - box_h / 2.0F, 0.0F, 1.0F),
                    std::clamp(box_x + box_w / 2.0F, 0.0F, 1.0F),
                    std::clamp(box_y + box_h / 2.0F, 0.0F, 1.0F),
                    max_class,
                    max_score
                )
            );
            const int box_xpx = std::clamp(int((data[0] - data[2] / 2.0F) / (this->w / proto_w)), proto_pad_w, proto_w - proto_pad_w);
            const int box_ypx = std::clamp(int((data[1] - data[3] / 2.0F) / (this->h / proto_h)), proto_pad_h, proto_h - proto_pad_h);
            const int box_wpx = std::clamp(int((data[2]                 ) / (this->w / proto_w)), 0, mask_w);
            const int box_hpx = std::clamp(int((data[3]                 ) / (this->h / proto_h)), 0, mask_h);
            std::vector<float> mask;
            mask.resize(box_wpx * box_hpx);
            caiwei::image::coeff_to_mask(coeffs, box_xpx, box_ypx, box_wpx, box_hpx, output_data1.data(), proto_c, proto_h, proto_w, proto_pad_h, proto_pad_w, mask.data());
            ret_mask.push_back(
                caiwei::image::Mask(
                    box_hpx,
                    box_wpx,
                    std::move(mask)
                )
            );
        }
        data += result_length;
    }
    auto index = caiwei::image::nms_boxes(ret_box, this->iou_threshold);
    std::vector<caiwei::image::Seg> result;
    result.resize(index.size());
    int j = 0;
    for (size_t i : index) {
        result[j].box  = ret_box[i];
        result[j].mask = std::move(ret_mask[i]);
        ++j;
    }
    return result;
}
