#include "caiwei/runtime/onnxruntime.hpp"

#include "caiwei/image_tool.hpp"

caiwei::context::SegONNXRuntimeContext::SegONNXRuntimeContext(std::string path, int c, int h, int w, int class_size, float iou_threshold, float confidence_threshold, Ort::Env* env, caiwei::runtime::Runtime* runtime)
 : SegContext(c, h, w, class_size, iou_threshold, confidence_threshold, runtime),
   ONNXRuntimeContext(path, c, h, w, env) {
}

caiwei::context::SegONNXRuntimeContext::~SegONNXRuntimeContext() {
}

std::vector<caiwei::image::Seg> caiwei::context::SegONNXRuntimeContext::run(const caiwei::media::ImageFrame& image) {
    auto output{ this->run(this->h, this->w, image) };
    float* output_data = output.front().GetTensorMutableData<float>();
    const auto& output_dims = output.front().GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    float* proto_data = output.back().GetTensorMutableData<float>();
    const auto& proto_dims = output.back().GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    const int64_t result_length = output_dims[1];
    const int64_t stride_length = output_dims[2];
    // TODO
    std::vector<float> out_dst;
    out_dst.resize(result_length * stride_length);
    caiwei::image::transpose(output_data, out_dst.data(), result_length, stride_length);
    float* data = out_dst.data();
    std::vector<caiwei::image::Box>  ret_box;
    std::vector<caiwei::image::Mask> ret_mask;
    const int proto_c = proto_dims[1];
    const int proto_h = proto_dims[2];
    const int proto_w = proto_dims[3];
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
            caiwei::image::coeff_to_mask(coeffs, box_xpx, box_ypx, box_wpx, box_hpx, proto_data, proto_c, proto_h, proto_w, proto_pad_h, proto_pad_w, mask.data());
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
