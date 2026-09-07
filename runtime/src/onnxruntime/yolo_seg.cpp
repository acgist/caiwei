#include "caiwei/runtime/onnxruntime.hpp"

#include "caiwei/image_tool.hpp"

caiwei::context::SegONNXRuntimeContext::SegONNXRuntimeContext(std::string path, int w, int h, int class_size, float iou_threshold, float confidence_threshold, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime)
 : SegContext(w, h, class_size, iou_threshold, confidence_threshold, runtime),
   ONNXRuntimeContext(path, runtime->env) {
    this->input_node_dims.push_back(1);
    this->input_node_dims.push_back(3);
    this->input_node_dims.push_back(this->h);
    this->input_node_dims.push_back(this->w);
    this->input_data_length = 3 * this->h * this->w;
}

caiwei::context::SegONNXRuntimeContext::~SegONNXRuntimeContext() {
}

std::vector<caiwei::image::Seg> caiwei::context::SegONNXRuntimeContext::run(const caiwei::media::ImageFrame& image) {
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    // TODO 全局变量 判断是否变化
    caiwei::image::resize(image.width, image.height, this->w, this->h, dst_w, dst_h, pad_w, pad_h, scale);
    std::vector<uint8_t> dst(dst_w   * dst_h   * image.channels);
    std::vector<uint8_t> pad(this->w * this->h * image.channels);
    std::vector<float>   hwc(this->w * this->h * image.channels);
    std::vector<float>   chw(this->w * this->h * image.channels);
    caiwei::image::resize(image.data.data(), dst.data(), image.width, image.height, dst_w, dst_h);
    caiwei::image::padding(dst.data(), pad.data(), dst_w, dst_h, pad_w, pad_h, this->w, this->h);
    caiwei::image::i8_to_f32(pad.data(), this->w * this->h * image.channels, hwc.data(), 255.0F);
    caiwei::image::hwc_to_chw(hwc.data(), chw.data(), this->h, this->w, image.channels);
    auto output{ this->run(chw.data()) };
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
    const int proto_pad_h = pad_h / (this->h / proto_h);
    const int proto_pad_w = pad_w / (this->w / proto_w);
    const int mask_h = proto_h - 2 * proto_pad_h;
    const int mask_w = proto_w - 2 * proto_pad_w;
    for (int index = 0; index < stride_length; ++index) {
        int   max_class; // 最大类别
        float max_score; // 最大分数
        float* scores = data   + 4;
        float* coeffs = scores + this->class_size;
        caiwei::image::max_loc(scores, this->class_size, max_score, max_class);
        if(max_score > this->confidence_threshold) {
            float ocx = (data[0] - pad_w) / (float) dst_w;
            float ocy = (data[1] - pad_h) / (float) dst_h;
            float ow  = (data[2]        ) / (float) dst_w;
            float oh  = (data[3]        ) / (float) dst_h;
            ret_box.push_back(
                caiwei::image::Box(
                    std::clamp(ocx - ow / 2.0F, 0.0F, 1.0F),
                    std::clamp(ocy - oh / 2.0F, 0.0F, 1.0F),
                    std::clamp(ocx + ow / 2.0F, 0.0F, 1.0F),
                    std::clamp(ocy + oh / 2.0F, 0.0F, 1.0F),
                    max_class,
                    max_score
                )
            );
            const int box_x = std::clamp(int((data[0] - data[2] / 2.0F) / (this->w / proto_w)), proto_pad_w, proto_w - proto_pad_w);
            const int box_y = std::clamp(int((data[1] - data[3] / 2.0F) / (this->h / proto_h)), proto_pad_h, proto_h - proto_pad_h);
            const int box_w = std::clamp(int((data[2]) / (this->w / proto_w)), 0, mask_w);
            const int box_h = std::clamp(int((data[3]) / (this->h / proto_h)), 0, mask_h);
            std::vector<float> mask;
            mask.resize(box_w * box_h);
            caiwei::image::coeff_to_mask(coeffs, box_x, box_y, box_w, box_h, proto_data, proto_c, proto_h, proto_w, proto_pad_h, proto_pad_w, mask.data());
            ret_mask.push_back(
                caiwei::image::Mask(
                    box_h,
                    box_w,
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
