#include "caiwei/runtime/onnxruntime.hpp"

#include "caiwei/image_tool.hpp"

caiwei::context::DetONNXRuntimeContext::DetONNXRuntimeContext(std::string path, int w, int h, int class_size, float iou_threshold, float confidence_threshold, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime)
 : DetContext(w, h, class_size, iou_threshold, confidence_threshold, runtime),
   ONNXRuntimeContext(path, runtime->env) {
    this->input_node_dims.push_back(1);
    this->input_node_dims.push_back(3);
    this->input_node_dims.push_back(this->h);
    this->input_node_dims.push_back(this->w);
    this->input_data_length = 3 * this->h * this->w;
}

caiwei::context::DetONNXRuntimeContext::~DetONNXRuntimeContext() {
}

std::vector<caiwei::image::Box> caiwei::context::DetONNXRuntimeContext::run(const caiwei::media::ImageFrame& image) {
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
    const int64_t result_length = output_dims[1];
    const int64_t stride_length = output_dims[2];
    // TODO
    std::vector<float> out_dst;
    out_dst.resize(result_length * stride_length);
    caiwei::image::transpose(output_data, out_dst.data(), result_length, stride_length);
    float* data = out_dst.data();
    std::vector<caiwei::image::Box> ret;
    for (int index = 0; index < stride_length; ++index) {
        int   max_class; // 最大类别
        float max_score; // 最大分数
        float* scores = data + 4;
        caiwei::image::max_loc(scores, this->class_size, max_score, max_class);
        if(max_score > this->confidence_threshold) {
            float ocx = (data[0] - pad_w) / (float) dst_w;
            float ocy = (data[1] - pad_h) / (float) dst_h;
            float ow  = (data[2]        ) / (float) dst_w;
            float oh  = (data[3]        ) / (float) dst_h;
            ret.push_back(
                caiwei::image::Box(
                    std::clamp(ocx - ow / 2.0F, 0.0F, 1.0F),
                    std::clamp(ocy - oh / 2.0F, 0.0F, 1.0F),
                    std::clamp(ocx + ow / 2.0F, 0.0F, 1.0F),
                    std::clamp(ocy + oh / 2.0F, 0.0F, 1.0F),
                    max_class,
                    max_score
                )
            );
        }
        data += result_length;
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
