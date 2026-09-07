#include "caiwei/runtime/onnxruntime.hpp"

#include "caiwei/image_tool.hpp"

caiwei::context::PoseONNXRuntimeContext::PoseONNXRuntimeContext(std::string path, int w, int h, int class_size, float iou_threshold, float confidence_threshold, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime)
 : PoseContext(w, h, class_size, iou_threshold, confidence_threshold, runtime)
 , ONNXRuntimeContext(path, runtime->env) {
    this->input_node_dims.push_back(1);
    this->input_node_dims.push_back(3);
    this->input_node_dims.push_back(this->h);
    this->input_node_dims.push_back(this->w);
    this->input_data_length = 3 * this->h * this->w;
}

caiwei::context::PoseONNXRuntimeContext::~PoseONNXRuntimeContext() {
}

std::vector<caiwei::image::Pose> caiwei::context::PoseONNXRuntimeContext::run(const caiwei::media::ImageFrame& image) {
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
    std::vector<caiwei::image::Box>       ret_box;
    std::vector<caiwei::image::PosePoint> ret_point;
    for (int index = 0; index < stride_length; ++index) {
        int   max_class; // 最大类别
        float max_score; // 最大分数
        float* scores = data   + 4;
        float* points = scores + this->class_size;
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
            ret_point.push_back(
                caiwei::image::PosePoint({
                    caiwei::image::Point{ std::clamp((points[ 0] - pad_w) / (float) dst_w, 0.0F, 1.0F), std::clamp((points[ 1] - pad_h) / (float) dst_h, 0.0F, 1.0F), points[ 2] },
                    caiwei::image::Point{ std::clamp((points[ 3] - pad_w) / (float) dst_w, 0.0F, 1.0F), std::clamp((points[ 4] - pad_h) / (float) dst_h, 0.0F, 1.0F), points[ 5] },
                    caiwei::image::Point{ std::clamp((points[ 6] - pad_w) / (float) dst_w, 0.0F, 1.0F), std::clamp((points[ 7] - pad_h) / (float) dst_h, 0.0F, 1.0F), points[ 8] },
                    caiwei::image::Point{ std::clamp((points[ 9] - pad_w) / (float) dst_w, 0.0F, 1.0F), std::clamp((points[10] - pad_h) / (float) dst_h, 0.0F, 1.0F), points[11] },
                    caiwei::image::Point{ std::clamp((points[12] - pad_w) / (float) dst_w, 0.0F, 1.0F), std::clamp((points[13] - pad_h) / (float) dst_h, 0.0F, 1.0F), points[14] },
                    caiwei::image::Point{ std::clamp((points[15] - pad_w) / (float) dst_w, 0.0F, 1.0F), std::clamp((points[16] - pad_h) / (float) dst_h, 0.0F, 1.0F), points[17] },
                    caiwei::image::Point{ std::clamp((points[18] - pad_w) / (float) dst_w, 0.0F, 1.0F), std::clamp((points[19] - pad_h) / (float) dst_h, 0.0F, 1.0F), points[20] },
                    caiwei::image::Point{ std::clamp((points[21] - pad_w) / (float) dst_w, 0.0F, 1.0F), std::clamp((points[22] - pad_h) / (float) dst_h, 0.0F, 1.0F), points[23] },
                    caiwei::image::Point{ std::clamp((points[24] - pad_w) / (float) dst_w, 0.0F, 1.0F), std::clamp((points[25] - pad_h) / (float) dst_h, 0.0F, 1.0F), points[26] },
                    caiwei::image::Point{ std::clamp((points[27] - pad_w) / (float) dst_w, 0.0F, 1.0F), std::clamp((points[28] - pad_h) / (float) dst_h, 0.0F, 1.0F), points[29] },
                    caiwei::image::Point{ std::clamp((points[30] - pad_w) / (float) dst_w, 0.0F, 1.0F), std::clamp((points[31] - pad_h) / (float) dst_h, 0.0F, 1.0F), points[32] },
                    caiwei::image::Point{ std::clamp((points[33] - pad_w) / (float) dst_w, 0.0F, 1.0F), std::clamp((points[34] - pad_h) / (float) dst_h, 0.0F, 1.0F), points[35] },
                    caiwei::image::Point{ std::clamp((points[36] - pad_w) / (float) dst_w, 0.0F, 1.0F), std::clamp((points[37] - pad_h) / (float) dst_h, 0.0F, 1.0F), points[38] },
                    caiwei::image::Point{ std::clamp((points[39] - pad_w) / (float) dst_w, 0.0F, 1.0F), std::clamp((points[40] - pad_h) / (float) dst_h, 0.0F, 1.0F), points[41] },
                    caiwei::image::Point{ std::clamp((points[42] - pad_w) / (float) dst_w, 0.0F, 1.0F), std::clamp((points[43] - pad_h) / (float) dst_h, 0.0F, 1.0F), points[44] },
                    caiwei::image::Point{ std::clamp((points[45] - pad_w) / (float) dst_w, 0.0F, 1.0F), std::clamp((points[46] - pad_h) / (float) dst_h, 0.0F, 1.0F), points[47] },
                    caiwei::image::Point{ std::clamp((points[48] - pad_w) / (float) dst_w, 0.0F, 1.0F), std::clamp((points[49] - pad_h) / (float) dst_h, 0.0F, 1.0F), points[50] },
                })
            );
        }
        data += result_length;
    }
    auto index = caiwei::image::nms_boxes(ret_box, this->iou_threshold);
    std::vector<caiwei::image::Pose> result;
    result.resize(index.size());
    int j = 0;
    for (size_t i : index) {
        result[j].box   = ret_box[i];
        result[j].point = std::move(ret_point[i]);
        ++j;
    }
    return result;
}
