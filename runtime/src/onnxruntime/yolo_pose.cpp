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
    auto output{ this->run(this->h, this->w, image) };
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
        int   max_class;
        float max_score;
        float* scores = data   + 4;
        float* points = scores + this->class_size;
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
            ret_point.push_back(
                caiwei::image::PosePoint({
                    caiwei::image::Point{ std::clamp((points[ 0] - this->pad_w) / (float) this->dst_w, 0.0F, 1.0F), std::clamp((points[ 1] - this->pad_h) / (float) this->dst_h, 0.0F, 1.0F), points[ 2] },
                    caiwei::image::Point{ std::clamp((points[ 3] - this->pad_w) / (float) this->dst_w, 0.0F, 1.0F), std::clamp((points[ 4] - this->pad_h) / (float) this->dst_h, 0.0F, 1.0F), points[ 5] },
                    caiwei::image::Point{ std::clamp((points[ 6] - this->pad_w) / (float) this->dst_w, 0.0F, 1.0F), std::clamp((points[ 7] - this->pad_h) / (float) this->dst_h, 0.0F, 1.0F), points[ 8] },
                    caiwei::image::Point{ std::clamp((points[ 9] - this->pad_w) / (float) this->dst_w, 0.0F, 1.0F), std::clamp((points[10] - this->pad_h) / (float) this->dst_h, 0.0F, 1.0F), points[11] },
                    caiwei::image::Point{ std::clamp((points[12] - this->pad_w) / (float) this->dst_w, 0.0F, 1.0F), std::clamp((points[13] - this->pad_h) / (float) this->dst_h, 0.0F, 1.0F), points[14] },
                    caiwei::image::Point{ std::clamp((points[15] - this->pad_w) / (float) this->dst_w, 0.0F, 1.0F), std::clamp((points[16] - this->pad_h) / (float) this->dst_h, 0.0F, 1.0F), points[17] },
                    caiwei::image::Point{ std::clamp((points[18] - this->pad_w) / (float) this->dst_w, 0.0F, 1.0F), std::clamp((points[19] - this->pad_h) / (float) this->dst_h, 0.0F, 1.0F), points[20] },
                    caiwei::image::Point{ std::clamp((points[21] - this->pad_w) / (float) this->dst_w, 0.0F, 1.0F), std::clamp((points[22] - this->pad_h) / (float) this->dst_h, 0.0F, 1.0F), points[23] },
                    caiwei::image::Point{ std::clamp((points[24] - this->pad_w) / (float) this->dst_w, 0.0F, 1.0F), std::clamp((points[25] - this->pad_h) / (float) this->dst_h, 0.0F, 1.0F), points[26] },
                    caiwei::image::Point{ std::clamp((points[27] - this->pad_w) / (float) this->dst_w, 0.0F, 1.0F), std::clamp((points[28] - this->pad_h) / (float) this->dst_h, 0.0F, 1.0F), points[29] },
                    caiwei::image::Point{ std::clamp((points[30] - this->pad_w) / (float) this->dst_w, 0.0F, 1.0F), std::clamp((points[31] - this->pad_h) / (float) this->dst_h, 0.0F, 1.0F), points[32] },
                    caiwei::image::Point{ std::clamp((points[33] - this->pad_w) / (float) this->dst_w, 0.0F, 1.0F), std::clamp((points[34] - this->pad_h) / (float) this->dst_h, 0.0F, 1.0F), points[35] },
                    caiwei::image::Point{ std::clamp((points[36] - this->pad_w) / (float) this->dst_w, 0.0F, 1.0F), std::clamp((points[37] - this->pad_h) / (float) this->dst_h, 0.0F, 1.0F), points[38] },
                    caiwei::image::Point{ std::clamp((points[39] - this->pad_w) / (float) this->dst_w, 0.0F, 1.0F), std::clamp((points[40] - this->pad_h) / (float) this->dst_h, 0.0F, 1.0F), points[41] },
                    caiwei::image::Point{ std::clamp((points[42] - this->pad_w) / (float) this->dst_w, 0.0F, 1.0F), std::clamp((points[43] - this->pad_h) / (float) this->dst_h, 0.0F, 1.0F), points[44] },
                    caiwei::image::Point{ std::clamp((points[45] - this->pad_w) / (float) this->dst_w, 0.0F, 1.0F), std::clamp((points[46] - this->pad_h) / (float) this->dst_h, 0.0F, 1.0F), points[47] },
                    caiwei::image::Point{ std::clamp((points[48] - this->pad_w) / (float) this->dst_w, 0.0F, 1.0F), std::clamp((points[49] - this->pad_h) / (float) this->dst_h, 0.0F, 1.0F), points[50] },
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
