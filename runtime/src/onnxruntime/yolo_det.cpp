#include "caiwei/image_tool.hpp"
#include "caiwei/runtime/onnxruntime.hpp"

caiwei::context::DetONNXRuntimeContext::DetONNXRuntimeContext(std::string path, int c, int h, int w, int class_size, float iou_threshold, float confidence_threshold, Ort::Env* env, caiwei::runtime::Runtime* runtime)
 : DetContext(c, h, w, class_size, iou_threshold, confidence_threshold, runtime),
   ONNXRuntimeContext(path, c, h, w, env) {
}

caiwei::context::DetONNXRuntimeContext::~DetONNXRuntimeContext() {
}

bool caiwei::context::DetONNXRuntimeContext::load() {
    return this->load_model();
}

std::vector<caiwei::image::Box> caiwei::context::DetONNXRuntimeContext::run(const caiwei::media::ImageFrame& image) {
    auto output{ this->run(this->h, this->w, image) };
    float* output_data = output.front().GetTensorMutableData<float>();
    const auto& output_dims = output.front().GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    const int64_t result_length = output_dims[1];
    const int64_t stride_length = output_dims[2];
    // TODO
    float* data = output_data;
    std::vector<caiwei::image::Box> ret;
    #ifdef ENABLE_CAIWEI_YOLO_E2E
    for (int index = 0; index < result_length; ++index) {
        int   max_class = data[5];
        float max_score = data[4];
        if(max_score > this->confidence_threshold) {
            float box_x1 = (data[0] - this->pad_w) / (float) this->dst_w;
            float box_y1 = (data[1] - this->pad_h) / (float) this->dst_h;
            float box_x2 = (data[2] - this->pad_w) / (float) this->dst_w;
            float box_y2 = (data[3] - this->pad_h) / (float) this->dst_h;
            ret.push_back(
                caiwei::image::Box(
                    std::clamp(box_x1, 0.0F, 1.0F),
                    std::clamp(box_y1, 0.0F, 1.0F),
                    std::clamp(box_x2, 0.0F, 1.0F),
                    std::clamp(box_y2, 0.0F, 1.0F),
                    max_class,
                    max_score
                )
            );
        }
        data += stride_length;
    }
    return ret;
    #else
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
    auto index = caiwei::image::nms_boxes(ret, this->iou_threshold);
    std::vector<caiwei::image::Box> result;
    result.resize(index.size());
    int j = 0;
    for (size_t i : index) {
        result[j] = ret[i];
        ++j;
    }
    return result;
    #endif
}
