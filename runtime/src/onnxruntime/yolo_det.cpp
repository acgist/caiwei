#include "onnxruntime.hpp"

#include <filesystem>

template <>
std::shared_ptr<caiwei::context::DetContext> caiwei::context::get_context(caiwei::context::Type type, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) {
    int w = caiwei::env::get_int("CAIWEI_DET_W");
    int h = caiwei::env::get_int("CAIWEI_DET_H");
    int class_size = caiwei::env::get_int("CAIWEI_DET_CLASS_SIZE");
    float iou_threshold = caiwei::env::get_float("CAIWEI_DET_IOU_THRESHOLD");
    float confidence_threshold = caiwei::env::get_float("CAIWEI_DET_CONFIDENCE_THRESHOLD");
    std::string path = caiwei::env::get_string("CAIWEI_DET_PATH");
    if (!std::filesystem::exists(path)) {
        LOG_WARN("ONNXRtuntime模型无效: %s", path.c_str());
        return nullptr;
    }
    return std::make_shared<DetONNXRuntimeContext>(path, w, h, class_size, iou_threshold, confidence_threshold, runtime);
}

caiwei::context::DetONNXRuntimeContext::DetONNXRuntimeContext(std::string path, int w, int h, int class_size, float iou_threshold, float confidence_threshold, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) : DetContext(w, h, class_size, iou_threshold, confidence_threshold, runtime), ONNXRuntimeContext(path, runtime->env) {
    this->input_node_dims.push_back(1);
    this->input_node_dims.push_back(3);
    this->input_node_dims.push_back(this->h);
    this->input_node_dims.push_back(this->w);
    this->input_data_length = 3 * this->h * this->w;
}

caiwei::context::DetONNXRuntimeContext::~DetONNXRuntimeContext() {
}

std::vector<caiwei::context::Box> caiwei::context::DetONNXRuntimeContext::run(const caiwei::media::ImageFrame& image) {
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    // TODO 全局变量 判断是否变化
    caiwei::transform::resize(image.width, image.height, this->w, this->h, dst_w, dst_h, pad_w, pad_h, scale);
    std::vector<uint8_t> dst(dst_w   * dst_h   * image.channels);
    std::vector<uint8_t> pad(this->w * this->h * image.channels);
    std::vector<float>   hwc(this->w * this->h * image.channels);
    std::vector<float>   chw(this->w * this->h * image.channels);
    caiwei::transform::resize(image.data.data(), dst.data(), image.width, image.height, dst_w, dst_h);
    caiwei::transform::padding(dst.data(), pad.data(), dst_w, dst_h, pad_w, pad_h, this->w, this->h);
    caiwei::transform::i8_to_f32(pad.data(), this->w * this->h * image.channels, hwc.data(), 255.0F);
    caiwei::transform::hwc_to_chw(hwc.data(), chw.data(), this->h, this->w, image.channels);
    std::vector<int64_t> dims;
    auto output{ this->run(chw.data(), dims) };
    const int64_t& signalResultNum = dims[1];
    const int64_t& strideNum       = dims[2];
    float* output_data = output.GetTensorMutableData<float>();
    // TODO
    std::vector<float> out_dst;
    out_dst.resize(signalResultNum * strideNum);
    caiwei::transform::transpose(output_data, out_dst.data(), signalResultNum, strideNum);
    float* data = out_dst.data();
    std::vector<caiwei::context::Box> ret;
    for (int index = 0; index < strideNum; ++index) {
        int   max_class; // 最大类别
        float max_score; // 最大分数
        float* scores_pos = data + 4;
        caiwei::transform::max_loc(scores_pos, this->class_size, max_score, max_class);
        if(max_score > this->confidence_threshold) {
            float ocx = (data[0] - pad_w) / this->w;
            float ocy = (data[1] - pad_h) / this->h;
            float ow  = (data[2]        ) / this->w;
            float oh  = (data[3]        ) / this->h;
            ret.push_back(
                caiwei::context::Box(
                    ocx - ow / 2.0F,
                    ocy - oh / 2.0F,
                    ocx + ow / 2.0F,
                    ocy + oh / 2.0F,
                    max_class,
                    max_score
                )
            );
        }
        data += signalResultNum;
    }
    return caiwei::transform::nms_boxes(ret, this->iou_threshold);
}
