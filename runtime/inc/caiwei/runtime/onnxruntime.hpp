#ifndef CAIWEI_RUNTIME_ONNXRUNTIME_HPP
#define CAIWEI_RUNTIME_ONNXRUNTIME_HPP

#include "caiwei/context.hpp"

#include "onnxruntime_cxx_api.h"

namespace caiwei  {
namespace context {

extern OrtLoggingLevel onnxruntime_log_level;

class ONNXRuntimeContext {
protected:
    std::string path;
    const Ort::Env * env        { nullptr };
    Ort::Session   * session    { nullptr };
    Ort::RunOptions* run_options{ nullptr };
    std::vector<std::vector<int64_t>> input_node_dims;
    std::vector<const char*>          input_node_names;
    std::vector<const char*>          output_node_names;
    int   dst_w; // 缩放目标宽度
    int   dst_h; // 缩放目标高度
    int   pad_w; // 缩放填充宽度
    int   pad_h; // 缩放填充高度
    float scale; // 缩放比例: 输入图片 / 原始图片
private:
    uint32_t image_width;  // 输入图片宽度
    uint32_t image_height; // 输入图片高度
    std::vector<uint8_t> dst; // 缩放图片
    std::vector<uint8_t> pad; // 填充图片
    std::vector<float>   hwc; // HWC填充图片
    std::vector<float>   chw; // CHW填充图片
public:
    ONNXRuntimeContext(std::string path, int c, int h, int w, const Ort::Env* env);
    virtual ~ONNXRuntimeContext();
protected:
    std::vector<Ort::Value> run(float* blob, size_t size, int batch = 1);
public:
    bool load_model();
    std::vector<Ort::Value> run(int h, int w, const caiwei::media::ImageFrame& image);
};

class ClsONNXRuntimeContext : public ClsContext, public ONNXRuntimeContext {
using ONNXRuntimeContext::run;
public:
    ClsONNXRuntimeContext(std::string path, int c, int h, int w, int top_k, int class_size, float confidence_threshold, Ort::Env* env, caiwei::runtime::Runtime* runtime);
    ~ClsONNXRuntimeContext();
public:
    bool load() override;
    std::vector<caiwei::image::Cls> run(const caiwei::media::ImageFrame& image) override;
};

class DetONNXRuntimeContext : public DetContext, public ONNXRuntimeContext {
using ONNXRuntimeContext::run;
public:
    DetONNXRuntimeContext(std::string path, int c, int h, int w, int class_size, float iou_threshold, float confidence_threshold, Ort::Env* env, caiwei::runtime::Runtime* runtime);
    ~DetONNXRuntimeContext();
public:
    bool load() override;
    std::vector<caiwei::image::Box> run(const caiwei::media::ImageFrame& image) override;
};

class SegONNXRuntimeContext : public SegContext, public ONNXRuntimeContext {
using ONNXRuntimeContext::run;
public:
    SegONNXRuntimeContext(std::string path, int c, int h, int w, int class_size, float iou_threshold, float confidence_threshold, Ort::Env* env, caiwei::runtime::Runtime* runtime);
    ~SegONNXRuntimeContext();
public:
    bool load() override;
    std::vector<caiwei::image::Seg> run(const caiwei::media::ImageFrame& image) override;
};

class PoseONNXRuntimeContext : public PoseContext, public ONNXRuntimeContext {
using ONNXRuntimeContext::run;
public:
    PoseONNXRuntimeContext(std::string path, int c, int h, int w, int class_size, float iou_threshold, float confidence_threshold, Ort::Env* env, caiwei::runtime::Runtime* runtime);
    ~PoseONNXRuntimeContext();
public:
    bool load() override;
    std::vector<caiwei::image::Pose> run(const caiwei::media::ImageFrame& image) override;
};

}
}

#endif //CAIWEI_RUNTIME_ONNXRUNTIME_HPP