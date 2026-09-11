#ifndef CAIWEI_RUNTIME_ONNXRUNTIME_HPP
#define CAIWEI_RUNTIME_ONNXRUNTIME_HPP

#include <thread>
#include <filesystem>

#include "caiwei/log.hpp"
#include "caiwei/env.hpp"
#include "caiwei/context.hpp"

#include "onnxruntime_cxx_api.h"

namespace caiwei  {
namespace context {

extern OrtLoggingLevel onnxruntime_log_level;

class ONNXRuntimeContext {
protected:
    std::string path;
    const Ort::Env * env;
    Ort::Session   * session    { nullptr };
    Ort::RunOptions* run_options{ nullptr };
    size_t input_data_length;
    std::vector<int64_t>     input_node_dims;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    int dst_w; // 缩放目标宽度
    int dst_h; // 缩放目标高度
    int pad_w; // 缩放填充宽度
    int pad_h; // 缩放填充高度
    float scale; // 缩放比例: 输入图片 / 原始图片
private:
    uint32_t image_width;
    uint32_t image_height;
    std::vector<uint8_t> dst;
    std::vector<uint8_t> pad;
    std::vector<float>   hwc;
    std::vector<float>   chw;
public:
    ONNXRuntimeContext(std::string path, int c, int h, int w, const Ort::Env* env);
    virtual ~ONNXRuntimeContext();
public:
    bool load_model();
    std::vector<Ort::Value> run(int h, int w, const caiwei::media::ImageFrame& image);
    virtual std::vector<Ort::Value> run(float* blob, int batch = 1);
};

class ClsONNXRuntimeContext : public ClsContext, public ONNXRuntimeContext {
public:
    ClsONNXRuntimeContext(std::string path, int c, int h, int w, int top_k, int class_size, float confidence_threshold, Ort::Env* env, caiwei::runtime::Runtime* runtime);
    ~ClsONNXRuntimeContext();
public:
    using ONNXRuntimeContext::run;
    bool load() override;
    std::vector<std::pair<uint32_t, float>> run(const caiwei::media::ImageFrame& image) override;
};

class DetONNXRuntimeContext : public DetContext, public ONNXRuntimeContext {
public:
    DetONNXRuntimeContext(std::string path, int c, int h, int w, int class_size, float iou_threshold, float confidence_threshold, Ort::Env* env, caiwei::runtime::Runtime* runtime);
    ~DetONNXRuntimeContext();
public:
    using ONNXRuntimeContext::run;
    bool load() override;
    std::vector<caiwei::image::Box> run(const caiwei::media::ImageFrame& image) override;
};

class SegONNXRuntimeContext : public SegContext, public ONNXRuntimeContext {
public:
    SegONNXRuntimeContext(std::string path, int c, int h, int w, int class_size, float iou_threshold, float confidence_threshold, Ort::Env* env, caiwei::runtime::Runtime* runtime);
    ~SegONNXRuntimeContext();
public:
    using ONNXRuntimeContext::run;
    bool load() override;
    std::vector<caiwei::image::Seg> run(const caiwei::media::ImageFrame& image) override;
};

class PoseONNXRuntimeContext : public PoseContext, public ONNXRuntimeContext {
public:
    PoseONNXRuntimeContext(std::string path, int c, int h, int w, int class_size, float iou_threshold, float confidence_threshold, Ort::Env* env, caiwei::runtime::Runtime* runtime);
    ~PoseONNXRuntimeContext();
public:
    using ONNXRuntimeContext::run;
    bool load() override;
    std::vector<caiwei::image::Pose> run(const caiwei::media::ImageFrame& image) override;
};

class ASRONNXRuntimeContext : public ASRContext, public ONNXRuntimeContext {};
class LLMONNXRuntimeContext : public LLMContext, public ONNXRuntimeContext {};
class VLMONNXRuntimeContext : public VLMContext, public ONNXRuntimeContext {};
class EmbeddingRKNN3CONNXRuntimet : public EmbeddingContext, public ONNXRuntimeContext {};
class RerankingRKNN3CONNXRuntimet : public RerankingContext, public ONNXRuntimeContext {};

}
}

#endif //CAIWEI_RUNTIME_ONNXRUNTIME_HPP