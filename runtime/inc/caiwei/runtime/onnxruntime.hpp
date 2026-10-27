#ifndef CAIWEI_RUNTIME_ONNXRUNTIME_HPP
#define CAIWEI_RUNTIME_ONNXRUNTIME_HPP

#include "caiwei/context.hpp"

#include "onnxruntime_cxx_api.h"

namespace caiwei  {
namespace context {

extern OrtLoggingLevel onnxruntime_log_level;

class ONNXRuntimeContext {
protected:
    const std::string path;
    const Ort::Env * env        { nullptr };
    Ort::Session   * session    { nullptr };
    Ort::RunOptions* run_options{ nullptr };
    Ort::MemoryInfo  memory_info{ nullptr };
    std::vector<std::vector<int64_t>>    input_node_dims;
    std::vector<Ort::AllocatedStringPtr> input_node_names;
    std::vector<Ort::AllocatedStringPtr> output_node_names;
    int   dst_w;
    int   dst_h;
    int   pad_w;
    int   pad_h;
    float scale;
private:
    uint32_t src_w = 0;
    uint32_t src_h = 0;
    std::vector<uint8_t> dst;
    std::vector<uint8_t> pad;
    std::vector<float>   hwc;
    std::vector<float>   chw;
public:
    ONNXRuntimeContext(std::string path, int c, int h, int w, const Ort::Env* env);
    virtual ~ONNXRuntimeContext();
public:
    bool load_model();
    std::vector<Ort::Value> run(const int h, const int w, const caiwei::media::ImageFrame& image);
    std::vector<Ort::Value> run(float* blob, const size_t size, const int batch = 1);
};

class ClsONNXRuntimeContext : public ClsContext, public ONNXRuntimeContext {
using ONNXRuntimeContext::run;
public:
    ClsONNXRuntimeContext(
        std::string path,
        int c,
        int h,
        int w,
        int class_size,
        int top_k,
        float confidence_threshold,
        Ort::Env* env,
        caiwei::runtime::Runtime* runtime
    );
    ~ClsONNXRuntimeContext();
public:
    bool load() override;
    std::vector<caiwei::image::Cls> run(const caiwei::media::ImageFrame& image) override;
};

class DetONNXRuntimeContext : public DetContext, public ONNXRuntimeContext {
using ONNXRuntimeContext::run;
private:
    std::vector<float> out_dst;
public:
    DetONNXRuntimeContext(
        std::string path,
        int c,
        int h,
        int w,
        int class_size,
        float iou_threshold,
        float confidence_threshold,
        Ort::Env* env,
        caiwei::runtime::Runtime* runtime
    );
    ~DetONNXRuntimeContext();
public:
    bool load() override;
    std::vector<caiwei::image::Box> run(const caiwei::media::ImageFrame& image) override;
};

} // context
} // caiwei

#endif // CAIWEI_RUNTIME_ONNXRUNTIME_HPP
