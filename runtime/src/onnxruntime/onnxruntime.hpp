#include <mutex>
#include <thread>
#include <vector>
#include <string>

#include "caiwei/log.hpp"
#include "caiwei/env.hpp"
#include "caiwei/media.hpp"
#include "caiwei/context.hpp"
#include "caiwei/transform.hpp"

#include "onnxruntime_cxx_api.h"

namespace caiwei  {
namespace context {

extern OrtLoggingLevel onnxruntime_log_level;

class ONNXRuntimeContext {

protected:
    std::string path;
    std::mutex  mutex;
    Ort::Session   * session    { nullptr };
    Ort::RunOptions* run_options{ nullptr };
    size_t input_data_length;
    std::vector<int64_t>     input_node_dims;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;

public:
    ONNXRuntimeContext(std::string path, const Ort::Env* env);
    ~ONNXRuntimeContext();
protected:
    virtual Ort::Value run(float* blob, std::vector<int64_t>& out_dims, int batch = 1);

};

class ClsONNXRuntimeContext : public caiwei::context::ClsContext, public ONNXRuntimeContext {

public:
    ClsONNXRuntimeContext(std::string path, int w, int h, int top_k, int class_size, float confidence_threshold, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
    ~ClsONNXRuntimeContext();
public:
    using ONNXRuntimeContext::run;
    std::vector<std::pair<uint32_t, float>> run(const caiwei::media::ImageFrame& image) override;

};

class DetONNXRuntimeContext : public caiwei::context::DetContext, public ONNXRuntimeContext {

public:
    DetONNXRuntimeContext(std::string path, int w, int h, int class_size, float iou_threshold, float confidence_threshold, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
    ~DetONNXRuntimeContext();
public:
    using ONNXRuntimeContext::run;
    std::vector<caiwei::yolo::Box> run(const caiwei::media::ImageFrame& image) override;

};

}
}