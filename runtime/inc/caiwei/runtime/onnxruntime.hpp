#ifndef CAIWEI_RUNTIME_ONNXRUNTIME_HPP

#include <mutex>
#include <vector>
#include <string>
#include <thread>
#include <filesystem>

#include "caiwei/log.hpp"
#include "caiwei/env.hpp"
#include "caiwei/context.hpp"
#include "caiwei/runtime.hpp"

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

class ClsONNXRuntimeContext : public ClsContext, public ONNXRuntimeContext {

public:
    ClsONNXRuntimeContext(std::string path, int w, int h, int top_k, int class_size, float confidence_threshold, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
    ~ClsONNXRuntimeContext();
public:
    using ONNXRuntimeContext::run;
    std::vector<std::pair<uint32_t, float>> run(const caiwei::media::ImageFrame& image) override;

};

class DetONNXRuntimeContext : public DetContext, public ONNXRuntimeContext {
public:
    DetONNXRuntimeContext(std::string path, int w, int h, int class_size, float iou_threshold, float confidence_threshold, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
    ~DetONNXRuntimeContext();
public:
    using ONNXRuntimeContext::run;
    std::vector<caiwei::yolo::Box> run(const caiwei::media::ImageFrame& image) override;
};

class OBBONNXRuntimeContext : public OBBContext, public ONNXRuntimeContext {};
class SegONNXRuntimeContext : public SegContext, public ONNXRuntimeContext {};
class SemONNXRuntimeContext : public SemContext, public ONNXRuntimeContext {};
class PoseRONNXRuntimeontext : public PoseContext, public ONNXRuntimeContext {};
class DepthRKONNXRuntimentext : public DepthContext, public ONNXRuntimeContext {};
class ASRONNXRuntimeContext : public ASRContext, public ONNXRuntimeContext {};
class TTSONNXRuntimeContext : public TTSContext, public ONNXRuntimeContext {};
class LLMONNXRuntimeContext : public LLMContext, public ONNXRuntimeContext {};
class VLMONNXRuntimeContext : public VLMContext, public ONNXRuntimeContext {};
class MLLMRONNXRuntimeontext : public MLLMContext, public ONNXRuntimeContext {};
class EmbeddingRKNN3CONNXRuntimet : public EmbeddingContext, public ONNXRuntimeContext {};
class RerankingRKNN3CONNXRuntimet : public RerankingContext, public ONNXRuntimeContext {};
class VLEmbeddingRKNN3ConONNXRuntime: public VLEmbeddingContext, public ONNXRuntimeContext {};
class VLRerankingRKNN3ConONNXRuntime: public VLRerankingContext, public ONNXRuntimeContext {};

std::shared_ptr<caiwei::context::ClsContext> get_cls_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
std::shared_ptr<caiwei::context::DetContext> get_det_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
std::shared_ptr<caiwei::context::OBBContext> get_obb_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
std::shared_ptr<caiwei::context::SegContext> get_seg_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
std::shared_ptr<caiwei::context::SemContext> get_sem_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
std::shared_ptr<caiwei::context::PoseContext> get_pose_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
std::shared_ptr<caiwei::context::DepthContext> get_depth_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
std::shared_ptr<caiwei::context::ASRContext> get_asr_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
std::shared_ptr<caiwei::context::TTSContext> get_tts_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
std::shared_ptr<caiwei::context::LLMContext> get_llm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
std::shared_ptr<caiwei::context::VLMContext> get_vlm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
std::shared_ptr<caiwei::context::MLLMContext> get_mllm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
std::shared_ptr<caiwei::context::EmbeddingContext> get_embedding_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
std::shared_ptr<caiwei::context::RerankingContext> get_reranking_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
std::shared_ptr<caiwei::context::VLEmbeddingContext> get_vl_embedding_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
std::shared_ptr<caiwei::context::VLRerankingContext> get_vl_reranking_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);

}
}

#endif //CAIWEI_RUNTIME_ONNXRUNTIME_HPP