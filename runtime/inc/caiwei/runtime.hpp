/**
 * 运行环境
 * 
 * session -> context -> runtime
 */
#ifndef CAIWEI_RUNTIME_HPP
#define CAIWEI_RUNTIME_HPP

#include <atomic>
#include <memory>

namespace caiwei  {

namespace context {

enum class Type {

    ASR,
    TTS,
    LLM,
    VLM,
    MLLM,
    YOLO,
    EMBEDDING,
    RERANKING,

};

} // namespace context

namespace runtime {

enum class Type {

    MNN,
    NCNN,
    CANN,
    RKNN,
    LLAMACPP,
    ONNXRUNTIME,

};

class Runtime {

private:
    std::atomic_int32_t ref_count = 0;
public:
    uint32_t ref();
    uint32_t unref();

};

std::shared_ptr<Runtime> get_runtime(caiwei::context::Type type);

#ifdef ENABLE_CAIWEI_RUNTIME_MNN
std::shared_ptr<Runtime> get_mnn_runtime(caiwei::context::Type type);
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_NCNN
std::shared_ptr<Runtime> get_ncnn_runtime(caiwei::context::Type type);
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_CANN
std::shared_ptr<Runtime> get_cann_runtime(caiwei::context::Type type);
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN
std::shared_ptr<Runtime> get_rknn_runtime(caiwei::context::Type type);
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_LLAMACPP
std::shared_ptr<Runtime> get_llamacpp_runtime(caiwei::context::Type type);
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
std::shared_ptr<Runtime> get_onnxruntime_runtime(caiwei::context::Type type);
#endif

} // namespace runtime
} // namespace caiwei

#endif // CAIWEI_RUNTIME_HPP
