/**
 * 模型上下文
 * 
 * session -> context -> runtime
 */
#ifndef CAIWEI_CONTEXT_HPP
#define CAIWEI_CONTEXT_HPP

#include "caiwei/yolo.hpp"
#include "caiwei/runtime.hpp"

#include <map>
#include <mutex>
#include <vector>

namespace caiwei {

namespace media {

class ImageFrame;

}

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

class Context {

private:
    std::atomic_int32_t ref_count = 0;
protected:
    std::shared_ptr<caiwei::runtime::Runtime> runtime{ nullptr };
public:
    Context(std::shared_ptr<caiwei::runtime::Runtime> runtime);
    virtual ~Context();
public:
    uint32_t ref();
    uint32_t unref();

};

template <typename C>
class ContextWrapper {
private:
    std::shared_ptr<C> context{ nullptr };
public:
    std::shared_ptr<C> ptr();
public:
    ContextWrapper(std::shared_ptr<C> context);
    ~ContextWrapper();
};

/**
 * 图像分类模型
 */
class ClsContext : public Context {};

/**
 * 目标检测模型
 */
class DetContext : public Context {

protected:
    int w;
    int h;
    int class_size;
    float iou_threshold;
    float confidence_threshold;
public:
    DetContext(int w, int h, int class_size, float iou_threshold, float confidence_threshold, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime);
    ~DetContext();
public:
    virtual std::vector<caiwei::context::Box> run(const caiwei::media::ImageFrame& image) = 0;

};

/**
 * 有向检测模型
 */
class OBBContext : public Context {};

/**
 * 实例分割模型
 */
class SegContext : public Context {};

/**
 * 语义分割模型
 */
class SemContext : public Context {};

/**
 * 姿态估计模型
 */
class PoseContext : public Context {};

/**
 * 深度估计模型
 */
class DepthContext : public Context {};

/**
 * 噪声抑制模型
 */
class ASNContext : public Context {};

/**
 * 语音活动检测模型
 */
class VADContext : public Context {};

/**
 * 自动增益控制模型
 */
class AGCContext : public Context {};

/**
 * 自动语音识别模型
 * 
 * 执行流程：ANS -> VAD -> AGC -> ASR
 * 
 * 完整语音处理流程
 * 
 * - AEC
 * - ANS
 * - VAD
 * - AGC
 * - ASR
 * - Diar
 * - SID
 * 
 * |英文缩写|英文全称|中文术语|
 * |:--|:--|:--|
 * |AEC |Acoustic Echo Cancellation  |回声消除|
 * |ANS |Acoustic Noise Suppression  |噪声抑制、降噪|
 * |VAD |Voice Activity Detection    |语音活动检测、语音端点检测、静音检测|
 * |AGC |Automatic Gain Control      |自动增益控制、音量自动调节|
 * |ASR |Automatic Speech Recognition|自动语音识别、语音转写|
 * |Diar|Speaker Diarization         |说话人分簇、说话人日志|
 * |SID |Speaker Identification      |说话人识别、声纹识别|
 */
class ASRContext : public Context {};

/**
 * 语音合成模型
 */
class TTSContext : public Context {};

/**
 * 大语言模型
 */
class LLMContext : public Context {};

/**
 * 视觉语言模型
 */
class VLMContext : public Context {};

/**
 * 多模态大模型
 */
class MLLMContext : public Context {};

/**
 * 嵌入模型
 */
class EmbeddingContext : public Context {};

/**
 * 文本嵌入模型
 */
class TextEmbeddingContext : public EmbeddingContext {};

/**
 * 音频嵌入模型
 */
class AudioEmbeddingContext : public EmbeddingContext {};

/**
 * 图片嵌入模型
 */
class ImageEmbeddingContext : public EmbeddingContext {};

/**
 * 视频嵌入模型
 */
class VideoEmbeddingContext : public EmbeddingContext {};

/**
 * 重排序模型
 */
class RerankingContext : public Context {};

/**
 * 文本重排序模型
 */
class TextRerankingContext : public RerankingContext {};

/**
 * 音频重排序模型
 */
class AudioRerankingContext : public RerankingContext {};

/**
 * 图片重排序模型
 */
class ImageRerankingContext : public RerankingContext {};

/**
 * 视频重排序模型
 */
class VideoRerankingContext : public RerankingContext {};

extern void init();
extern void stop();

extern std::mutex context_mutex;
extern std::mutex runtime_mutex;
extern std::map<caiwei::context::Type, std::shared_ptr<caiwei::context::Context>> context_map;
extern std::map<caiwei::runtime::Type, std::shared_ptr<caiwei::runtime::Runtime>> runtime_map;

template <typename C>
caiwei::context::ContextWrapper<C>::ContextWrapper(std::shared_ptr<C> context) : context(std::move(context)) {
    this->context->ref();
}

template <typename C>
caiwei::context::ContextWrapper<C>::~ContextWrapper() {
    this->context->unref();
}

template <typename C>
std::shared_ptr<C> caiwei::context::ContextWrapper<C>::ptr() {
    return this->context;
}

template <typename R>
inline std::shared_ptr<R> get_context_runtime(caiwei::runtime::Type type) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    auto iter = runtime_map.find(type);
    if (iter != runtime_map.end()) {
        return std::dynamic_pointer_cast<R>(iter->second);
    }
    auto runtime = caiwei::runtime::get_runtime<R>(type);
    if (runtime) {
        runtime_map[type] = runtime;
        return runtime;
    }
    return nullptr;
}

template <typename C, typename R>
extern std::shared_ptr<C> get_context(caiwei::context::Type type, std::shared_ptr<R> runtime);

template <typename C>
std::unique_ptr<ContextWrapper<C>> get_context(caiwei::context::Type type, caiwei::runtime::Type runtime_type = caiwei::runtime::Type::NONE) {
    std::lock_guard<std::mutex> lock(context_mutex);
    auto iter = context_map.find(type);
    if (iter != context_map.end()) {
        return std::make_unique<ContextWrapper<C>>(std::dynamic_pointer_cast<C>(iter->second));
    }
    std::shared_ptr<C> context = nullptr;
    #ifdef ENABLE_CAIWEI_RUNTIME_MNN
    if (runtime_type == caiwei::runtime::Type::NONE || runtime_type == caiwei::runtime::Type::MNN) {
        auto runtime = get_context_runtime<caiwei::runtime::MNNRuntime>(caiwei::runtime::Type::MNN);
        if (runtime) {
            context = caiwei::context::get_context<C, caiwei::runtime::MNNRuntime>(type, runtime);
            if (context != nullptr) {
                context_map[type] = context;
                return std::make_unique<ContextWrapper<C>>(context);
            }
        }
    }
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_CANN
    if (runtime_type == caiwei::runtime::Type::NONE || runtime_type == caiwei::runtime::Type::CANN) {
        auto runtime = get_context_runtime<caiwei::runtime::CANNRuntime>(caiwei::runtime::Type::CANN);
        if (runtime) {
            context = caiwei::context::get_context<C, caiwei::runtime::CANNRuntime>(type, runtime);
            if (context != nullptr) {
                context_map[type] = context;
                return std::make_unique<ContextWrapper<C>>(context);
            }
        }
    }
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_RKNN
    if (runtime_type == caiwei::runtime::Type::NONE || runtime_type == caiwei::runtime::Type::RKNN2 || runtime_type == caiwei::runtime::Type::RKNN3) {
        if (type == caiwei::context::Type::YOLO) {
            auto runtime = get_context_runtime<caiwei::runtime::RKNN2Runtime>(caiwei::runtime::Type::RKNN2);
            if (runtime) {
                context = caiwei::context::get_context<C, caiwei::runtime::RKNN2Runtime>(type, runtime);
                if (context != nullptr) {
                    context_map[type] = context;
                    return std::make_unique<ContextWrapper<C>>(context);
                }
            }
        } else {
            auto runtime = get_context_runtime<caiwei::runtime::RKNN3Runtime>(caiwei::runtime::Type::RKNN3);
            if (runtime) {
                context = caiwei::context::get_context<C, caiwei::runtime::RKNN3Runtime>(type, runtime);
                if (context != nullptr) {
                    context_map[type] = context;
                    return std::make_unique<ContextWrapper<C>>(context);
                }
            }
        }
    }
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_LLAMACPP
    if (runtime_type == caiwei::runtime::Type::NONE || runtime_type == caiwei::runtime::Type::LLAMACPP) {
        auto runtime = get_context_runtime<caiwei::runtime::LlamaCPPRuntime>(caiwei::runtime::Type::LLAMACPP);
        if (runtime) {
            context = caiwei::context::get_context<C, caiwei::runtime::LlamaCPPRuntime>(type, runtime);
            if (context != nullptr) {
                context_map[type] = context;
                return std::make_unique<ContextWrapper<C>>(context);
            }
        }
    }
    #endif
    #ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
    if (runtime_type == caiwei::runtime::Type::NONE || runtime_type == caiwei::runtime::Type::ONNXRUNTIME) {
        auto runtime = get_context_runtime<caiwei::runtime::ONNXRuntimeRuntime>(caiwei::runtime::Type::ONNXRUNTIME);
        if (runtime) {
            context = caiwei::context::get_context<C, caiwei::runtime::ONNXRuntimeRuntime>(type, runtime);
            if (context != nullptr) {
                context_map[type] = context;
                return std::make_unique<ContextWrapper<C>>(context);
            }
        }
    }
    #endif
    return nullptr;
}

} // namespace context
} // namespace caiwei

#endif // CAIWEI_CONTEXT_HPP
