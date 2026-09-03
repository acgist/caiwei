/**
 * 模型上下文
 * 
 * session -> context -> runtime
 */
#ifndef CAIWEI_RUNTIME_CONTEXT_HPP
#define CAIWEI_RUNTIME_CONTEXT_HPP

#include "caiwei/text_data.hpp"
#include "caiwei/image_data.hpp"
#include "caiwei/media_data.hpp"

#include <map>
#include <mutex>
#include <chrono>
#include <vector>

namespace caiwei  {
namespace runtime {

class Runtime;

}
};

namespace caiwei  {
namespace context {

enum class Type {
    CLS,
    DET,
    OBB,
    SEG,
    SEM,
    POSE,
    DEPTH,
    ASR,
    TTS,
    LLM,
    VLM,
    MLLM,
    EMBEDDING,
    RERANKING,
    VL_EMBEDDING,
    VL_RERANKING,
    NONE,
};

struct ContextInfo {
    Type type;
    std::string name;
    std::string path;
    std::string vendor;
};

extern std::vector<ContextInfo> context_info_list;

const ContextInfo* get_context_info(const std::string& name);

class Context {
private:
    std::atomic_int32_t ref_count = 0;
public:
    std::chrono::system_clock::time_point last_run_time;
protected:
    std::shared_ptr<caiwei::runtime::Runtime> runtime{ nullptr };
public:
    Context(std::shared_ptr<caiwei::runtime::Runtime> runtime);
    virtual ~Context();
public:
    uint32_t ref();
    uint32_t unref();
};

/**
 * 图像分类模型
 */
class ClsContext : public Context {
protected:
    int w;
    int h;
    int top_k;
    int class_size;
    float confidence_threshold;
public:
    ClsContext(int w, int h, int top_k, int class_size, float confidence_threshold, std::shared_ptr<caiwei::runtime::Runtime> runtime);
    ~ClsContext();
public:
    virtual std::vector<std::pair<uint32_t, float>> run(const caiwei::media::ImageFrame& image) = 0;
};

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
    DetContext(int w, int h, int class_size, float iou_threshold, float confidence_threshold, std::shared_ptr<caiwei::runtime::Runtime> runtime);
    ~DetContext();
public:
    virtual std::vector<caiwei::image::Box> run(const caiwei::media::ImageFrame& image) = 0;
};

/**
 * 有向检测模型
 */
class OBBContext : public Context {
public:
    OBBContext(std::shared_ptr<caiwei::runtime::Runtime> runtime);
    ~OBBContext();
};

/**
 * 实例分割模型
 */
class SegContext : public Context {
public:
    SegContext(std::shared_ptr<caiwei::runtime::Runtime> runtime);
    ~SegContext();
};

/**
 * 语义分割模型
 */
class SemContext : public Context {
public:
    SemContext(std::shared_ptr<caiwei::runtime::Runtime> runtime);
    ~SemContext();
};

/**
 * 姿态估计模型
 */
class PoseContext : public Context {
public:
    PoseContext(std::shared_ptr<caiwei::runtime::Runtime> runtime);
    ~PoseContext();
};

/**
 * 深度估计模型
 */
class DepthContext : public Context {
public:
    DepthContext(std::shared_ptr<caiwei::runtime::Runtime> runtime);
    ~DepthContext();
};

/**
 * 噪声抑制模型
 */
class ASNContext : public Context {};

/**
 * 活动检测模型
 */
class VADContext : public Context {};

/**
 * 自动增益模型
 */
class AGCContext : public Context {};

/**
 * 语音识别模型
 */
class ASRContext : public Context {
public:
    ASRContext(std::shared_ptr<caiwei::runtime::Runtime> runtime);
    ~ASRContext();
};

/**
 * 语音合成模型
 */
class TTSContext : public Context {
public:
    TTSContext(std::shared_ptr<caiwei::runtime::Runtime> runtime);
    ~TTSContext();
};

/**
 * 语言大模型
 * TODO 使用c++23协程generator+co_yield
 * TODO tools测试
 */
class LLMContext : public Context {
public:
    LLMContext(std::shared_ptr<caiwei::runtime::Runtime> runtime);
    ~LLMContext();
public:
    virtual std::vector<std::string> run(const caiwei::text::CompletionsRequest& request) = 0;
};

/**
 * 视觉大模型
 */
class VLMContext : public Context {
public:
    VLMContext(std::shared_ptr<caiwei::runtime::Runtime> runtime);
    ~VLMContext();
};

/**
 * 多模态大模型
 */
class MLLMContext : public Context {
public:
    MLLMContext(std::shared_ptr<caiwei::runtime::Runtime> runtime);
    ~MLLMContext();
};

/**
 * 文本嵌入模型
 */
class EmbeddingContext : public Context {
public:
    EmbeddingContext(std::shared_ptr<caiwei::runtime::Runtime> runtime);
    ~EmbeddingContext();
};

/**
 * 视觉嵌入模型
 */
class VLEmbeddingContext : public Context {
public:
    VLEmbeddingContext(std::shared_ptr<caiwei::runtime::Runtime> runtime);
    ~VLEmbeddingContext();
};

/**
 * 文本重排序模型
 */
class RerankingContext : public Context {
public:
    RerankingContext(std::shared_ptr<caiwei::runtime::Runtime> runtime);
    ~RerankingContext();
};

/**
 * 视觉重排序模型
 */
class VLRerankingContext : public Context {
public:
    VLRerankingContext(std::shared_ptr<caiwei::runtime::Runtime> runtime);
    ~VLRerankingContext();
};

void init();
void stop();

} // namespace context
} // namespace caiwei

#endif // CAIWEI_RUNTIME_CONTEXT_HPP
