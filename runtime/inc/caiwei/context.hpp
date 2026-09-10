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
#include <atomic>
#include <chrono>
#include <vector>
#include <generator>

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
    SEG,
    POSE,
    ASR,
    LLM,
    VLM,
    EMBEDDING,
    RERANKING,
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
public:
    bool share = false; // 是否可以共享
    bool usage = false; // 是否已被使用
private:
    std::atomic_int32_t ref_count = 0;
public:
    std::chrono::system_clock::time_point last_run_time;
protected:
    caiwei::runtime::Runtime* runtime = nullptr;
public:
    Context(caiwei::runtime::Runtime* runtime);
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
    int c;
    int h;
    int w;
    int top_k;
    int class_size;
    float confidence_threshold;
public:
    ClsContext(int c, int h, int w, int top_k, int class_size, float confidence_threshold, caiwei::runtime::Runtime* runtime);
    ~ClsContext();
public:
    virtual std::vector<std::pair<uint32_t, float>> run(const caiwei::media::ImageFrame& image) = 0;
};

/**
 * 目标检测模型
 */
class DetContext : public Context {
protected:
    int c;
    int h;
    int w;
    int class_size;
    float iou_threshold;
    float confidence_threshold;
public:
    DetContext(int c, int h, int w, int class_size, float iou_threshold, float confidence_threshold, caiwei::runtime::Runtime* runtime);
    ~DetContext();
public:
    virtual std::vector<caiwei::image::Box> run(const caiwei::media::ImageFrame& image) = 0;
};

/**
 * 实例分割模型
 */
class SegContext : public Context {
protected:
    int c;
    int h;
    int w;
    int class_size;
    float iou_threshold;
    float confidence_threshold;
public:
    SegContext(int c, int h, int w, int class_size, float iou_threshold, float confidence_threshold, caiwei::runtime::Runtime* runtime);
    ~SegContext();
public:
    virtual std::vector<caiwei::image::Seg> run(const caiwei::media::ImageFrame& image) = 0;
};

/**
 * 姿态估计模型
 */
class PoseContext : public Context {
protected:
    int c;
    int h;
    int w;
    int class_size;
    float iou_threshold;
    float confidence_threshold;
public:
    PoseContext(int c, int h, int w, int class_size, float iou_threshold, float confidence_threshold, caiwei::runtime::Runtime* runtime);
    ~PoseContext();
public:
    virtual std::vector<caiwei::image::Pose> run(const caiwei::media::ImageFrame& image) = 0;
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
    ASRContext(caiwei::runtime::Runtime* runtime);
    ~ASRContext();
};

/**
 * 语言大模型
 */
class LLMContext : public Context {
public:
    LLMContext(caiwei::runtime::Runtime* runtime);
    ~LLMContext();
public:
    virtual std::generator<std::string> run(const caiwei::text::CompletionsRequest& request) = 0;
};

/**
 * 视觉大模型
 */
class VLMContext : public Context {
public:
    VLMContext(caiwei::runtime::Runtime* runtime);
    ~VLMContext();
};

/**
 * 文本嵌入模型
 */
class EmbeddingContext : public Context {
public:
    EmbeddingContext(caiwei::runtime::Runtime* runtime);
    ~EmbeddingContext();
};

/**
 * 文本重排序模型
 */
class RerankingContext : public Context {
public:
    RerankingContext(caiwei::runtime::Runtime* runtime);
    ~RerankingContext();
};

void init();
void stop();

} // namespace context
} // namespace caiwei

#endif // CAIWEI_RUNTIME_CONTEXT_HPP
