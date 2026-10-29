/**
 * 模型上下文
 * 负责调用模型: 每个模型上下文代表一个模型实例
 * 
 * 非线程安全
 * 
 * session -> context -> runtime
 */
#ifndef CAIWEI_RUNTIME_CONTEXT_HPP
#define CAIWEI_RUNTIME_CONTEXT_HPP

#include <map>
#include <mutex>
#include <atomic>
#include <chrono>
#include <vector>
#include <generator>

#include "caiwei/text_data.hpp"
#include "caiwei/audio_data.hpp"
#include "caiwei/image_data.hpp"
#include "caiwei/media_data.hpp"

namespace caiwei  {
namespace runtime {

class Runtime;

} // runtime
} // caiwei

namespace caiwei  {
namespace context {

enum class Type {
    CLS,
    DET,
    ASR,
    LLM,
    VLM,
    EMBEDDING,
    RERANKING,
    NONE,
};

struct ContextInfo {
    Type type;
    std::string type_name;
    std::string name;
    std::string path;
    std::string vendor;
    std::vector<std::string> paths;
};

extern std::vector<ContextInfo> context_info_list;

const ContextInfo* get_context_info(const std::string& name);

class Context {
public:
    const std::string id;
    std::atomic_int32_t ref_count = 0;
    std::chrono::steady_clock::time_point last_run_time;
protected:
    caiwei::runtime::Runtime* runtime = nullptr;
public:
    Context(caiwei::runtime::Runtime* runtime);
    virtual ~Context();
public:
    virtual bool load();
};

class ClsContext : public Context {
protected:
    int c;
    int h;
    int w;
    int class_size;
    int top_k;
    float confidence_threshold;
public:
    ClsContext(int c, int h, int w, int class_size, int top_k, float confidence_threshold, caiwei::runtime::Runtime* runtime);
    ~ClsContext();
public:
    using I = caiwei::media::ImageFrame;
    using O = std::vector<caiwei::image::Cls>;
    virtual O run(const I& image) = 0;
};

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
    using I = caiwei::media::ImageFrame;
    using O = std::vector<caiwei::image::Box>;
    virtual O run(const I& image) = 0;
};

class ASRContext : public Context {
public:
    ASRContext(caiwei::runtime::Runtime* runtime);
    ~ASRContext();
public:
    using I = caiwei::text::CompletionsRequest;
    using O = std::generator<caiwei::text::CompletionResult>;
    virtual O run(const I& request) = 0;
};

class LLMContext : public Context {
public:
    LLMContext(caiwei::runtime::Runtime* runtime);
    ~LLMContext();
public:
    using I = caiwei::text::CompletionsRequest;
    using O = std::generator<caiwei::text::CompletionResult>;
    virtual O run(const I& request) = 0;
};

class VLMContext : public Context {
public:
    VLMContext(caiwei::runtime::Runtime* runtime);
    ~VLMContext();
public:
    using I = caiwei::text::CompletionsRequest;
    using O = std::generator<caiwei::text::CompletionResult>;
    virtual O run(const I& request) = 0;
};

class EmbeddingContext : public Context {
public:
    EmbeddingContext(caiwei::runtime::Runtime* runtime);
    ~EmbeddingContext();
public:
    using I = caiwei::text::EmbeddingsRequest;
    using O = caiwei::text::EmbeddingResult;
    virtual O run(const I& request) = 0;
};

class RerankingContext : public Context {
public:
    RerankingContext(caiwei::runtime::Runtime* runtime);
    ~RerankingContext();
public:
    using I = caiwei::text::RerankingsRequest;
    using O = caiwei::text::RerankingResult;
    virtual O run(const I& request) = 0;
};

void init();
void stop();

} // context
} // caiwei

#endif // CAIWEI_RUNTIME_CONTEXT_HPP
