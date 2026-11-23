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

#include "caiwei/media_data.hpp"
#include "caiwei/text_data.hpp"

#include <atomic>
#include <cstdint>
#include <generator>
#include <type_traits>
#include <vector>

namespace caiwei {
namespace runtime {

class Runtime;

} // namespace runtime
} // namespace caiwei

namespace caiwei {
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

inline const char* get_type_name(Type type) {
    // clang-format off
    switch (type) {
    case Type::CLS:       return "CLS";
    case Type::DET:       return "DET";
    case Type::ASR:       return "ASR";
    case Type::LLM:       return "LLM";
    case Type::VLM:       return "VLM";
    case Type::EMBEDDING: return "EMBEDDING";
    case Type::RERANKING: return "RERANKING";
    case Type::NONE:      return "NONE";
    }
    // clang-format on
    return "UNKNOWN";
}

struct ContextInfo {
    Type type = Type::NONE;
    std::string name;
    std::string path;
    std::string vendor;
    std::vector<std::string> path_list;
};

class Context {
public:
    const Type type;
    const std::string id;
    std::atomic<int32_t> ref_count;
    std::atomic<int64_t> last_run_time;
protected:
    caiwei::runtime::Runtime* runtime = nullptr;
public:
    Context(Type type, caiwei::runtime::Runtime* runtime);
    virtual ~Context();
public:
    virtual bool load() = 0;
};

class ClsContext : public Context {
protected:
    int c = 0;
    int h = 0;
    int w = 0;
    int class_size = 0;
    int top_k = 0;
    float confidence_threshold = 0.0F;
public:
    ClsContext(int c, int h, int w, int class_size, int top_k, float confidence_threshold, caiwei::runtime::Runtime* runtime);
    ~ClsContext();
public:
    static constexpr Type TYPE = Type::CLS;
    using I = caiwei::media::ImageFrame;
    using O = std::vector<caiwei::media::Cls>;
    virtual O run(const I& image) = 0;
};

class DetContext : public Context {
protected:
    int c = 0;
    int h = 0;
    int w = 0;
    int class_size = 0;
    float iou_threshold = 0.0F;
    float confidence_threshold = 0.0F;
public:
    DetContext(
        int c, int h, int w, int class_size, float iou_threshold, float confidence_threshold, caiwei::runtime::Runtime* runtime
    );
    ~DetContext();
public:
    static constexpr Type TYPE = Type::DET;
    using I = caiwei::media::ImageFrame;
    using O = std::vector<caiwei::media::Box>;
    virtual O run(const I& image) = 0;
};

class ASRContext : public Context {
public:
    ASRContext(caiwei::runtime::Runtime* runtime);
    ~ASRContext();
public:
    static constexpr Type TYPE = Type::ASR;
    using I = caiwei::text::CompletionsRequest;
    using O = std::generator<caiwei::text::CompletionResult>;
    virtual O run(const I& request) = 0;
};

class LLMContext : public Context {
public:
    LLMContext(caiwei::runtime::Runtime* runtime);
    ~LLMContext();
public:
    static constexpr Type TYPE = Type::LLM;
    using I = caiwei::text::CompletionsRequest;
    using O = std::generator<caiwei::text::CompletionResult>;
    virtual O run(const I& request) = 0;
};

class VLMContext : public Context {
public:
    VLMContext(caiwei::runtime::Runtime* runtime);
    ~VLMContext();
public:
    static constexpr Type TYPE = Type::VLM;
    using I = caiwei::text::CompletionsRequest;
    using O = std::generator<caiwei::text::CompletionResult>;
    virtual O run(const I& request) = 0;
};

class EmbeddingContext : public Context {
public:
    EmbeddingContext(caiwei::runtime::Runtime* runtime);
    ~EmbeddingContext();
public:
    static constexpr Type TYPE = Type::EMBEDDING;
    using I = caiwei::text::EmbeddingsRequest;
    using O = caiwei::text::EmbeddingResult;
    virtual O run(const I& request) = 0;
};

class RerankingContext : public Context {
public:
    RerankingContext(caiwei::runtime::Runtime* runtime);
    ~RerankingContext();
public:
    static constexpr Type TYPE = Type::RERANKING;
    using I = caiwei::text::RerankingsRequest;
    using O = caiwei::text::RerankingResult;
    virtual O run(const I& request) = 0;
};

void init();
void stop();

const ContextInfo* get_context_info(const std::string& name);
std::vector<const ContextInfo*> get_all_context_info();
bool check_model_path(const ContextInfo* info, size_t size = 0);
void normalize_l2(const float* embd, float* out, int size, bool normalize_single = false);

template<typename F>
struct RaiiGuard {
    std::decay_t<F> fun;
    explicit RaiiGuard(F&& fun) : fun(std::forward<F>(fun)) {
    }
    ~RaiiGuard() {
        this->fun();
    }
    RaiiGuard(const RaiiGuard&) = delete;
    RaiiGuard& operator=(const RaiiGuard&) = delete;
    RaiiGuard(RaiiGuard&&) = delete;
    RaiiGuard& operator=(RaiiGuard&&) = delete;
};

} // namespace context
} // namespace caiwei

#endif // CAIWEI_RUNTIME_CONTEXT_HPP
