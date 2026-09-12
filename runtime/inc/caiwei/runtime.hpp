/**
 * 运行环境
 * 
 * session -> context -> runtime
 */
#ifndef CAIWEI_RUNTIME_RUNTIME_HPP
#define CAIWEI_RUNTIME_RUNTIME_HPP

#include <map>
#include <mutex>
#include <atomic>
#include <memory>
#include <vector>
#include <condition_variable>

#include "caiwei/context.hpp"

#ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
namespace Ort {

struct Env;

};
#endif

namespace caiwei  {
namespace runtime {

enum class Type {
    RKNN2,
    RKNN3,
    LLAMACPP,
    ONNXRUNTIME,
    NONE,
};

class Runtime {
private:
    int min_pool = 4;
    int max_pool = 8;
    int timeout   = 30;
    int keepalive = 600;
    std::mutex mutex;
    std::condition_variable cv;
    caiwei::runtime::Type type;
    std::map<std::string, std::vector<std::shared_ptr<caiwei::context::Context>>> context_map;
public:
    const std::string id;
public:
    Runtime(int min_pool, int max_pool, int timeout, int keepalive, caiwei::runtime::Type type);
    virtual ~Runtime();
public:
    virtual std::shared_ptr<caiwei::context::Context> get_context(const caiwei::context::ContextInfo* info);
    virtual void put_context(std::shared_ptr<caiwei::context::Context> context);
    virtual size_t optimize();
protected:
    virtual std::shared_ptr<caiwei::context::ClsContext>  get_cls_context  (const caiwei::context::ContextInfo* info);
    virtual std::shared_ptr<caiwei::context::DetContext>  get_det_context  (const caiwei::context::ContextInfo* info);
    virtual std::shared_ptr<caiwei::context::SegContext>  get_seg_context  (const caiwei::context::ContextInfo* info);
    virtual std::shared_ptr<caiwei::context::PoseContext> get_pose_context (const caiwei::context::ContextInfo* info);
    virtual std::shared_ptr<caiwei::context::ASRContext>  get_asr_context  (const caiwei::context::ContextInfo* info);
    virtual std::shared_ptr<caiwei::context::LLMContext>  get_llm_context  (const caiwei::context::ContextInfo* info);
    virtual std::shared_ptr<caiwei::context::VLMContext>  get_vlm_context  (const caiwei::context::ContextInfo* info);
    virtual std::shared_ptr<caiwei::context::EmbeddingContext> get_embedding_context(const caiwei::context::ContextInfo* info);
    virtual std::shared_ptr<caiwei::context::RerankingContext> get_reranking_context(const caiwei::context::ContextInfo* info);
};

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
class RKNN2Runtime : public Runtime {
public:
    RKNN2Runtime(int min_pool, int max_pool, int timeout, int keepalive);
    ~RKNN2Runtime();
public:
    std::shared_ptr<caiwei::context::ClsContext>  get_cls_context (const caiwei::context::ContextInfo* info) override;
    std::shared_ptr<caiwei::context::DetContext>  get_det_context (const caiwei::context::ContextInfo* info) override;
    std::shared_ptr<caiwei::context::SegContext>  get_seg_context (const caiwei::context::ContextInfo* info) override;
    std::shared_ptr<caiwei::context::PoseContext> get_pose_context(const caiwei::context::ContextInfo* info) override;
};
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN3
class RKNN3Runtime : public Runtime {
public:
    RKNN3Runtime(int min_pool, int max_pool, int timeout, int keepalive);
    ~RKNN3Runtime();
public:
    std::shared_ptr<caiwei::context::LLMContext> get_llm_context(const caiwei::context::ContextInfo* info) override;
};
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_LLAMACPP
class LlamaCPPRuntime : public Runtime {
public:
    LlamaCPPRuntime(int min_pool, int max_pool, int timeout, int keepalive);
    ~LlamaCPPRuntime();
public:
    std::shared_ptr<caiwei::context::LLMContext> get_llm_context(const caiwei::context::ContextInfo* info) override;
    std::shared_ptr<caiwei::context::VLMContext> get_vlm_context(const caiwei::context::ContextInfo* info) override;
};
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
class ONNXRuntimeRuntime : public Runtime {
public:
    Ort::Env* env = nullptr;
public:
    std::shared_ptr<caiwei::context::ClsContext>  get_cls_context (const caiwei::context::ContextInfo* info) override;
    std::shared_ptr<caiwei::context::DetContext>  get_det_context (const caiwei::context::ContextInfo* info) override;
    std::shared_ptr<caiwei::context::SegContext>  get_seg_context (const caiwei::context::ContextInfo* info) override;
    std::shared_ptr<caiwei::context::PoseContext> get_pose_context(const caiwei::context::ContextInfo* info) override;
public:
    ONNXRuntimeRuntime(int min_pool, int max_pool, int timeout, int keepalive);
    ~ONNXRuntimeRuntime();
};
#endif

void init();
void stop();

extern std::mutex runtime_mutex;
extern std::map<caiwei::runtime::Type, std::shared_ptr<caiwei::runtime::Runtime>> runtime_map;

template <typename R>
std::shared_ptr<R> get_runtime(caiwei::runtime::Type type);

} // namespace runtime
} // namespace caiwei

#endif // CAIWEI_RUNTIME_RUNTIME_HPP
