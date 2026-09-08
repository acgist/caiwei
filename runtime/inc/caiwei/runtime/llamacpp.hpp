#ifndef CAIWEI_RUNTIME_LLAMACPP_HPP
#define CAIWEI_RUNTIME_LLAMACPP_HPP

#include "caiwei/log.hpp"
#include "caiwei/env.hpp"
#include "caiwei/context.hpp"
#include "caiwei/runtime.hpp"
#include "caiwei/text_tool.hpp"

#include "mtmd.h"
#include "llama-cpp.h"

#include <generator>
#include <filesystem>

namespace caiwei  {
namespace context {

extern std::string token_to_string(const llama_vocab* vocab, llama_token token, std::string default_token = "");
extern llama_token string_to_token(const llama_vocab* vocab, const std::string& token);

class LlamaCPPContext {
protected:
    int32_t max_token_length;
    std::string path;
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    caiwei::text::ChatTemplate chat_template;
    caiwei::text::SpecialToken special_token;
protected:
    llama_context* get_context(const caiwei::text::CompletionsRequest& request);
    llama_sampler* get_sampler(const caiwei::text::CompletionsRequest& request);
    std::generator<caiwei::text::Result> generate(llama_context* context, llama_sampler* sampler, uint32_t max_tokens, const std::string& prompt);
public:
    LlamaCPPContext(std::string path, int32_t max_token_length, caiwei::text::SpecialToken special_token);
    ~LlamaCPPContext();
};

class ClsLlamaCPPContext  : public ClsContext,  public LlamaCPPContext {};
class DetLlamaCPPContext  : public DetContext,  public LlamaCPPContext {};
class SegLlamaCPPContext  : public SegContext,  public LlamaCPPContext {};
class PoseRLlamaCPPontext : public PoseContext, public LlamaCPPContext {};

class ASRLlamaCPPContext : public ASRContext, public LlamaCPPContext {

};

class LLMLlamaCPPContext : public LLMContext, public LlamaCPPContext {
public:
    LLMLlamaCPPContext(std::string path, int32_t max_token_length, caiwei::text::SpecialToken special_token, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime);
    ~LLMLlamaCPPContext();
public:
    std::generator<std::string> run(const caiwei::text::CompletionsRequest& request) override;
};

class VLMLlamaCPPContext : public VLMContext, public LlamaCPPContext {};

class EmbeddingRKNN3CLlamaCPPt : public EmbeddingContext, public LlamaCPPContext {};
class RerankingRKNN3CLlamaCPPt : public RerankingContext, public LlamaCPPContext {};

std::shared_ptr<caiwei::context::ClsContext>  get_cls_context (const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime);
std::shared_ptr<caiwei::context::DetContext>  get_det_context (const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime);
std::shared_ptr<caiwei::context::SegContext>  get_seg_context (const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime);
std::shared_ptr<caiwei::context::PoseContext> get_pose_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime);
std::shared_ptr<caiwei::context::ASRContext>  get_asr_context (const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime);
std::shared_ptr<caiwei::context::LLMContext>  get_llm_context (const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime);
std::shared_ptr<caiwei::context::VLMContext>  get_vlm_context (const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime);
std::shared_ptr<caiwei::context::EmbeddingContext> get_embedding_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime);
std::shared_ptr<caiwei::context::RerankingContext> get_reranking_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime);

} // context
} // caiwei

#endif //CAIWEI_RUNTIME_LLAMACPP_HPP
