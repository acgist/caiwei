#ifndef CAIWEI_RUNTIME_RKNN2_HPP
#define CAIWEI_RUNTIME_RKNN2_HPP

#include "caiwei/log.hpp"
#include "caiwei/env.hpp"
#include "caiwei/context.hpp"
#include "caiwei/runtime.hpp"

namespace caiwei  {
namespace context {

class RKNN2Context {
    
};

class ClsRKNN2Context : public ClsContext, public RKNN2Context {};
class DetRKNN2Context : public DetContext, public RKNN2Context {};
class SegRKNN2Context : public SegContext, public RKNN2Context {};
class PoseRKNN2Context : public PoseContext, public RKNN2Context {};
class ASRRKNN2Context : public ASRContext, public RKNN2Context {};
class LLMRKNN2Context : public LLMContext, public RKNN2Context {};
class VLMRKNN2Context : public VLMContext, public RKNN2Context {};
class EmbeddingRKNN2Context : public EmbeddingContext, public RKNN2Context {};
class RerankingRKNN2Context : public RerankingContext, public RKNN2Context {};

std::shared_ptr<caiwei::context::ClsContext> get_cls_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::DetContext> get_det_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::SegContext> get_seg_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::PoseContext> get_pose_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::ASRContext> get_asr_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::LLMContext> get_llm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::VLMContext> get_vlm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::EmbeddingContext> get_embedding_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::RerankingContext> get_reranking_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);

}
}

#endif //CAIWEI_RUNTIME_RKNN2_HPP
