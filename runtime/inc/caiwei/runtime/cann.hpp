#ifndef CAIWEI_RUNTIME_CANN_HPP
#define CAIWEI_RUNTIME_CANN_HPP

#include "caiwei/log.hpp"
#include "caiwei/env.hpp"
#include "caiwei/context.hpp"
#include "caiwei/runtime.hpp"

namespace caiwei  {
namespace context {

class CANNContext {};

class ClsCANNContext  : public ClsContext,  public CANNContext {};
class DetCANNContext  : public DetContext,  public CANNContext {};
class SegCANNContext  : public SegContext,  public CANNContext {};
class PoseCANNContext : public PoseContext, public CANNContext {};
class ASRCANNContext  : public ASRContext,  public CANNContext {};
class LLMCANNContext  : public LLMContext,  public CANNContext {};
class VLMCANNContext  : public VLMContext,  public CANNContext {};
class EmbeddingCANNContext : public EmbeddingContext, public CANNContext {};
class RerankingCANNContext : public RerankingContext, public CANNContext {};

std::shared_ptr<caiwei::context::ClsContext>  get_cls_context (const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::CANNRuntime> runtime);
std::shared_ptr<caiwei::context::DetContext>  get_det_context (const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::CANNRuntime> runtime);
std::shared_ptr<caiwei::context::SegContext>  get_seg_context (const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::CANNRuntime> runtime);
std::shared_ptr<caiwei::context::PoseContext> get_pose_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::CANNRuntime> runtime);
std::shared_ptr<caiwei::context::ASRContext>  get_asr_context (const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::CANNRuntime> runtime);
std::shared_ptr<caiwei::context::LLMContext>  get_llm_context (const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::CANNRuntime> runtime);
std::shared_ptr<caiwei::context::VLMContext>  get_vlm_context (const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::CANNRuntime> runtime);
std::shared_ptr<caiwei::context::EmbeddingContext> get_embedding_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::CANNRuntime> runtime);
std::shared_ptr<caiwei::context::RerankingContext> get_reranking_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::CANNRuntime> runtime);

}
}

#endif //CAIWEI_RUNTIME_CANN_HPP
