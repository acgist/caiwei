#ifndef CAIWEI_RUNTIME_RKNN_HPP

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
class OBBRKNN2Context : public OBBContext, public RKNN2Context {};
class SegRKNN2Context : public SegContext, public RKNN2Context {};
class SemRKNN2Context : public SemContext, public RKNN2Context {};
class PoseRKNN2Context : public PoseContext, public RKNN2Context {};
class DepthRKNN2Context : public DepthContext, public RKNN2Context {};
class ASRRKNN2Context : public ASRContext, public RKNN2Context {};
class TTSRKNN2Context : public TTSContext, public RKNN2Context {};
class LLMRKNN2Context : public LLMContext, public RKNN2Context {};
class VLMRKNN2Context : public VLMContext, public RKNN2Context {};
class MLLMRKNN2Context : public MLLMContext, public RKNN2Context {};
class EmbeddingRKNN2Context : public EmbeddingContext, public RKNN2Context {};
class RerankingRKNN2Context : public RerankingContext, public RKNN2Context {};
class VLEmbeddingRKNN2Context : public VLEmbeddingContext, public RKNN2Context {};
class VLRerankingRKNN2Context : public VLRerankingContext, public RKNN2Context {};

std::shared_ptr<caiwei::context::ClsContext> get_cls_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::DetContext> get_det_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::OBBContext> get_obb_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::SegContext> get_seg_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::SemContext> get_sem_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::PoseContext> get_pose_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::DepthContext> get_depth_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::ASRContext> get_asr_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::TTSContext> get_tts_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::LLMContext> get_llm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::VLMContext> get_vlm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::MLLMContext> get_mllm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::EmbeddingContext> get_embedding_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::RerankingContext> get_reranking_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::VLEmbeddingContext> get_vl_embedding_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::VLRerankingContext> get_vl_reranking_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);

class RKNN3Context {

};

class ClsRKNN3Context : public ClsContext, public RKNN3Context {};
class DetRKNN3Context : public DetContext, public RKNN3Context {};
class OBBRKNN3Context : public OBBContext, public RKNN3Context {};
class SegRKNN3Context : public SegContext, public RKNN3Context {};
class SemRKNN3Context : public SemContext, public RKNN3Context {};
class PoseRKNN3Context : public PoseContext, public RKNN3Context {};
class DepthRKNN3Context : public DepthContext, public RKNN3Context {};
class ASRRKNN3Context : public ASRContext, public RKNN3Context {};
class TTSRKNN3Context : public TTSContext, public RKNN3Context {};
class LLMRKNN3Context : public LLMContext, public RKNN3Context {};
class VLMRKNN3Context : public VLMContext, public RKNN3Context {};
class MLLMRKNN3Context : public MLLMContext, public RKNN3Context {};
class EmbeddingRKNN3Context : public EmbeddingContext, public RKNN3Context {};
class RerankingRKNN3Context : public RerankingContext, public RKNN3Context {};
class VLEmbeddingRKNN3Context : public VLEmbeddingContext, public RKNN3Context {};
class VLRerankingRKNN3Context : public VLRerankingContext, public RKNN3Context {};

std::shared_ptr<caiwei::context::ClsContext> get_cls_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN3Runtime> runtime);
std::shared_ptr<caiwei::context::DetContext> get_det_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN3Runtime> runtime);
std::shared_ptr<caiwei::context::OBBContext> get_obb_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN3Runtime> runtime);
std::shared_ptr<caiwei::context::SegContext> get_seg_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN3Runtime> runtime);
std::shared_ptr<caiwei::context::SemContext> get_sem_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN3Runtime> runtime);
std::shared_ptr<caiwei::context::PoseContext> get_pose_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN3Runtime> runtime);
std::shared_ptr<caiwei::context::DepthContext> get_depth_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN3Runtime> runtime);
std::shared_ptr<caiwei::context::ASRContext> get_asr_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN3Runtime> runtime);
std::shared_ptr<caiwei::context::TTSContext> get_tts_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN3Runtime> runtime);
std::shared_ptr<caiwei::context::LLMContext> get_llm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN3Runtime> runtime);
std::shared_ptr<caiwei::context::VLMContext> get_vlm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN3Runtime> runtime);
std::shared_ptr<caiwei::context::MLLMContext> get_mllm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN3Runtime> runtime);
std::shared_ptr<caiwei::context::EmbeddingContext> get_embedding_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN3Runtime> runtime);
std::shared_ptr<caiwei::context::RerankingContext> get_reranking_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN3Runtime> runtime);
std::shared_ptr<caiwei::context::VLEmbeddingContext> get_vl_embedding_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN3Runtime> runtime);
std::shared_ptr<caiwei::context::VLRerankingContext> get_vl_reranking_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN3Runtime> runtime);

}
}

#endif //CAIWEI_RUNTIME_RKNN_HPP
