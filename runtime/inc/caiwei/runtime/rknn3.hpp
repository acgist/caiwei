#ifndef CAIWEI_RUNTIME_RKNN3_HPP
#define CAIWEI_RUNTIME_RKNN3_HPP

#include "caiwei/log.hpp"
#include "caiwei/env.hpp"
#include "caiwei/context.hpp"

#include "rknn3/rknn3_api.h"

namespace caiwei  {
namespace context {

class RKNN3Context {
protected:
public:
};

class ClsRKNN3Context  : public ClsContext,  public RKNN3Context {};
class DetRKNN3Context  : public DetContext,  public RKNN3Context {};
class SegRKNN3Context  : public SegContext,  public RKNN3Context {};
class PoseRKNN3Context : public PoseContext, public RKNN3Context {};
class ASRRKNN3Context  : public ASRContext,  public RKNN3Context {};
class LLMRKNN3Context  : public LLMContext,  public RKNN3Context {};
class VLMRKNN3Context  : public VLMContext,  public RKNN3Context {};
class EmbeddingRKNN3Context : public EmbeddingContext, public RKNN3Context {};
class RerankingRKNN3Context : public RerankingContext, public RKNN3Context {};

}
}

#endif // CAIWEI_RUNTIME_RKNN3_HPP
