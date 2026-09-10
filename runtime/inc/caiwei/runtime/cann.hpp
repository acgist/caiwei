#ifndef CAIWEI_RUNTIME_CANN_HPP
#define CAIWEI_RUNTIME_CANN_HPP

#include "caiwei/log.hpp"
#include "caiwei/env.hpp"
#include "caiwei/context.hpp"

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

}
}

#endif //CAIWEI_RUNTIME_CANN_HPP
