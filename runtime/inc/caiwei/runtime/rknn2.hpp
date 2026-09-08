#ifndef CAIWEI_RUNTIME_RKNN2_HPP
#define CAIWEI_RUNTIME_RKNN2_HPP

#include "caiwei/log.hpp"
#include "caiwei/env.hpp"
#include "caiwei/context.hpp"
#include "caiwei/runtime.hpp"

#include "rknn2/rknn_api.h"

namespace caiwei  {
namespace context {

class RKNN2Context {
protected:
    std::string path;
    std::mutex  mutex;
    rknn_context context = 0;
    int input_size;
    int output_size;
    size_t input_data_length;
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    int dst_w; // 缩放目标宽度
    int dst_h; // 缩放目标高度
    int pad_w; // 缩放填充宽度
    int pad_h; // 缩放填充高度
    float scale; // 缩放比例: 输入图片 / 原始图片
private:
    uint32_t image_width;
    uint32_t image_height;
    std::vector<uint8_t> dst;
    std::vector<uint8_t> pad;
    std::vector<float>   hwc;
    std::vector<float>   chw;
    std::vector<uint8_t> chw_i8;
public:
    RKNN2Context(std::string path);
    virtual ~RKNN2Context();
public:
    std::vector<rknn_output> run(int h, int w, const caiwei::media::ImageFrame& image);
    std::vector<rknn_output> run(uint8_t* blob, int batch = 1);
    std::vector<rknn_output> run(float  * blob, int batch = 1);
};

class ClsRKNN2Context : public ClsContext, public RKNN2Context {
public:
    ClsRKNN2Context(std::string path, int w, int h, int top_k, int class_size, float confidence_threshold, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
    ~ClsRKNN2Context();
public:
    using RKNN2Context::run;
    std::vector<std::pair<uint32_t, float>> run(const caiwei::media::ImageFrame& image) override;
};

class DetRKNN2Context : public DetContext, public RKNN2Context {
public:
    DetRKNN2Context(std::string path, int w, int h, int top_k, int class_size, float confidence_threshold, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
    ~DetRKNN2Context();
public:
    using RKNN2Context::run;
    std::vector<caiwei::image::Box> run(const caiwei::media::ImageFrame& image) override;
};

class SegRKNN2Context : public SegContext, public RKNN2Context {
public:
    SegRKNN2Context(std::string path, int w, int h, int top_k, int class_size, float confidence_threshold, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
    ~SegRKNN2Context();
public:
    using RKNN2Context::run;
    std::vector<caiwei::image::Seg> run(const caiwei::media::ImageFrame& image) override;
};

class PoseRKNN2Context : public PoseContext, public RKNN2Context {
public:
    PoseRKNN2Context(std::string path, int w, int h, int top_k, int class_size, float confidence_threshold, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
    ~PoseRKNN2Context();
public:
    using RKNN2Context::run;
    std::vector<caiwei::image::Pose> run(const caiwei::media::ImageFrame& image) override;
};

class ASRRKNN2Context  : public ASRContext,  public RKNN2Context {};
class LLMRKNN2Context  : public LLMContext,  public RKNN2Context {};
class VLMRKNN2Context  : public VLMContext,  public RKNN2Context {};
class EmbeddingRKNN2Context : public EmbeddingContext, public RKNN2Context {};
class RerankingRKNN2Context : public RerankingContext, public RKNN2Context {};

std::shared_ptr<caiwei::context::ClsContext>  get_cls_context (const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::DetContext>  get_det_context (const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::SegContext>  get_seg_context (const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::PoseContext> get_pose_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::ASRContext>  get_asr_context (const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::LLMContext>  get_llm_context (const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::VLMContext>  get_vlm_context (const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::EmbeddingContext> get_embedding_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);
std::shared_ptr<caiwei::context::RerankingContext> get_reranking_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime);

}
}

#endif //CAIWEI_RUNTIME_RKNN2_HPP
