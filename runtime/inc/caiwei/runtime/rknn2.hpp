#ifndef CAIWEI_RUNTIME_RKNN2_HPP
#define CAIWEI_RUNTIME_RKNN2_HPP

#include "caiwei/context.hpp"

#include "rknn2/rknn_api.h"

namespace caiwei  {
namespace context {

class RKNN2Context {
protected:
    std::string path;
    rknn_context context = 0;
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    int dst_w;   // 缩放目标宽度
    int dst_h;   // 缩放目标高度
    int pad_w;   // 缩放填充宽度
    int pad_h;   // 缩放填充高度
    float scale; // 缩放比例: 输入图片 / 原始图片
private:
    uint32_t image_width;  // 输入图片宽度
    uint32_t image_height; // 输入图片高度
    std::vector<uint8_t> dst; // 缩放图片
    std::vector<uint8_t> pad; // 填充图片
    std::vector<float>   hwc; // HWC填充图片
    std::vector<float>   chw; // CHW填充图片
    std::vector<uint8_t> chw_i8; // CHW填充图片
public:
    RKNN2Context(std::string path, int c, int h, int w);
    virtual ~RKNN2Context();
public:
    bool load_model();
    std::vector<rknn_output> run(int h, int w, const caiwei::media::ImageFrame& image);
    std::vector<rknn_output> run(uint8_t* blob, size_t size, int batch = 1);
    std::vector<rknn_output> run(float  * blob, size_t size, int batch = 1);
};

class ClsRKNN2Context : public ClsContext, public RKNN2Context {
using RKNN2Context::run;
public:
    ClsRKNN2Context(std::string path, int c, int h, int w, int top_k, int class_size, float confidence_threshold, caiwei::runtime::Runtime* runtime);
    ~ClsRKNN2Context();
public:
    bool load() override;
    std::vector<caiwei::image::Cls> run(const caiwei::media::ImageFrame& image) override;
};

class DetRKNN2Context : public DetContext, public RKNN2Context {
using RKNN2Context::run;
public:
    DetRKNN2Context(std::string path, int c, int h, int w, int class_size, float iou_threshold, float confidence_threshold, caiwei::runtime::Runtime* runtime);
    ~DetRKNN2Context();
public:
    bool load() override;
    std::vector<caiwei::image::Box> run(const caiwei::media::ImageFrame& image) override;
};

class SegRKNN2Context : public SegContext, public RKNN2Context {
using RKNN2Context::run;
public:
    SegRKNN2Context(std::string path, int c, int h, int w, int class_size, float iou_threshold, float confidence_threshold, caiwei::runtime::Runtime* runtime);
    ~SegRKNN2Context();
public:
    bool load() override;
    std::vector<caiwei::image::Seg> run(const caiwei::media::ImageFrame& image) override;
};

class PoseRKNN2Context : public PoseContext, public RKNN2Context {
using RKNN2Context::run;
public:
    PoseRKNN2Context(std::string path, int c, int h, int w, int class_size, float iou_threshold, float confidence_threshold, caiwei::runtime::Runtime* runtime);
    ~PoseRKNN2Context();
public:
    bool load() override;
    std::vector<caiwei::image::Pose> run(const caiwei::media::ImageFrame& image) override;
};

} // context
} // caiwei

#endif //CAIWEI_RUNTIME_RKNN2_HPP
