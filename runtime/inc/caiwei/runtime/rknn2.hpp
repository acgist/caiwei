#ifndef CAIWEI_RUNTIME_RKNN2_HPP
#define CAIWEI_RUNTIME_RKNN2_HPP

#include "rknn2/rknn_api.h"

#include "caiwei/context.hpp"

namespace caiwei  {
namespace context {

struct RKNN2Result {
    rknn_context context = 0;
    std::vector<rknn_output> outputs;
    
    RKNN2Result(rknn_context context, std::vector<rknn_output> outputs);
    ~RKNN2Result();
    RKNN2Result() noexcept = default;
    RKNN2Result(const RKNN2Result& result) noexcept = delete;
    RKNN2Result& operator=(const RKNN2Result& result) noexcept = delete;
    RKNN2Result(RKNN2Result&& result) noexcept = default;
    RKNN2Result& operator=(RKNN2Result&& result) noexcept = default;
};

class RKNN2Context {
protected:
    const std::string path;
    rknn_context context = 0;
    std::vector<uint16_t> data_f16;
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    int   dst_w;
    int   dst_h;
    int   pad_w;
    int   pad_h;
    float scale;
private:
    uint32_t src_w = 0;
    uint32_t src_h = 0;
    std::vector<uint8_t> dst;
    std::vector<uint8_t> pad;
    std::vector<float>   hwc;
    std::vector<float>   chw;
    std::vector<int8_t>  pad_i8;
    std::vector<int8_t>  chw_i8;
    std::vector<uint8_t> chw_u8;
public:
    RKNN2Context(std::string path, int c, int h, int w);
    virtual ~RKNN2Context();
public:
    bool load_model();
    RKNN2Result run(const int h, const int w, const caiwei::media::ImageFrame& image);
    RKNN2Result run(uint8_t* blob, const size_t size, const int batch = 1);
    RKNN2Result run(int8_t * blob, const size_t size, const int batch = 1);
    RKNN2Result run(float  * blob, const size_t size, const int batch = 1);
};

class ClsRKNN2Context : public ClsContext, public RKNN2Context {
using RKNN2Context::run;
private:
    std::vector<float> out_dst;
public:
    ClsRKNN2Context(
        std::string path,
        int c,
        int h,
        int w,
        int class_size,
        int top_k,
        float confidence_threshold,
        caiwei::runtime::Runtime* runtime
    );
    ~ClsRKNN2Context();
public:
    bool load() override;
    std::vector<caiwei::image::Cls> run(const caiwei::media::ImageFrame& image) override;
};

class DetRKNN2Context : public DetContext, public RKNN2Context {
using RKNN2Context::run;
private:
    std::vector<int8_t> out_dst_i8;
    std::vector<float>  out_buf_f32;
    std::vector<float>  out_dst_f32;
public:
    DetRKNN2Context(
        std::string path,
        int c,
        int h,
        int w,
        int class_size,
        float iou_threshold,
        float confidence_threshold,
        caiwei::runtime::Runtime* runtime
    );
    ~DetRKNN2Context();
public:
    bool load() override;
    std::vector<caiwei::image::Box> run(const caiwei::media::ImageFrame& image) override;
};

} // context
} // caiwei

#endif // CAIWEI_RUNTIME_RKNN2_HPP
