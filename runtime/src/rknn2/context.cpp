#include "caiwei/runtime/rknn2.hpp"

#include "caiwei/image_tool.hpp"

#include <fstream>
#include <filesystem>

#if defined(__aarch64__)
#include <arm_neon.h>

inline void f32_to_fp16(uint16_t* dst, const float* src, int count) {
    int i = 0;
    for (; i <= count - 4; i += 4) {
        float32x4_t f32v = vld1q_f32(src + i);
        float16x4_t f16v = vcvt_f16_f32(f32v);
        vst1_f16((__fp16*)(dst + i), f16v);
    }
    for (; i < count; ++i) {
        __fp16 h = static_cast<__fp16>(src[i]);
        std::memcpy(dst + i, &h, sizeof(uint16_t));
    }
}

inline void fp16_to_f32(float* dst, const uint16_t* src, int count) {
    int i = 0;
    for (; i <= count - 4; i += 4) {
        float16x4_t f16v = vld1_f16((const __fp16*)(src + i));
        float32x4_t f32v = vcvt_f32_f16(f16v);
        vst1q_f32(dst + i, f32v);
    }
    for (; i < count; ++i) {
        __fp16 h;
        std::memcpy(&h, src + i, sizeof(uint16_t));
        dst[i] = static_cast<float>(h);
    }
}
#else
inline void f32_to_fp16(uint16_t* dst, const float* src, int count) {
    // 忽略
}
inline void fp16_to_f32(float* dst, const uint16_t* src, int count) {
    // 忽略
}
#endif

inline static float qnt_to_f32(int8_t qnt, int32_t zp, float scale) {
    return ((float) qnt - (float) zp) * scale;
}

inline static int8_t qnt_to_f32(float f32, int32_t zp, float scale) {
    float  dst = (f32 / scale) + zp;
    int8_t res = (int8_t) ((int32_t) std::clamp(dst, -128.0F, 127.0F));
    return res;
}

caiwei::context::RKNN2Context::RKNN2Context(std::string path) : path(std::move(path)) {
    auto size = std::filesystem::file_size(this->path);
    std::fstream stream(this->path);
    if (!stream.is_open()) {
        CW_LOG_W("打开RKNN2模型失败: %s", this->path.c_str());
        return;
    }
    std::vector<char> data;
    data.resize(size);
    stream.read(reinterpret_cast<char *>(data.data()), size);
    stream.close();
    int ret = 0;
    ret = rknn_init(&this->context, data.data(), size, RKNN_FLAG_PRIOR_HIGH, nullptr);
    if (ret < 0) {
        CW_LOG_W("加载RKNN2模型失败: %d = %s", ret, this->path.c_str());
        return;
    }
    rknn_core_mask mask = RKNN_NPU_CORE_ALL;
    rknn_set_core_mask(this->context, mask);
    rknn_sdk_version version;
    ret = rknn_query(this->context, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        CW_LOG_W("查询RKNN2模型版本失败: %d = %s", ret, this->path.c_str());
    }
    CW_LOG_I("加载RKNN2模型版本: %s = %s - %s", this->path.c_str(), version.api_version, version.drv_version);
    rknn_input_output_num io_num;
    ret = rknn_query(this->context, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        CW_LOG_W("读取RKNN2参数失败: %d - %s", ret, this->path.c_str());
        return;
    }
    CW_LOG_I("读取RKNN2参数成功: %d - %d", io_num.n_input, io_num.n_output);
    this->input_size  = io_num.n_input;
    this->output_size = io_num.n_output;
    this->input_attrs.resize(this->input_size);
    for (uint32_t i = 0; i < this->input_size; ++i) {
        auto& input_attr = this->input_attrs[i];
        input_attr.index = i;
        ret = rknn_query(this->context, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(rknn_tensor_attr));
        if (ret < 0) {
            CW_LOG_W("读取RKNN2输入参数失败: %d - %d - %s", i, ret, this->path.c_str());
            return;
        }
        CW_LOG_I(
            "RKNN2输入参数: %d - %s - %d - %d - %d - %s - %s - %s",
            input_attr.index,
            input_attr.name,
            input_attr.size,
            input_attr.n_dims,
            input_attr.n_elems,
            get_format_string  (input_attr.fmt),
            get_type_string    (input_attr.type),
            get_qnt_type_string(input_attr.qnt_type)
        );
        CW_LOG_I(
            "RKNN2输入维度: %d - %d - %d - %d - %d - %d - %d - %d",
            input_attr.dims[0],
            input_attr.dims[1],
            input_attr.dims[2],
            input_attr.dims[3],
            input_attr.dims[4],
            input_attr.dims[5],
            input_attr.dims[6],
            input_attr.dims[7]
        );
    }
    this->output_attrs.resize(io_num.n_output);
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        auto& output_attr = this->output_attrs[i];
        output_attr.index = i;
        ret = rknn_query(this->context, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(rknn_tensor_attr));
        if (ret < 0) {
            CW_LOG_W("读取RKNN2输出参数失败: %d - %d - %s", i, ret, this->path.c_str());
            return;
        }
        CW_LOG_I(
            "RKNN2输出参数: %d - %s - %d - %d - %d -%d - %.6f - %s - %s - %s",
            output_attr.index,
            output_attr.name,
            output_attr.size,
            output_attr.n_dims,
            output_attr.n_elems,
            output_attr.zp,
            output_attr.scale,
            get_format_string  (output_attr.fmt),
            get_type_string    (output_attr.type),
            get_qnt_type_string(output_attr.qnt_type)
        );
        CW_LOG_I(
            "RKNN2输出维度: %d - %d - %d - %d - %d - %d - %d - %d",
            output_attr.dims[0],
            output_attr.dims[1],
            output_attr.dims[2],
            output_attr.dims[3],
            output_attr.dims[4],
            output_attr.dims[5],
            output_attr.dims[6],
            output_attr.dims[7]
        );
    }
}

caiwei::context::RKNN2Context::~RKNN2Context() {
    if (this->context != 0) {
        rknn_destroy(this->context);
        this->context = 0;
    }
}

std::vector<rknn_output> caiwei::context::RKNN2Context::run(int h, int w, const caiwei::media::ImageFrame& image) {
    if (this->image_width != image.width || this->image_height != image.height) {
        this->image_width  = image.width;
        this->image_height = image.height;
        caiwei::image::resize(image.width, image.height, w, h, this->dst_w, this->dst_h, this->pad_w, this->pad_h, this->scale);
        this->dst.resize   (this->dst_w * this->dst_h * image.channels);
        this->pad.resize   (          w *           h * image.channels);
        this->hwc.resize   (          w *           h * image.channels);
        this->chw.resize   (          w *           h * image.channels);
        this->chw_i8.resize(          w *           h * image.channels);
    }
    caiwei::image::resize(image.data.data(), this->dst.data(), image.width, image.height, this->dst_w, this->dst_h);
    caiwei::image::padding(this->dst.data(), this->pad.data(), this->dst_w, this->dst_h, this->pad_w, this->pad_h, w, h);
    const auto& input_attr  = this->input_attrs[0];
    const auto& output_attr = this->output_attrs[0];
    std::vector<rknn_output> output;
    if (input_attr.fmt == RKNN_TENSOR_NCHW) {
        if (input_attr.type == RKNN_TENSOR_INT8 || input_attr.type == RKNN_TENSOR_UINT8) {
            caiwei::image::hwc_to_chw(this->pad.data(), this->chw_i8.data(), h, w, image.channels);
            return this->run(this->chw_i8.data());
        } else if (input_attr.type == RKNN_TENSOR_FLOAT16) {
            caiwei::image::i8_to_f32(this->pad.data(), w * h * image.channels, this->hwc.data());
            caiwei::image::hwc_to_chw(this->hwc.data(), this->chw.data(), h, w, image.channels);
            return this->run(this->chw.data());
        } else {
            CW_LOG_E("不支持的输入类型: %d", get_type_string(input_attr.type));
            return {};
        }
    } else if (input_attr.fmt == RKNN_TENSOR_NHWC) {
        if (input_attr.type == RKNN_TENSOR_INT8 || input_attr.type == RKNN_TENSOR_UINT8) {
            return this->run(this->pad.data());
        } else if (input_attr.type == RKNN_TENSOR_FLOAT16) {
            caiwei::image::i8_to_f32(this->pad.data(), w * h * image.channels, this->hwc.data());
            return this->run(this->hwc.data());
        } else {
            CW_LOG_E("不支持的输入类型: %d", get_type_string(input_attr.type));
            return {};
        }
    } else {
        CW_LOG_E("不支持的输入格式: %d", get_format_string(input_attr.fmt));
        return {};
    }
}

std::vector<rknn_output> caiwei::context::RKNN2Context::run(uint8_t* blob, int batch) {
    std::lock_guard<std::mutex> lock(this->mutex);
    std::vector<rknn_input>  inputs (this->input_size);
    std::vector<rknn_output> outputs(this->output_size);
    for (int i = 0; i < this->input_size; ++i) {
        inputs[i].buf   = blob;
        inputs[i].fmt   = this->input_attrs[i].fmt;
        inputs[i].type  = this->input_attrs[i].type;
        inputs[i].size  = this->input_attrs[i].size;
        inputs[i].index = i;
        inputs[i].pass_through = 0;
    }
    for (int i = 0; i < this->output_size; ++i) {
        outputs[i].index = i;
        outputs[i].want_float = 0;
    }
    int ret = 0;
    ret = rknn_inputs_set(this->context, this->input_size, inputs.data());
    if (ret < 0) {
        CW_LOG_W("RKNN2设置输入失败: %d", ret);
    }
    ret = rknn_run(this->context, NULL);
    if (ret < 0) {
        CW_LOG_W("RKNN2执行运算失败: %d", ret);
    }
    ret = rknn_outputs_get(this->context, this->output_size, outputs.data(), nullptr);
    if (ret < 0) {
        CW_LOG_W("RKNN2读取输出失败: %d", ret);
    }
    return outputs;
}

std::vector<rknn_output> caiwei::context::RKNN2Context::run(float* blob, int batch) {
    std::lock_guard<std::mutex> lock(this->mutex);
    std::vector<uint16_t> data(this->input_data_length);
    f32_to_fp16(data.data(), blob, this->input_data_length);
    std::vector<rknn_input>  inputs (this->input_size);
    std::vector<rknn_output> outputs(this->output_size);
    for (int i = 0; i < this->input_size; ++i) {
        inputs[i].buf   = data.data();
        inputs[i].fmt   = this->input_attrs[i].fmt;
        inputs[i].type  = this->input_attrs[i].type;
        inputs[i].size  = this->input_attrs[i].size;
        inputs[i].index = i;
        inputs[i].pass_through = 0;
    }
    for (int i = 0; i < this->output_size; ++i) {
        outputs[i].index = i;
        outputs[i].want_float = 0;
    }
    int ret = 0;
    ret = rknn_inputs_set(this->context, this->input_size, inputs.data());
    if (ret < 0) {
        CW_LOG_W("RKNN2设置输入失败: %d", ret);
    }
    ret = rknn_run(this->context, NULL);
    if (ret < 0) {
        CW_LOG_W("RKNN2执行运算失败: %d", ret);
    }
    ret = rknn_outputs_get(this->context, this->output_size, outputs.data(), nullptr);
    if (ret < 0) {
        CW_LOG_W("RKNN2读取输出失败: %d", ret);
    }
    return outputs;
}

std::shared_ptr<caiwei::context::ClsContext> caiwei::context::get_cls_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime) {
    int w = caiwei::env::get_int("CAIWEI_CLS_W");
    int h = caiwei::env::get_int("CAIWEI_CLS_H");
    int top_k = caiwei::env::get_int("CAIWEI_CLS_TOP_K");
    int class_size = caiwei::env::get_int("CAIWEI_CLS_CLASS_SIZE");
    float confidence_threshold = caiwei::env::get_float("CAIWEI_CLS_CONFIDENCE_THRESHOLD");
    if (!std::filesystem::exists(info->path)) {
        CW_LOG_W("ClsContext模型无效: %s", info->path.c_str());
        return nullptr;
    }
    return std::make_shared<ClsRKNN2Context>(info->path, w, h, top_k, class_size, confidence_threshold, runtime);
}

std::shared_ptr<caiwei::context::DetContext> caiwei::context::get_det_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::SegContext> caiwei::context::get_seg_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::PoseContext> caiwei::context::get_pose_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::ASRContext> caiwei::context::get_asr_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::LLMContext> caiwei::context::get_llm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::VLMContext> caiwei::context::get_vlm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::EmbeddingContext> caiwei::context::get_embedding_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::RerankingContext> caiwei::context::get_reranking_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::RKNN2Runtime> runtime) {
    return nullptr;
}
