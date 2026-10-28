#include <fstream>
#include <filesystem>

#include "rga/im2d.hpp"

#include "caiwei/log.hpp"
#include "caiwei/type.hpp"
#include "caiwei/image_tool.hpp"
#include "caiwei/runtime/rknn2.hpp"

static void print_tensor_info(const char* title, const rknn_tensor_attr& attr);

caiwei::context::RKNN2Result::RKNN2Result(
    rknn_context context,
    std::vector<rknn_output> outputs
) : context(context), outputs(std::move(outputs)) {
}

caiwei::context::RKNN2Result::~RKNN2Result() {
    if (this->context != 0 && !this->outputs.empty()) {
        rknn_outputs_release(this->context, this->outputs.size(), this->outputs.data());
    }
}

caiwei::context::RKNN2Context::RKNN2Context(std::string path, int c, int h, int w) : path(std::move(path)) {
    CW_LOG_I("创建RKNN2Context: %s", this->path.c_str());
}

caiwei::context::RKNN2Context::~RKNN2Context() {
    CW_LOG_I("释放RKNN2Context: %s", this->path.c_str());
    if (this->context != 0) {
        CW_LOG_I("释放RKNN2Context context");
        rknn_destroy(this->context);
        this->context = 0;
    }
}

bool caiwei::context::RKNN2Context::load_model() {
    if (this->context != 0) {
        return true;
    }
    std::fstream stream(this->path, std::ios::binary | std::ios::in);
    if (!stream.is_open()) {
        CW_LOG_W("模型打开失败: %s", this->path.c_str());
        return false;
    }
    const auto size = std::filesystem::file_size(this->path);
    std::vector<char> data;
    data.resize(size);
    stream.read(data.data(), size);
    if (stream.gcount() != size) {
        CW_LOG_W("模型读取失败: %s", this->path.c_str());
        return false;
    }
    stream.close();
    int ret = rknn_init(&this->context, data.data(), size, RKNN_FLAG_PRIOR_HIGH, nullptr);
    if (ret != RKNN_SUCC) {
        CW_LOG_W("模型加载失败: %d = %s", ret, this->path.c_str());
        return false;
    }
    rknn_core_mask mask = RKNN_NPU_CORE_ALL;
    ret = rknn_set_core_mask(this->context, mask);
    if (ret != RKNN_SUCC) {
        CW_LOG_W("核心掩码设置失败: %d = %s", ret, this->path.c_str());
        return false;
    }
    rknn_sdk_version version;
    ret = rknn_query(this->context, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret != RKNN_SUCC) {
        CW_LOG_W("模型版本查询失败: %d = %s", ret, this->path.c_str());
        return false;
    }
    CW_LOG_I("模型加载成功: %s = %s - %s", this->path.c_str(), version.api_version, version.drv_version);
    rknn_input_output_num io_num;
    ret = rknn_query(this->context, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        CW_LOG_W("模型参数查询失败: %d - %s", ret, this->path.c_str());
        return false;
    }
    this->input_attrs.resize(io_num.n_input);
    for (uint32_t i = 0; i < io_num.n_input; ++i) {
        auto& input_attr = this->input_attrs[i];
        input_attr.index = i;
        ret = rknn_query(this->context, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            CW_LOG_W("输入参数查询失败: %u - %d - %s", i, ret, this->path.c_str());
            return false;
        }
        print_tensor_info("输入参数", input_attr);
    }
    this->output_attrs.resize(io_num.n_output);
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        auto& output_attr = this->output_attrs[i];
        output_attr.index = i;
        ret = rknn_query(this->context, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            CW_LOG_W("输出参数查询失败: %u - %d - %s", i, ret, this->path.c_str());
            return false;
        }
        print_tensor_info("输出参数", output_attr);
    }
    return true;
}

template<typename T>
caiwei::context::RKNN2Result caiwei::context::RKNN2Context::run(T* blob, const size_t size, const int batch) {
    std::vector<rknn_input>  inputs (this->input_attrs.size());
    std::vector<rknn_output> outputs(this->output_attrs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i].buf   = blob;
        inputs[i].fmt   = this->input_attrs[i].fmt;
        inputs[i].type  = this->input_attrs[i].type;
        inputs[i].size  = this->input_attrs[i].size;
        inputs[i].index = i;
        inputs[i].pass_through = 1;
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        outputs[i].index = i;
        outputs[i].want_float = 0;
    }
    int ret = rknn_inputs_set(this->context, inputs.size(), inputs.data());
    if (ret != RKNN_SUCC) {
        CW_LOG_W("输入失败: %d", ret);
        return {};
    }
    ret = rknn_run(this->context, nullptr);
    if (ret != RKNN_SUCC) {
        CW_LOG_W("计算失败: %d", ret);
        return {};
    }
    ret = rknn_outputs_get(this->context, outputs.size(), outputs.data(), nullptr);
    if (ret != RKNN_SUCC) {
        CW_LOG_W("输出失败: %d", ret);
        return {};
    }
    return { this->context, std::move(outputs) };
}

caiwei::context::RKNN2Result caiwei::context::RKNN2Context::run(const int h, const int w, const caiwei::media::ImageFrame& image) {
    if (this->src_w != image.width || this->src_h != image.height) {
        this->src_w = image.width;
        this->src_h = image.height;
        caiwei::image::resize(image.width, image.height, w, h, this->dst_w, this->dst_h, this->pad_w, this->pad_h, this->scale);
        this->dst.resize    (this->dst_w * this->dst_h * image.channels);
        this->pad.resize    (          w *           h * image.channels, caiwei::image::DEFAULT_PADDING);
        this->hwc.resize    (          w *           h * image.channels);
        this->chw.resize    (          w *           h * image.channels);
        this->hwc_i8.resize (          w *           h * image.channels);
        this->chw_i8.resize (          w *           h * image.channels);
        this->chw_u8.resize (          w *           h * image.channels);
        this->dst_f16.resize(          w *           h * image.channels);
    }
    rga_buffer_t src_img = wrapbuffer_virtualaddr((void*) image.data.data(), image.width, image.height, RK_FORMAT_RGB_888);
    rga_buffer_t dst_img = wrapbuffer_virtualaddr(this->pad.data(),          w,           h,            RK_FORMAT_RGB_888);
    rga_buffer_t pat_img = wrapbuffer_virtualaddr(nullptr,                   0,           0,            0                );
    im_rect src_rect = { 0,           0,           image.width, image.height };
    im_rect dst_rect = { this->pad_w, this->pad_h, this->dst_w, this->dst_h  };
    im_rect pat_rect = { 0,           0,           0,           0            };
    const auto status = improcess(src_img, dst_img, pat_img, src_rect, dst_rect, pat_rect, IM_SYNC);
    if (status != IM_STATUS_SUCCESS) {
        CW_LOG_W("图片处理失败: %d", status);
        return {};
    }
    const auto& input_attr = this->input_attrs[0];
    if (input_attr.fmt == RKNN_TENSOR_NCHW) {
        if (input_attr.type == RKNN_TENSOR_INT8) {
            caiwei::image::hwc_to_chw_u8_i8(this->pad.data(), this->chw_i8.data(), h, w, image.channels);
            return this->run<int8_t>(this->chw_i8.data(), this->chw_i8.size());
        } else if (input_attr.type == RKNN_TENSOR_UINT8) {
            caiwei::image::hwc_to_chw(this->pad.data(), this->chw_u8.data(), h, w, image.channels);
            return this->run<uint8_t>(this->chw_u8.data(), this->chw_u8.size());
        } else if (input_attr.type == RKNN_TENSOR_FLOAT16) {
            caiwei::image::hwc_to_chw_u8_f32(this->pad.data(), this->chw.data(), h, w, image.channels, 255.0F);
            caiwei::type::f32_to_fp16(this->dst_f16.data(), this->chw.data(), this->chw.size());
            return this->run<uint16_t>(this->dst_f16.data(), this->dst_f16.size());
        } else {
            CW_LOG_W("不支持的输入类型: %s", get_type_string(input_attr.type));
            return {};
        }
    } else if (input_attr.fmt == RKNN_TENSOR_NHWC) {
        if (input_attr.type == RKNN_TENSOR_INT8) {
            caiwei::type::u8_to_i8(this->pad.data(), w * h * image.channels, this->hwc_i8.data());
            return this->run<int8_t>(this->hwc_i8.data(), this->hwc_i8.size());
        } else if (input_attr.type == RKNN_TENSOR_UINT8) {
            return this->run<uint8_t>(this->pad.data(), this->pad.size());
        } else if (input_attr.type == RKNN_TENSOR_FLOAT16) {
            caiwei::type::u8_to_f32(this->pad.data(), w * h * image.channels, this->hwc.data(), 255.0F);
            caiwei::type::f32_to_fp16(this->dst_f16.data(), this->hwc.data(), this->hwc.size());
            return this->run<uint16_t>(this->dst_f16.data(), this->dst_f16.size());
        } else {
            CW_LOG_W("不支持的输入类型: %s", get_type_string(input_attr.type));
            return {};
        }
    } else {
        CW_LOG_W("不支持的输入格式: %s", get_format_string(input_attr.fmt));
        return {};
    }
}

static void print_tensor_info(const char* title, const rknn_tensor_attr& attr) {
    CW_LOG_I(
        "%s: %u name: %s size: %u n_dims: %u n_elems: %u zp: %d scale: %.6f "
        "fmt: %s type: %s qnt_type: %s "
        "shape: (%u, %u, %u, %u, %u, %u, %u, %u)",
        title,
        attr.index,
        attr.name,
        attr.size,
        attr.n_dims,
        attr.n_elems,
        attr.zp,
        attr.scale,
        get_format_string  (attr.fmt),
        get_type_string    (attr.type),
        get_qnt_type_string(attr.qnt_type),
        attr.dims[0],
        attr.dims[1],
        attr.dims[2],
        attr.dims[3],
        attr.dims[4],
        attr.dims[5],
        attr.dims[6],
        attr.dims[7]
    );
}

template caiwei::context::RKNN2Result caiwei::context::RKNN2Context::run<int8_t>  (int8_t  * blob, const size_t size, const int batch);
template caiwei::context::RKNN2Result caiwei::context::RKNN2Context::run<uint8_t> (uint8_t * blob, const size_t size, const int batch);
template caiwei::context::RKNN2Result caiwei::context::RKNN2Context::run<uint16_t>(uint16_t* blob, const size_t size, const int batch);