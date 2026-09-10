#include "caiwei/runtime/rknn2.hpp"

#include "rga/im2d.hpp"

#include "caiwei/type.hpp"
#include "caiwei/image_tool.hpp"

#include <fstream>
#include <filesystem>

caiwei::context::RKNN2Context::RKNN2Context(std::string path, int c, int h, int w) : path(std::move(path)), input_data_length(c * h * w) {
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
            "RKNN2输出参数: %d - %s - %d - %d - %d - %d - %.6f - %s - %s - %s",
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
        this->pad.resize   (          w *           h * image.channels, caiwei::image::DEFAULT_PADDING);
        this->hwc.resize   (          w *           h * image.channels);
        this->chw.resize   (          w *           h * image.channels);
        this->chw_i8.resize(          w *           h * image.channels);
    }
    #ifdef ENABLE_CAIWEI_RUNTIME_RKNN2
    rga_buffer_t src_img = wrapbuffer_virtualaddr((void*) image.data.data(), image.width, image.height, RK_FORMAT_RGB_888);
    rga_buffer_t dst_img = wrapbuffer_virtualaddr(this->pad.data(), w, h, RK_FORMAT_RGB_888);
    rga_buffer_t pat_img = wrapbuffer_virtualaddr(nullptr, 0, 0, 0);
    im_rect src_rect = { 0, 0, image.width, image.height };
    im_rect dst_rect = { this->pad_w, this->pad_h, this->dst_w, this->dst_h };
    im_rect pat_rect = { 0, 0, 0, 0 };
    improcess(src_img, dst_img, pat_img, src_rect, dst_rect, pat_rect, IM_SYNC);
    #else
    caiwei::image::resize(image.data.data(), this->dst.data(), image.width, image.height, this->dst_w, this->dst_h);
    caiwei::image::padding(this->dst.data(), this->pad.data(), this->dst_w, this->dst_h, this->pad_w, this->pad_h, w, h);
    #endif
    const auto& input_attr = this->input_attrs[0];
    std::vector<rknn_output> output;
    if (input_attr.fmt == RKNN_TENSOR_NCHW) {
        if (input_attr.type == RKNN_TENSOR_INT8) {
            caiwei::image::hwc_to_chw(this->pad.data(), this->chw_i8.data(), h, w, image.channels);
            return this->run(this->chw_i8.data());
        } else if (input_attr.type == RKNN_TENSOR_FLOAT16) {
            caiwei::type::i8_to_f32(this->pad.data(), w * h * image.channels, this->hwc.data(), 255.0F);
            caiwei::image::hwc_to_chw(this->hwc.data(), this->chw.data(), h, w, image.channels);
            return this->run(this->chw.data());
        } else {
            CW_LOG_E("不支持的输入类型: %s", get_type_string(input_attr.type));
            return {};
        }
    } else if (input_attr.fmt == RKNN_TENSOR_NHWC) {
        if (input_attr.type == RKNN_TENSOR_INT8) {
            return this->run(this->pad.data());
        } else if (input_attr.type == RKNN_TENSOR_FLOAT16) {
            caiwei::type::i8_to_f32(this->pad.data(), w * h * image.channels, this->hwc.data(), 255.0F);
            return this->run(this->hwc.data());
        } else {
            CW_LOG_E("不支持的输入类型: %s", get_type_string(input_attr.type));
            return {};
        }
    } else {
        CW_LOG_E("不支持的输入格式: %s", get_format_string(input_attr.fmt));
        return {};
    }
}

std::vector<rknn_output> caiwei::context::RKNN2Context::run(uint8_t* blob, int batch) {
    std::vector<rknn_input>  inputs (this->input_size);
    std::vector<rknn_output> outputs(this->output_size);
    for (int i = 0; i < this->input_size; ++i) {
        inputs[i].buf   = blob;
        inputs[i].fmt   = this->input_attrs[i].fmt;
        inputs[i].type  = this->input_attrs[i].type;
        inputs[i].size  = this->input_attrs[i].size;
        inputs[i].index = i;
        // TODO
        inputs[i].pass_through = 1;
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
    ret = rknn_run(this->context, nullptr);
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
    caiwei::type::f32_to_fp16(data.data(), blob, this->input_data_length);
    std::vector<rknn_input>  inputs (this->input_size);
    std::vector<rknn_output> outputs(this->output_size);
    for (int i = 0; i < this->input_size; ++i) {
        inputs[i].buf   = data.data();
        inputs[i].fmt   = this->input_attrs[i].fmt;
        inputs[i].type  = this->input_attrs[i].type;
        inputs[i].size  = this->input_attrs[i].size;
        inputs[i].index = i;
        // TODO
        inputs[i].pass_through = 1;
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
