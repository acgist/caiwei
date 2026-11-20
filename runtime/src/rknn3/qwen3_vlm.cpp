#include <cstddef>
#include <cstring>
#include <tuple>

#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/media_tool.hpp"
#include "caiwei/runtime/rknn3.hpp"
#include "caiwei/type.hpp"

caiwei::context::VLMRKNN3Context::VLMRKNN3Context(
    std::string model_path,
    std::string weight_path,
    std::string embedding_path,
    std::string tokenizer_path,
    std::string media_model_path,
    std::string media_weight_path,
    int32_t max_token_length,
    caiwei::text::SpecialToken special_token,
    caiwei::runtime::Runtime* runtime
) :
    VLMContext(runtime),
    RKNN3Context(
        std::move(model_path),
        std::move(weight_path),
        std::move(embedding_path),
        std::move(tokenizer_path),
        std::move(media_model_path),
        std::move(media_weight_path),
        max_token_length,
        std::move(special_token)
    ) {
}

caiwei::context::VLMRKNN3Context::~VLMRKNN3Context() {
}

bool caiwei::context::VLMRKNN3Context::load() {
    this->release();
    if (!(this->load_model(true) && this->load_media_model(true) && this->load_template())) {
        this->release();
        return false;
    }
    if (this->media_input.size() != 1 || this->media_output.size() != CAIWEI_RKNN3_DEEPSTACK_TENSORS + 1) {
        CW_LOG_W(
            "模型错误: %zu != 1 || %zu != %d",
            this->media_input.size(),
            this->media_output.size(),
            CAIWEI_RKNN3_DEEPSTACK_TENSORS + 1
        );
        this->release();
        return false;
    }
    // clang-format off
    uint32_t image_width  = 0;
    uint32_t image_height = 0;
    auto& input_tensor = this->media_input[0];
    if (input_tensor.attr->n_dims != 4) {
        CW_LOG_W("模型维度错误: %u != 4", input_tensor.attr->n_dims);
        this->release();
        return false;
    }
    if (input_tensor.attr->layout == RKNN3_TENSOR_NCHW) {
        image_width  = input_tensor.attr->shape[3];
        image_height = input_tensor.attr->shape[2];
    } else if (input_tensor.attr->layout == RKNN3_TENSOR_NHWC) {
        image_width  = input_tensor.attr->shape[2];
        image_height = input_tensor.attr->shape[1];
    } else {
        CW_LOG_W("不支持的输入布局: %s", rknn3_get_layout_string(input_tensor.attr->layout));
        this->release();
        return false;
    }
    // clang-format on
    for (const auto& output_tensor : this->media_output) {
        if (output_tensor.attr->n_dims != 2) {
            CW_LOG_W("模型维度错误: %u != 2", output_tensor.attr->n_dims);
            this->release();
            return false;
        }
    }
    const int w = caiwei::env::get_int("CAIWEI_VLM_IMAGE_WIDTH");
    const int h = caiwei::env::get_int("CAIWEI_VLM_IMAGE_HEIGHT");
    if (static_cast<uint32_t>(w) != image_width || static_cast<uint32_t>(h) != image_height) {
        CW_LOG_W("图片尺寸与模型尺寸不匹配: %dx%d != %ux%u", w, h, image_width, image_height);
        this->release();
        return false;
    }
    return true;
}

std::generator<caiwei::text::CompletionResult>
caiwei::context::VLMRKNN3Context::run(const caiwei::text::CompletionsRequest& request) {
    return this->generate(request);
}

bool caiwei::context::VLMRKNN3Context::fill_media(
    const int media_i,
    const int deepstack_i,
    const int width,
    const int height,
    const int channels,
    const int w,
    const int h,
    std::vector<float16>& embeds,
    std::vector<rknn3_aux_tensor>& deepstack_tensors,
    const std::vector<uint8_t>& media_data
) {
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    if (!caiwei::media::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale)) {
        return false;
    }
    // clang-format off
    thread_local std::vector<uint8_t> pad;
    thread_local std::vector<int8_t>  pad_i8;
    thread_local std::vector<uint8_t> chw_u8;
    thread_local std::vector<int8_t>  chw_i8;
    // clang-format on
    const size_t image_size = static_cast<size_t>(w) * h * channels;
    if (pad.size() != image_size) {
        pad.resize(image_size);
    }
    std::fill(pad.begin(), pad.end(), 0);
    if (!caiwei::media::resize(media_data.data(), pad.data(), width, height, w, h, dst_w, dst_h, pad_w, pad_h, channels)) {
        return false;
    }
    auto& tensor = this->media_input[0];
    if (tensor.attr->layout == RKNN3_TENSOR_NCHW) {
        if (tensor.attr->dtype == RKNN3_TENSOR_INT8) {
            if (chw_i8.size() != image_size) {
                chw_i8.resize(image_size);
            }
            caiwei::media::hwc_to_chw_u8_i8(pad.data(), chw_i8.data(), h, w, channels);
            std::memcpy((int8_t*) tensor.mem->virt_addr, chw_i8.data(), chw_i8.size());
        } else if (tensor.attr->dtype == RKNN3_TENSOR_UINT8) {
            if (chw_u8.size() != image_size) {
                chw_u8.resize(image_size);
            }
            caiwei::media::hwc_to_chw(pad.data(), chw_u8.data(), h, w, channels);
            std::memcpy((uint8_t*) tensor.mem->virt_addr, chw_u8.data(), chw_u8.size());
        } else {
            CW_LOG_W("不支持的输入格式: %s", rknn3_get_type_string(tensor.attr->dtype));
            return false;
        }
    } else if (tensor.attr->layout == RKNN3_TENSOR_NHWC) {
        if (tensor.attr->dtype == RKNN3_TENSOR_INT8) {
            if (pad_i8.size() != image_size) {
                pad_i8.resize(image_size);
            }
            caiwei::type::u8_to_i8(pad.data(), pad.size(), pad_i8.data());
            std::memcpy((int8_t*) tensor.mem->virt_addr, pad_i8.data(), pad_i8.size());
        } else if (tensor.attr->dtype == RKNN3_TENSOR_UINT8) {
            std::memcpy((uint8_t*) tensor.mem->virt_addr, pad.data(), pad.size());
        } else {
            CW_LOG_W("不支持的输入格式: %s", rknn3_get_type_string(tensor.attr->dtype));
            return false;
        }
    } else {
        CW_LOG_W("不支持的输入布局: %s", rknn3_get_layout_string(tensor.attr->layout));
        return false;
    }
    int ret = 0;
    for (auto& v : this->media_input) {
        ret = rknn3_mem_sync(this->media_context, v.mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            CW_LOG_W("输入失败: %d", ret);
            return false;
        }
    }
    ret = rknn3_run(
        this->media_context,
        this->media_input.data(),
        this->media_input.size(),
        this->media_output.data(),
        this->media_output.size()
    );
    if (ret != RKNN3_SUCCESS) {
        CW_LOG_W("计算失败: %d", ret);
        return false;
    }
    for (auto& v : this->media_output) {
        ret = rknn3_mem_sync(this->media_context, v.mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            CW_LOG_W("输出失败: %d", ret);
            return false;
        }
    }
    std::memcpy(
        ((char*) embeds.data()) + media_i * this->media_output[0].attr->aligned_size,
        this->media_output[0].mem->virt_addr,
        this->media_output[0].attr->aligned_size
    );
    for (size_t index = 0; index < deepstack_tensors.size(); ++index) {
        std::memcpy(
            ((char*) deepstack_tensors[index].mem->virt_addr) + deepstack_i * this->media_output[index + 1].attr->aligned_size,
            this->media_output[index + 1].mem->virt_addr,
            this->media_output[index + 1].attr->aligned_size
        );
    }
    CW_LOG_D("媒体嵌入: %d => %d", media_i, deepstack_i);
    return true;
}

std::vector<rknn3_llm_input> caiwei::context::VLMRKNN3Context::get_inputs(
    std::string& prompt, ContextSession* context_session, const caiwei::text::CompletionsRequest& request
) {
    prompt = this->chat_template.apply(
        this->special_token,
        request,
        this->special_token.audio_marker,
        this->special_token.image_marker,
        this->special_token.video_marker
    );
    if (prompt.empty()) {
        CW_LOG_W("提示词为空");
        return {};
    }
    uint64_t image_count = 0;
    uint64_t video_count = 0;
    uint64_t frame_count = 0;
    auto& output_tensor = this->media_output[0];
    std::vector<std::tuple<const caiwei::media::ImageFrame*, const std::vector<caiwei::media::VideoFrame>*>> data_list;
    for (const auto& message : request.messages) {
        for (const auto& image : message.image_data) {
            data_list.emplace_back(&image, nullptr);
            ++image_count;
        }
        for (const auto& video : message.video_data) {
            data_list.emplace_back(nullptr, &video);
            frame_count += video.size();
            ++video_count;
        }
    }
    CW_LOG_D("图片数量: %" PRIu64 " 视频数量: %" PRIu64 " 视频帧数: %" PRIu64, image_count, video_count, frame_count);
    context_session->image_frames = image_count + frame_count;
    context_session->image_embeds.resize(output_tensor.attr->aligned_size / sizeof(float16) * image_count);
    context_session->video_embeds.resize(output_tensor.attr->aligned_size / sizeof(float16) * frame_count);
    const int w = caiwei::env::get_int("CAIWEI_VLM_IMAGE_WIDTH");
    const int h = caiwei::env::get_int("CAIWEI_VLM_IMAGE_HEIGHT");
    if (!data_list.empty()) {
        if (!context_session->init_deepstack_tensors(this->media_output, image_count + frame_count)) {
            return {};
        }
        int image_i = 0;
        int video_i = 0;
        int deepstack_i = 0;
        uint64_t per_video_frame = 0;
        std::lock_guard<std::mutex> lock(this->media_mutex);
        for (const auto& [image_ptr, video_ptr] : data_list) {
            if (image_ptr != nullptr) {
                // clang-format off
                if (!this->fill_media(
                    image_i,
                    deepstack_i,
                    image_ptr->width,
                    image_ptr->height,
                    image_ptr->channels,
                    w,
                    h,
                    context_session->image_embeds,
                    context_session->deepstack_tensors,
                    image_ptr->data
                )) {
                    return {};
                }
                // clang-format on
                ++image_i;
                ++deepstack_i;
            }
            if (video_ptr != nullptr) {
                for (const auto& image_data : *video_ptr) {
                    // clang-format off
                    if (!this->fill_media(
                        video_i,
                        deepstack_i,
                        image_data.width,
                        image_data.height,
                        image_data.channels,
                        w,
                        h,
                        context_session->video_embeds,
                        context_session->deepstack_tensors,
                        image_data.data
                    )) {
                        return {};
                    }
                    // clang-format on
                    ++video_i;
                    ++deepstack_i;
                    if (per_video_frame == 0) {
                        per_video_frame = video_ptr->size();
                    } else if (per_video_frame != video_ptr->size()) {
                        CW_LOG_W("多个视频帧数没有固定: %" PRIu64 " != %zu", per_video_frame, video_ptr->size());
                        return {};
                    } else {
                        // -
                    }
                }
            }
        }
    }
    rknn3_llm_multimodal_tensor tensor {};
    // clang-format off
    tensor.name     = "input_embeds";
    tensor.prompt   = prompt.c_str();
    tensor.tokens   = nullptr;
    tensor.n_tokens = 0;
    // clang-format on
    tensor.enable_thinking = request.enable_thinking.value_or(false);
    if (image_count > 0) {
        // clang-format off
        tensor.image.image_embed    = context_session->image_embeds.data();
        tensor.image.n_image_tokens = output_tensor.attr->shape[0];
        tensor.image.n_image        = image_count;
        tensor.image.image_start    = this->special_token.b_image.c_str();
        tensor.image.image_end      = this->special_token.e_image.c_str();
        tensor.image.image_content  = this->special_token.c_image.c_str();
        tensor.image.image_width    = w;
        tensor.image.image_height   = h;
        // clang-format on
    }
    if (frame_count > 0) {
        // clang-format off
        tensor.video.video_embed       = context_session->video_embeds.data();
        tensor.video.n_frame_tokens    = output_tensor.attr->shape[0];
        tensor.video.n_frame_per_video = frame_count / video_count;
        tensor.video.n_video           = video_count;
        tensor.video.video_start       = this->special_token.b_video.c_str();
        tensor.video.video_end         = this->special_token.e_video.c_str();
        tensor.video.video_content     = this->special_token.c_video.c_str();
        tensor.video.frame_width       = w;
        tensor.video.frame_height      = h;
        // clang-format on
    }
    // clang-format off
    std::vector<rknn3_llm_input> inputs(1 + context_session->deepstack_tensors.size());
    inputs[0].input_type       = RKNN3_LLM_INPUT_MULTIMODAL;
    inputs[0].multimodal_input = tensor;
    for (size_t i = 0; i < context_session->deepstack_tensors.size(); ++i) {
        inputs[i + 1].input_type = RKNN3_LLM_INPUT_AUX;
        inputs[i + 1].aux_input  = context_session->deepstack_tensors[i];
    }
    // clang-format on
    return inputs;
}
