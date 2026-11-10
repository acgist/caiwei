#include <tuple>
#include <cstddef>

#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/type.hpp"
#include "caiwei/image_tool.hpp"
#include "caiwei/runtime/rknn3.hpp"

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
) : VLMContext(runtime),
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
    return true;
}

std::generator<caiwei::text::CompletionResult> caiwei::context::VLMRKNN3Context::run(const caiwei::text::CompletionsRequest& request) {
    return this->generate(request);
}

bool caiwei::context::VLMRKNN3Context::fill_media(
    int media_i,
    int deepstack_i,
    int width,
    int height,
    int channels,
    int image_width,
    int image_height,
    std::vector<float16>& embeds,
    std::vector<rknn3_aux_tensor>& deepstack_tensors,
    const std::vector<uint8_t>& media_data
) {
    if (deepstack_tensors.size() != 3 || this->media_output.size() != 4) {
        CW_LOG_W("模型没有配置成功");
        return false;
    }
    const int w = caiwei::env::get_int("CAIWEI_VLM_IMAGE_WIDTH");
    const int h = caiwei::env::get_int("CAIWEI_VLM_IMAGE_HEIGHT");
    if (w != image_width || h != image_height) {
        CW_LOG_W("图片尺寸与模型尺寸不匹配");
        return false;
    }
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    if (!caiwei::image::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale)) {
        return false;
    }
    const size_t image_size = static_cast<size_t>(w) * h * channels;
    std::vector<uint8_t> pad(image_size);
    if (!caiwei::image::resize(media_data.data(), pad.data(), width, height, w, h, dst_w, dst_h, pad_w, pad_h, channels)) {
        return false;
    }
    auto& tensor = this->media_input[0];
    if (tensor.attr->layout == RKNN3_TENSOR_NCHW) {
        if (tensor.attr->dtype == RKNN3_TENSOR_INT8) {
            std::vector<int8_t> chw(image_size);
            caiwei::image::hwc_to_chw_u8_i8(pad.data(), chw.data(), h, w, channels);
            std::memcpy((int8_t*) tensor.mem->virt_addr, chw.data(), chw.size());
        } else if (tensor.attr->dtype == RKNN3_TENSOR_UINT8) {
            std::vector<uint8_t> chw(image_size);
            caiwei::image::hwc_to_chw(pad.data(), chw.data(), h, w, channels);
            std::memcpy((uint8_t*) tensor.mem->virt_addr, chw.data(), chw.size());
        } else {
            CW_LOG_W("不支持的输入格式: %s", rknn3_get_type_string(tensor.attr->dtype));
            return false;
        }
    } else if (tensor.attr->layout == RKNN3_TENSOR_NHWC) {
        if (tensor.attr->dtype == RKNN3_TENSOR_INT8) {
            std::vector<int8_t> pad_i8(image_size);
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
    ret = rknn3_run(this->media_context, this->media_input.data(), this->media_input.size(), this->media_output.data(), this->media_output.size());
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
    std::memcpy(((uint8_t*) embeds.data()                      ) + media_i     * this->media_output[0].attr->aligned_size, (uint8_t*) this->media_output[0].mem->virt_addr, this->media_output[0].attr->aligned_size);
    std::memcpy(((uint8_t*) deepstack_tensors[0].mem->virt_addr) + deepstack_i * this->media_output[1].attr->aligned_size, (uint8_t*) this->media_output[1].mem->virt_addr, this->media_output[1].attr->aligned_size);
    std::memcpy(((uint8_t*) deepstack_tensors[1].mem->virt_addr) + deepstack_i * this->media_output[2].attr->aligned_size, (uint8_t*) this->media_output[2].mem->virt_addr, this->media_output[2].attr->aligned_size);
    std::memcpy(((uint8_t*) deepstack_tensors[2].mem->virt_addr) + deepstack_i * this->media_output[3].attr->aligned_size, (uint8_t*) this->media_output[3].mem->virt_addr, this->media_output[3].attr->aligned_size);
    CW_LOG_D("媒体嵌入: %d => %d", media_i, deepstack_i);
    return true;
}

std::vector<rknn3_llm_input> caiwei::context::VLMRKNN3Context::get_inputs(std::string& prompt, ContextSession* context_session, const caiwei::text::CompletionsRequest& request) {
    prompt = this->chat_template.apply(this->special_token, request, this->special_token.audio_marker, this->special_token.image_marker, this->special_token.video_marker);
    if (prompt.empty()) {
        CW_LOG_W("提示词为空");
        return {};
    }
    if (this->media_input.size() != 1 || this->media_output.size() != 4) {
        CW_LOG_W("模型没有加载完成");
        return {};
    }
    uint64_t image_count = 0;
    uint64_t video_count = 0;
    uint64_t frame_count = 0;
    auto& input_tensor  = this->media_input[0];
    auto& output_tensor = this->media_output[0];
    uint32_t image_width  = 0;
    uint32_t image_height = 0;
    if (input_tensor.attr->layout == RKNN3_TENSOR_NCHW) {
        image_width   = input_tensor.attr->shape[3];
        image_height  = input_tensor.attr->shape[2];
    } else if (input_tensor.attr->layout == RKNN3_TENSOR_NHWC) {
        image_width   = input_tensor.attr->shape[2];
        image_height  = input_tensor.attr->shape[1];
    } else {
        CW_LOG_W("不支持的输入布局: %s", rknn3_get_layout_string(input_tensor.attr->layout));
        return {};
    }
    std::vector<std::tuple<const caiwei::media::ImageFrame*, const std::vector<caiwei::media::VideoFrame>*>> data_list;
    for (const auto& message : request.messages) {
        for (const auto& image : message.image_data) {
            data_list.emplace_back(&image, nullptr);
            image_count++;
        }
        for (const auto& video : message.video_data) {
            data_list.emplace_back(nullptr, &video);
            frame_count += video.size();
            video_count++;
        }
    }
    CW_LOG_D("图片数量: %" PRIu64 " 视频数量: %" PRIu64 " 视频帧数: %" PRIu64, image_count, video_count, frame_count);
    context_session->image_frames = image_count + frame_count;
    context_session->image_embeds.resize(output_tensor.attr->aligned_size / sizeof(float16) * image_count);
    context_session->video_embeds.resize(output_tensor.attr->aligned_size / sizeof(float16) * frame_count);
    if (!data_list.empty()) {
        if (!context_session->init_deepstack_tensors(this->media_output, image_count + frame_count)) {
            return {};
        }
    }
    int image_i = 0;
    int video_i = 0;
    int deepstack_i = 0;
    uint64_t per_video_frame = 0;
    {
        std::lock_guard<std::mutex> lock(this->media_mutex);
        for (const auto& [image_ptr, video_ptr] : data_list) {
            if (image_ptr != nullptr) {
                if (!this->fill_media(
                    image_i,
                    deepstack_i,
                    image_ptr->width,
                    image_ptr->height,
                    image_ptr->channels,
                    image_width,
                    image_height,
                    context_session->image_embeds,
                    context_session->deepstack_tensors,
                    image_ptr->data
                )) {
                    return {};
                }
                ++image_i;
                ++deepstack_i;
            }
            if (video_ptr != nullptr) {
                for (const auto& image_data : *video_ptr) {
                    if (!this->fill_media(
                        video_i,
                        deepstack_i,
                        image_data.width,
                        image_data.height,
                        image_data.channels,
                        image_width,
                        image_height,
                        context_session->video_embeds,
                        context_session->deepstack_tensors,
                        image_data.data
                    )) {
                        return {};
                    }
                    ++video_i;
                    ++deepstack_i;
                    if (per_video_frame == 0) {
                        per_video_frame = video_ptr->size();
                    } else if (per_video_frame != video_ptr->size()) {
                        CW_LOG_W("视频帧数没有固定: %" PRIu64 " != %zu", per_video_frame, video_ptr->size());
                        return {};
                    } else {
                        // -
                    }
                }
            }
        }
    }
    rknn3_llm_multimodal_tensor tensor{};
    tensor.name     = "input_embeds";
    tensor.prompt   = prompt.c_str();
    tensor.tokens   = nullptr;
    tensor.n_tokens = 0;
    tensor.enable_thinking = request.enable_thinking.value_or(false);
    if (image_count > 0) {
        tensor.image.image_embed    = context_session->image_embeds.data();
        tensor.image.n_image_tokens = output_tensor.attr->shape[0];
        tensor.image.n_image        = image_count;
        tensor.image.image_start    = this->special_token.b_image.c_str();
        tensor.image.image_end      = this->special_token.e_image.c_str();
        tensor.image.image_content  = this->special_token.c_image.c_str();
        tensor.image.image_width    = image_width;
        tensor.image.image_height   = image_height;
    }
    if (frame_count > 0) {
        tensor.video.video_embed       = context_session->video_embeds.data();
        tensor.video.n_frame_tokens    = output_tensor.attr->shape[0];
        tensor.video.n_frame_per_video = frame_count / video_count;
        tensor.video.n_video           = video_count;
        tensor.video.video_start       = this->special_token.b_video.c_str();
        tensor.video.video_end         = this->special_token.e_video.c_str();
        tensor.video.video_content     = this->special_token.c_video.c_str();
        tensor.video.frame_width       = image_width;
        tensor.video.frame_height      = image_height;
    }
    std::vector<rknn3_llm_input> inputs(1 + context_session->deepstack_tensors.size());
    inputs[0].input_type       = RKNN3_LLM_INPUT_MULTIMODAL;
    inputs[0].multimodal_input = tensor;
    for (size_t i = 0; i < context_session->deepstack_tensors.size(); ++i) {
        inputs[i + 1].input_type = RKNN3_LLM_INPUT_AUX;
        inputs[i + 1].aux_input  = context_session->deepstack_tensors[i];
    }
    return inputs;
}
