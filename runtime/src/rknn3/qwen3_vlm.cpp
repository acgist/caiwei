#include <tuple>
#include <cstddef>

#include "rga/im2d.hpp"

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
    if (!this->deepstack_tensors.empty()) {
        for (size_t i = 0; i < this->deepstack_tensors.size(); ++i) {
            auto& tensor = this->deepstack_tensors[i];
            if (tensor.mem) {
                CW_LOG_I("释放VLMRKNN3Context deepstack_tensors mem: %p", tensor.mem);
                rknn3_destroy_mem(this->context, tensor.mem);
                tensor.mem = nullptr;
            }
            if (tensor.attr) {
                CW_LOG_I("释放VLMRKNN3Context deepstack_tensors attr: %p", tensor.attr);
                delete tensor.attr;
                tensor.attr = nullptr;
            }
        }
        this->deepstack_tensors.clear();
    }
}

bool caiwei::context::VLMRKNN3Context::load() {
    return
    this->load_model(true)       &&
    this->load_media_model(true) &&
    this->init_internal_tensor_mems();
}

std::generator<caiwei::text::CompletionResult> caiwei::context::VLMRKNN3Context::run(const caiwei::text::CompletionsRequest& request) {
    return this->generate(request);
}

bool caiwei::context::VLMRKNN3Context::fill_media(int media_i, int deepstack_i, std::vector<float16>& embeds, int width, int height, int channels, const std::vector<uint8_t>& data, std::vector<rknn3_aux_tensor>& deepstack_tensors) {
    if (deepstack_tensors.size() < 3 || this->media_output.size() < 4) {
        CW_LOG_W("模型没有配置成功");
        return false;
    }
    const int w = caiwei::env::get_int("CAIWEI_VLM_WIDTH");
    const int h = caiwei::env::get_int("CAIWEI_VLM_HEIGHT");
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    caiwei::image::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale);
    std::vector<uint8_t> dst(dst_w * dst_h * channels);
    std::vector<uint8_t> pad(    w *     h * channels);
    rga_buffer_t src_img = wrapbuffer_virtualaddr((void*) data.data(), width, height, RK_FORMAT_RGB_888);
    rga_buffer_t dst_img = wrapbuffer_virtualaddr(pad.data(),          w,     h,      RK_FORMAT_RGB_888);
    rga_buffer_t pat_img = wrapbuffer_virtualaddr(nullptr,             0,     0,      0                );
    im_rect src_rect = { 0,     0,     width, height };
    im_rect dst_rect = { pad_w, pad_h, dst_w, dst_h  };
    im_rect pat_rect = { 0,     0,     0,     0      };
    const auto status = improcess(src_img, dst_img, pat_img, src_rect, dst_rect, pat_rect, IM_SYNC);
    if (status != IM_STATUS_SUCCESS) {
        CW_LOG_W("图片处理失败: %d", status);
        return {};
    }
    int ret = 0;
    auto& tensor = this->media_input[0];
    if (tensor.attr->layout == RKNN3_TENSOR_NCHW) {
        if (tensor.attr->dtype == RKNN3_TENSOR_INT8) {
            std::vector<int8_t> chw(w * h * channels);
            caiwei::image::hwc_to_chw_u8_i8(pad.data(), chw.data(), h, w, channels);
            std::memcpy((int8_t*) tensor.mem->virt_addr, chw.data(), chw.size());
        } else if (tensor.attr->dtype == RKNN3_TENSOR_UINT8) {
            std::vector<uint8_t> chw(w * h * channels);
            caiwei::image::hwc_to_chw(pad.data(), chw.data(), h, w, channels);
            std::memcpy((uint8_t*) tensor.mem->virt_addr, chw.data(), chw.size());
        } else {
            CW_LOG_W("不支持的输入格式: %s", rknn3_get_type_string(tensor.attr->dtype));
            return false;
        }
    } else if (tensor.attr->layout == RKNN3_TENSOR_NHWC) {
        if (tensor.attr->dtype == RKNN3_TENSOR_INT8) {
            std::vector<int8_t> pad_i8(w * h * channels);
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
    for (auto& v : this->media_input) {
        ret = rknn3_mem_sync(this->media_context, v.mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            CW_LOG_W("输入失败: %d", ret);
            return false;
        }
    }
    ret = rknn3_run(this->media_context, this->media_input.data(), this->media_input.size(), this->media_output.data(), this->media_output.size());
    if (ret < 0) {
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
    std::memcpy(((uint8_t*) embeds.data()                      ) + media_i     * this->media_output[0].mem->size, (uint8_t*) this->media_output[0].mem->virt_addr, this->media_output[0].mem->size);
    std::memcpy(((uint8_t*) deepstack_tensors[0].mem->virt_addr) + deepstack_i * this->media_output[1].mem->size, (uint8_t*) this->media_output[1].mem->virt_addr, this->media_output[1].mem->size);
    std::memcpy(((uint8_t*) deepstack_tensors[1].mem->virt_addr) + deepstack_i * this->media_output[2].mem->size, (uint8_t*) this->media_output[2].mem->virt_addr, this->media_output[2].mem->size);
    std::memcpy(((uint8_t*) deepstack_tensors[2].mem->virt_addr) + deepstack_i * this->media_output[3].mem->size, (uint8_t*) this->media_output[3].mem->virt_addr, this->media_output[3].mem->size);
    CW_LOG_D("媒体嵌入: %d => %d", media_i, deepstack_i);
    return true;
}

bool caiwei::context::VLMRKNN3Context::init_deepstack_tensors(int length, int size) {
    this->deepstack_tensors.resize(size);
    for (size_t i = 0; i < this->deepstack_tensors.size(); ++i) {
        auto& tensor = this->deepstack_tensors[i];
        if (tensor.attr == nullptr) {
            tensor.attr = new rknn3_tensor_attr;
            tensor.attr->index = i + 2;
            int ret = rknn3_query(this->context, RKNN3_QUERY_INPUT_ATTR, tensor.attr, sizeof(rknn3_tensor_attr));
            if (ret != RKNN3_SUCCESS) {
                CW_LOG_W("输入参数查询失败: %d", ret);
                return false;
            }
            rknn3_dump_tensor_attr("输入参数", tensor.attr);
        }
        const auto mem_size = this->media_output[i + 1].attr->aligned_size * length;
        if (tensor.mem == nullptr) {
            tensor.mem = rknn3_create_mem(this->context, mem_size, tensor.attr->core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
            if (tensor.mem == nullptr) {
                CW_LOG_W("deepstack内存创建失败: %zu = %lld", i, mem_size);
                return false;
            }
            CW_LOG_I("deepstack内存创建成功: %zu => %p = %lld | %lld", i, tensor.mem, tensor.mem->size, mem_size);
        } else if (tensor.mem->size != mem_size) {
            rknn3_destroy_mem(this->context, tensor.mem);
            tensor.mem = rknn3_create_mem(this->context, mem_size, tensor.attr->core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
            if (tensor.mem == nullptr) {
                CW_LOG_W("deepstack内存修改失败: %zu = %lld", i, mem_size);
                return false;
            }
            CW_LOG_I("deepstack内存修改成功: %zu => %p = %lld | %lld", i, tensor.mem, tensor.mem->size, mem_size);
        } else {
            // -
        }
    }
    return true;
}

std::vector<rknn3_llm_input> caiwei::context::VLMRKNN3Context::get_inputs(std::string& prompt, ContextSession* context_session, const caiwei::text::CompletionsRequest& request) {
    uint64_t image_count = 0;
    uint64_t video_count = 0;
    uint64_t frame_count = 0;
    std::vector<std::tuple<const caiwei::media::ImageFrame*, const std::vector<caiwei::media::VideoFrame>*>> data_list;
    auto& input_tensor  = this->media_input[0];
    auto& output_tensor = this->media_output[0];
    uint32_t image_width = 0;
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
    for (const auto& message : request.messages) {
        if (message.role != caiwei::text::ROLE_USER) {
            continue;
        }
        if (!message.content.has_value()) {
            continue;
        }
        if (std::holds_alternative<std::string>(message.content.value())) {
            continue;
        }
        const auto& items = std::get<std::vector<caiwei::text::CompletionsRequestMessageContentItem>>(message.content.value());
        auto iter = items.begin();
        uint64_t image_index = 0;
        uint64_t video_index = 0;
        while (iter != items.end()) {
            if (iter->type == "text") {
                // -
            } else if (iter->type == "image" || iter->type == "image_url") {
                if (image_index < message.image_data.size()) {
                    const auto& image = message.image_data[image_index];
                    data_list.emplace_back(&image, nullptr);
                    image_index++;
                    image_count++;
                }
            } else if (iter->type == "video" || iter->type == "video_url") {
                if (video_index < message.video_data.size()) {
                    const auto& video = message.video_data[video_index];
                    data_list.emplace_back(nullptr, &video);
                    frame_count += video.size();
                    video_index++;
                    video_count++;
                }
            } else {
                CW_LOG_W("未知类型: %s", iter->type.value_or("-").c_str());
            }
            iter++;
        }
        if (image_index < message.image_data.size()) {
            for (; image_index < message.image_data.size();) {
                const auto& image = message.image_data[image_index];
                data_list.emplace_back(&image, nullptr);
                image_index++;
                image_count++;
            }
        }
        if (video_index < message.video_data.size()) {
            for (; video_index < message.video_data.size();) {
                const auto& video = message.video_data[video_index];
                data_list.emplace_back(nullptr, &video);
                frame_count += video.size();
                video_index++;
                video_count++;
            }
        }
    }
    CW_LOG_D("图片数量: %d 视频数量: %d 视频帧数: %d", image_count, video_count, frame_count);
    context_session->image_frames = image_count + frame_count;
    this->image_embeds.resize(output_tensor.attr->aligned_size / sizeof(float16) * image_count);
    this->video_embeds.resize(output_tensor.attr->aligned_size / sizeof(float16) * frame_count);
    this->init_deepstack_tensors(image_count + frame_count);
    int image_i = 0;
    int video_i = 0;
    int deepstack_i = 0;
    uint64_t per_video_frame = 0;
    for (const auto& [image_ptr, video_ptr] : data_list) {
        if (image_ptr != nullptr) {
            if (!this->fill_media(
                image_i,
                deepstack_i,
                this->image_embeds,
                image_ptr->width,
                image_ptr->height,
                image_ptr->channels,
                image_ptr->data,
                this->deepstack_tensors
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
                    this->video_embeds,
                    image_data.width,
                    image_data.height,
                    image_data.channels,
                    image_data.data,
                    this->deepstack_tensors
                )) {
                    return {};
                }
                ++video_i;
                ++deepstack_i;
                if (per_video_frame == 0) {
                    per_video_frame = video_ptr->size();
                } else if (per_video_frame != video_ptr->size()) {
                    CW_LOG_W("视频帧数没有统一: %d != %d", per_video_frame, video_ptr->size());
                } else {
                    // -
                }
            }
        }
    }
    prompt = this->chat_template.apply(this->special_token, request, this->special_token.audio_marker, this->special_token.image_marker, this->special_token.video_marker);
    rknn3_llm_multimodal_tensor tensor{};
    tensor.name     = "input_embeds";
    tensor.prompt   = prompt.c_str();
    tensor.tokens   = nullptr;
    tensor.n_tokens = 0;
    tensor.enable_thinking = request.enable_thinking.value_or(false);
    if (image_count > 0) {
        tensor.image.image_embed    = this->image_embeds.data();
        tensor.image.n_image_tokens = output_tensor.attr->shape[0];
        tensor.image.n_image        = image_count;
        tensor.image.image_start    = this->special_token.b_image.c_str();
        tensor.image.image_end      = this->special_token.e_image.c_str();
        tensor.image.image_content  = this->special_token.c_image.c_str();
        tensor.image.image_width    = image_width;
        tensor.image.image_height   = image_height;
    }
    if (frame_count > 0) {
        tensor.video.video_embed       = this->video_embeds.data();
        tensor.video.n_frame_tokens    = output_tensor.attr->shape[0];
        tensor.video.n_frame_per_video = frame_count / video_count;
        tensor.video.n_video           = video_count;
        tensor.video.video_start       = this->special_token.b_video.c_str();
        tensor.video.video_end         = this->special_token.e_video.c_str();
        tensor.video.video_content     = this->special_token.c_video.c_str();
        tensor.video.frame_width       = image_width;
        tensor.video.frame_height      = image_height;
    }
    std::vector<rknn3_llm_input> inputs(1 + this->deepstack_tensors.size());
    inputs[0].input_type       = RKNN3_LLM_INPUT_MULTIMODAL;
    inputs[0].multimodal_input = tensor;
    for (size_t i = 0; i < this->deepstack_tensors.size(); ++i) {
        inputs[i + 1].input_type = RKNN3_LLM_INPUT_AUX;
        inputs[i + 1].aux_input  = this->deepstack_tensors[i];
    }
    return inputs;
}
