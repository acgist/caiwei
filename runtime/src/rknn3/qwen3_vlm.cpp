#include "caiwei/log.hpp"
#include "caiwei/image_tool.hpp"
#include "caiwei/runtime/rknn3.hpp"

#include <tuple>

#include "rga/im2d.hpp"

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
) : VLMContext(runtime)
  , RKNN3Context(
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
    for (int i = 0; i < this->deepstack_tensors.size(); ++i) {
        if (this->deepstack_tensors[i].attr) {
            delete this->deepstack_tensors[i].attr;
            this->deepstack_tensors[i].attr = nullptr;
        }
        if (this->deepstack_tensors[i].mem) {
            rknn3_destroy_mem(this->context, this->deepstack_tensors[i].mem);
            this->deepstack_tensors[i].mem = nullptr;
        }
    }
    this->deepstack_tensors.clear();
}

bool caiwei::context::VLMRKNN3Context::load() {
    return this->load_model(true) && this->load_media_model(true) && this->init_internal_tensor_mems();
}

std::generator<caiwei::text::Result> caiwei::context::VLMRKNN3Context::run(const caiwei::text::CompletionsRequest& request) {
    return this->generate(request);
}

bool caiwei::context::VLMRKNN3Context::fill_media(int media_i, int deepstack_i, std::vector<float16>& embeds, int width, int height, int channels, const std::vector<uint8_t>& data, std::vector<rknn3_aux_tensor>& deepstack_tensors) {
    const int w = 384;
    const int h = 384;
    float scale;
    int dst_w, dst_h, pad_w, pad_h;
    // TODO
    caiwei::image::resize(width, height, w, h, dst_w, dst_h, pad_w, pad_h, scale);
    std::vector<uint8_t> dst(dst_w * dst_h * channels);
    std::vector<uint8_t> pad(    w *     h * channels);
    rga_buffer_t src_img = wrapbuffer_virtualaddr((void*) data.data(), width, height, RK_FORMAT_RGB_888);
    rga_buffer_t dst_img = wrapbuffer_virtualaddr(pad.data(), w, h, RK_FORMAT_RGB_888);
    rga_buffer_t pat_img = wrapbuffer_virtualaddr(nullptr, 0, 0, 0);
    im_rect src_rect = { 0, 0, width, height };
    im_rect dst_rect = { pad_w, pad_h, dst_w, dst_h };
    im_rect pat_rect = { 0, 0, 0, 0 };
    improcess(src_img, dst_img, pat_img, src_rect, dst_rect, pat_rect, IM_SYNC);
    int ret = 0;
    memcpy((uint8_t*)this->media_input[0].mem->virt_addr, pad.data(), pad.size());
    for (auto& v : this->media_input) {
        ret = rknn3_mem_sync(this->media_context, v.mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            printf("rknn3_mem_sync input[%d] failed! ret=%d\n", 0, ret);
            return false;
        }
    }
    ret = rknn3_run(this->media_context, this->media_input.data(), this->media_input.size(), this->media_output.data(), this->media_output.size());
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return false;
    }
    for (auto& v : this->media_output) {
        ret = rknn3_mem_sync(this->media_context, v.mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            printf("rknn3_mem_sync output[%d] failed! ret=%d\n", 0, ret);
            return false;
        }
    }
    memcpy(((uint8_t*)embeds.data()                      ) + media_i     * this->media_output[0].mem->size, (uint8_t*)this->media_output[0].mem->virt_addr, this->media_output[0].mem->size);
    memcpy(((uint8_t*)deepstack_tensors[0].mem->virt_addr) + deepstack_i * this->media_output[1].mem->size, (uint8_t*)this->media_output[1].mem->virt_addr, this->media_output[1].mem->size);
    memcpy(((uint8_t*)deepstack_tensors[1].mem->virt_addr) + deepstack_i * this->media_output[2].mem->size, (uint8_t*)this->media_output[2].mem->virt_addr, this->media_output[2].mem->size);
    memcpy(((uint8_t*)deepstack_tensors[2].mem->virt_addr) + deepstack_i * this->media_output[3].mem->size, (uint8_t*)this->media_output[3].mem->virt_addr, this->media_output[3].mem->size);
    CW_LOG_D("fill_media %d = %d", media_i, deepstack_i);
    return true;
}

bool caiwei::context::VLMRKNN3Context::init_deepstack_tensors(int length, int size) {
    this->deepstack_tensors.resize(size);
    for (int i = 0; i < this->deepstack_tensors.size(); ++i) {
        if (this->deepstack_tensors[i].attr == nullptr) {
            this->deepstack_tensors[i].attr = new rknn3_tensor_attr;
            this->deepstack_tensors[i].attr->index = i + 2;
            int ret = rknn3_query(this->context, RKNN3_QUERY_INPUT_ATTR, this->deepstack_tensors[i].attr, sizeof(rknn3_tensor_attr));
        }
        auto mem_size = this->media_output[i + 1].attr->aligned_size * length;
        if (this->deepstack_tensors[i].mem == nullptr) {
            this->deepstack_tensors[i].mem = rknn3_create_mem(this->context, mem_size, this->deepstack_tensors[i].attr->core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
            if (this->deepstack_tensors[i].mem == nullptr) {
                return false;
            }
            CW_LOG_D("deepstack_tensors[%d].mem=%p = %d/%d", i, this->deepstack_tensors[i].mem, this->deepstack_tensors[i].mem->size, mem_size);
        } else if (this->deepstack_tensors[i].mem->size != mem_size) {
            rknn3_destroy_mem(this->context, this->deepstack_tensors[i].mem);
            this->deepstack_tensors[i].mem = rknn3_create_mem(this->context, mem_size, this->deepstack_tensors[i].attr->core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
            if (this->deepstack_tensors[i].mem == nullptr) {
                return false;
            }
            CW_LOG_D("deepstack_tensors[%d].mem=%p = %d/%d", i, this->deepstack_tensors[i].mem, this->deepstack_tensors[i].mem->size, mem_size);
        } else {
            // -
        }
    }
    return true;
}

std::vector<rknn3_llm_input> caiwei::context::VLMRKNN3Context::get_inputs(std::string& prompt, ContextSession* context_session, const caiwei::text::CompletionsRequest& request) {
    // TODO 线程安全
    bool enable_thinking = request.enable_thinking.value_or(false);
    if (this->media_output[0].attr->n_dims != 2) {
        CW_LOG_W("不支持的输出维度 %d", this->media_output[0].attr->n_dims);
        return {};
    }
    int ret = 0;
    // TODO 文件拷贝
    // TODO 判断 NCHW or NHWC
    int image_count = 0;
    int video_count = 0;
    int frame_count = 0;
    std::vector<std::tuple<const caiwei::media::ImageFrame*, const std::vector<caiwei::media::VideoFrame>*>> data_list;
    for (const auto& message : request.messages) {
        if (message.role != caiwei::text::ROLE_USER) {
            continue;
        }
        if (std::holds_alternative<std::string>(message.content.value())) {
            continue;
        }
        const std::vector<caiwei::text::CompletionsRequestMessageContentItem>& items = std::get<std::vector<caiwei::text::CompletionsRequestMessageContentItem>>(message.content.value());
        auto iter = items.begin();
        int image_index = 0;
        int video_index = 0;
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
                    for (const auto& frame : video) {
                        frame_count++;
                    }
                    data_list.emplace_back(nullptr, &video);
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
                for (const auto& frame : video) {
                    frame_count++;
                }
                data_list.emplace_back(nullptr, &video);
                video_index++;
                video_count++;
            }
        }
    }
    context_session->image_frames = image_count + frame_count;
    CW_LOG_D("图片数量: %d 视频数量: %d 视频帧数: %d", image_count, video_count, frame_count);
    // TODO 视频图片长度必须一致
    this->image_embeds.resize(this->media_output[0].attr->aligned_size / sizeof(float16) * image_count);
    this->video_embeds.resize(this->media_output[0].attr->aligned_size / sizeof(float16) * frame_count);
    this->init_deepstack_tensors(image_count + frame_count);
    int image_i = 0;
    int video_i = 0;
    int deepstack_i = 0;
    for (const auto& [image_ptr, video_ptr] : data_list) {
        if (image_ptr != nullptr) {
            if (!fill_media(image_i, deepstack_i, this->image_embeds, image_ptr->width, image_ptr->height, image_ptr->channels, image_ptr->data, deepstack_tensors)) {
                return {};
            }
            ++image_i;
            ++deepstack_i;
        }
        if (video_ptr != nullptr) {
            for (const auto& image_data : *video_ptr) {
                if (!fill_media(video_i, deepstack_i, this->video_embeds, image_data.width, image_data.height, image_data.channels, image_data.data, deepstack_tensors)) {
                    return {};
                }
                ++video_i;
                ++deepstack_i;
            }
        }
    }
    prompt = this->chat_template.apply(this->special_token, request, this->special_token.audio_marker, this->special_token.image_marker, this->special_token.video_marker);
    CW_LOG_I("prompt: %s", prompt.c_str());
    rknn3_llm_multimodal_tensor tensor{};
    tensor.name     = "input_embeds";
    tensor.prompt   = prompt.c_str();
    tensor.tokens   = nullptr;
    tensor.n_tokens = 0;
    tensor.enable_thinking = enable_thinking;
    if (image_count == 0 && frame_count == 0) {
        CW_LOG_W("视频数据为空");
        return {};
    }
    // 图片
    if (image_count > 0) {
        tensor.image.image_embed = this->image_embeds.data();
        tensor.image.n_image_tokens = this->media_output[0].attr->shape[0];
        tensor.image.n_image        = image_count;
        if (this->media_input[0].attr->layout == RKNN3_TENSOR_NCHW) {
            tensor.image.image_width   = this->media_input[0].attr->shape[3];
            tensor.image.image_height  = this->media_input[0].attr->shape[2];
        } else if (this->media_input[0].attr->layout == RKNN3_TENSOR_NHWC) {
            tensor.image.image_width   = this->media_input[0].attr->shape[2];
            tensor.image.image_height  = this->media_input[0].attr->shape[1];
        } else {
            printf("rknn3_mem_sync input[%d] failed! layout=%d\n", 0, this->media_input[0].attr->layout);
            return {};
        }
        tensor.image.image_start   = this->special_token.b_image.c_str();
        tensor.image.image_end     = this->special_token.e_image.c_str();
        tensor.image.image_content = this->special_token.c_image.c_str();
    }
    // 视频
    if (frame_count > 0) {
        tensor.video.video_embed = this->video_embeds.data();
        tensor.video.n_frame_tokens    = this->media_output[0].attr->shape[0];
        tensor.video.n_frame_per_video = frame_count / video_count;
        tensor.video.n_video           = video_count;
        if (this->media_input[0].attr->layout == RKNN3_TENSOR_NCHW) {
            tensor.video.frame_width   = this->media_input[0].attr->shape[3];
            tensor.video.frame_height  = this->media_input[0].attr->shape[2];
        } else if (this->media_input[0].attr->layout == RKNN3_TENSOR_NHWC) {
            tensor.video.frame_width   = this->media_input[0].attr->shape[2];
            tensor.video.frame_height  = this->media_input[0].attr->shape[1];
        } else {
            printf("rknn3_mem_sync input[%d] failed! layout=%d\n", 0, this->media_input[0].attr->layout);
            return {};
        }
        tensor.video.video_start   = this->special_token.b_video.c_str();
        tensor.video.video_end     = this->special_token.e_video.c_str();
        tensor.video.video_content = this->special_token.c_video.c_str();
    }
    std::vector<rknn3_llm_input> inputs(4);
    inputs[0].input_type = RKNN3_LLM_INPUT_MULTIMODAL;
    inputs[0].multimodal_input = tensor;
    for (int i = 0; i < 3; ++i) {
        inputs[i + 1].input_type = RKNN3_LLM_INPUT_AUX;
        inputs[i + 1].aux_input  = deepstack_tensors[i];
    }
    return inputs;
}
