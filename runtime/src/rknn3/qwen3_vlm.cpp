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
}

bool caiwei::context::VLMRKNN3Context::load() {
    return this->load_model() && this->load_media_model() && this->init_internal_mems(0xFF, 0xFF);
}

std::generator<caiwei::text::Result> caiwei::context::VLMRKNN3Context::run(caiwei::text::CompletionsRequest& request) {
    return this->generate(request);
}

std::vector<rknn3_llm_input> caiwei::context::VLMRKNN3Context::get_inputs(rknn3_session* session, ContextSession* context_session, caiwei::text::CompletionsRequest& request) {
    int ret = 0;
    // TODO 文件拷贝
    // TODO 判断 NCHW or NHWC

    static std::vector<float16> img_embeds;
    static std::vector<rknn3_aux_tensor> deepstack_tensors(3);
    img_embeds.resize(this->media_output[0].attr->aligned_size / sizeof(float16) * 2);
    for (int i = 0; i < 3; ++i) {
        deepstack_tensors[i].attr = new rknn3_tensor_attr;
        deepstack_tensors[i].attr->index = i + 2;
        ret = rknn3_query(this->context, RKNN3_QUERY_INPUT_ATTR, deepstack_tensors[i].attr, sizeof(rknn3_tensor_attr));
        deepstack_tensors[i].mem = rknn3_create_mem(this->context, this->media_output[i + 1].attr->aligned_size * 2, deepstack_tensors[i].attr->core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        // TODO check mem nullptr
    }

    auto& message = request.messages[0];
    for (int i = 0; i < 2; ++i) {
        auto& video_data = message.video_data[0][0];
        memcpy((uint8_t*)this->media_input[0].mem->virt_addr, video_data.data.data(), video_data.data.size());
        for (auto& v : this->media_input) {
            ret = rknn3_mem_sync(this->media_context, v.mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
            if (ret != RKNN3_SUCCESS) {
                printf("rknn3_mem_sync input[%d] failed! ret=%d\n", 0, ret);
                return {};
            }
        }
        ret = rknn3_run(this->media_context, this->media_input.data(), this->media_input.size(), this->media_output.data(), this->media_output.size());
        if (ret < 0) {
            printf("rknn_run fail! ret=%d\n", ret);
            return {};
        }
        for (auto& v : this->media_output) {
            ret = rknn3_mem_sync(this->media_context, v.mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
            if (ret != RKNN3_SUCCESS) {
                printf("rknn3_mem_sync output[%d] failed! ret=%d\n", 0, ret);
                return {};
            }
        }
        memcpy(((uint8_t*)img_embeds.data()                  ) + i * this->media_output[0].mem->size,   (uint8_t*)this->media_output[0].mem->virt_addr, this->media_output[0].mem->size);
        memcpy(((uint8_t*)deepstack_tensors[0].mem->virt_addr) + i * this->media_output[1].mem->size,   (uint8_t*)this->media_output[1].mem->virt_addr, this->media_output[1].mem->size);
        memcpy(((uint8_t*)deepstack_tensors[1].mem->virt_addr) + i * this->media_output[2].mem->size,   (uint8_t*)this->media_output[2].mem->virt_addr, this->media_output[2].mem->size);
        memcpy(((uint8_t*)deepstack_tensors[2].mem->virt_addr) + i * this->media_output[3].mem->size,   (uint8_t*)this->media_output[3].mem->virt_addr, this->media_output[3].mem->size);
    }
    rknn3_llm_multimodal_tensor tensor{};
    // LLM Input
    tensor.name = "input_embeds";
    // Add image start tags to the prompt
    static std::string prompt_with_image = "描述图片内容。<image><image>";
    CW_LOG_I("prompt_with_image: %s", prompt_with_image.c_str());
    tensor.prompt = (prompt_with_image).c_str();
    tensor.tokens = nullptr;
    tensor.n_tokens = 0;
    tensor.image.image_embed = img_embeds.data(); // TODO
    if(this->media_output[0].attr->n_dims == 2) {
        tensor.image.n_image_tokens = this->media_output[0].attr->shape[0];
        tensor.image.n_image        = 2;
    } else {
        tensor.image.n_image_tokens = this->media_output[0].attr->shape[1];
        tensor.image.n_image        = this->media_output[0].attr->shape[0];
    }
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
    tensor.image.image_start   = "<|vision_start|>";
    tensor.image.image_end     = "<|vision_end|>";
    tensor.image.image_content = "<|image_pad|>";
    tensor.enable_thinking     = false;
    std::vector<rknn3_llm_input> inputs(1);
    inputs[0].input_type = RKNN3_LLM_INPUT_MULTIMODAL;
    inputs[0].multimodal_input = tensor;
    for (int i = 0; i < 3; ++i) {
        inputs[i + 1].input_type = RKNN3_LLM_INPUT_AUX;
        inputs[i + 1].aux_input  = deepstack_tensors[i];
    }
    return inputs;
}
