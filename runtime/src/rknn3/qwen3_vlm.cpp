#include "caiwei/runtime/rknn3.hpp"

caiwei::context::VLMRKNN3Context::VLMRKNN3Context() {
    this->deepstack_tensor.resize(3);
}

caiwei::context::VLMRKNN3Context::~VLMRKNN3Context() {
    if (!this->internal_mems.empty()) {
        for (int i = 0; i < this->n_internal_mems; i++) {
            if (this->internal_mems[i]) {
                rknn3_destroy_mem(this->context, this->internal_mems[i]);
                this->internal_mems[i] = NULL;
            }
        }
        internal_mems.clear();
        this->n_internal_mems = 0;
    }
    for(int i = 0; i < 3; i++) {
        if (this->deepstack_tensor[i].mem) {
          rknn3_destroy_mem(this->context, this->deepstack_tensor[i].mem);
          this->deepstack_tensor[i].mem = nullptr;
        }
    }
    for (int i = 0; i < this->input_size; i++) {
        if (this->inputs[i].mem) {
            rknn3_destroy_mem(this->vlm_context, this->inputs[i].mem);
        }
        if (this->inputs[i].attr != NULL) {
            free(this->inputs[i].attr);
            this->inputs[i].attr = NULL;
        }
    }
    for (int i = 0; i < this->output_size; i++) {
        if (this->outputs[i].mem) {
            rknn3_destroy_mem(this->vlm_context, this->outputs[i].mem);
        }
        if (this->outputs[i].attr != NULL) {
            free(this->outputs[i].attr);
            this->outputs[i].attr = NULL;
        }
    }
    if (this->vlm_context != 0)
    {
        rknn3_destroy(this->vlm_context);
        this->vlm_context = 0;
    }
}


void normalize_and_to_nchw(const unsigned char *src, float16 *dst, int width, int height) {
    float mean[3] = {127.5f, 127.5f, 127.5f};
    float std[3] = {127.5f, 127.5f, 127.5f};
    int num_pixels = width * height;
    
    // 转换到NCHW
    for (int c = 0; c < 3; ++c) {  // 遍历R,G,B通道
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int nhwc_idx = (h * width + w) * 3 + c;  // NHWC索引
                int nchw_idx = c * num_pixels + h * width + w;  // NCHW索引
                dst[nchw_idx] = fp32_to_fp16(((float)src[nhwc_idx] - mean[c]) / std[c]); // 归一化
            }
        }
    }   
}

void transpose(float16 *in, float16 *out, int model_width, int model_height) {
    int64_t dims[8] = {2,3,int(model_height/32),2,16,int(model_width/32),2,16};    
    //[temporal_patch_size,channel, grid_h // merge_size,  merge_size, patch_size, 
    // grid_w // merge_size, merge_size, patch_size]
     int perm[8] = {2, 5, 3, 6, 1, 0, 4, 7};
    int64_t new_dims[8];
    for (int k = 0; k < 8; k++) {
        new_dims[k] = dims[perm[k]];
    }
    int64_t total = 1;
    for (int k = 0; k < 8; k++) {
        total *= dims[k];
    }
    // 计算原始strides
    int64_t strides[8];
    strides[7] = 1;
    for (int d = 6; d >= 0; d--) {
        strides[d] = strides[d + 1] * dims[d + 1];
    }

    // 使用嵌套循环遍历输出索引，避免内循环计算
    int64_t out_idx = 0;
    for (int64_t o0 = 0; o0 < new_dims[0]; o0++) {
    for (int64_t o1 = 0; o1 < new_dims[1]; o1++) {
    for (int64_t o2 = 0; o2 < new_dims[2]; o2++) {
    for (int64_t o3 = 0; o3 < new_dims[3]; o3++) {
    for (int64_t o4 = 0; o4 < new_dims[4]; o4++) {
    for (int64_t o5 = 0; o5 < new_dims[5]; o5++) {
    for (int64_t o6 = 0; o6 < new_dims[6]; o6++) {
    for (int64_t o7 = 0; o7 < new_dims[7]; o7++) {
        int64_t i[8];
        i[perm[0]] = o0;
        i[perm[1]] = o1;
        i[perm[2]] = o2;
        i[perm[3]] = o3;
        i[perm[4]] = o4;
        i[perm[5]] = o5;
        i[perm[6]] = o6;
        i[perm[7]] = o7;
        int64_t flat_in = i[0] * strides[0] + i[1] * strides[1] + i[2] * strides[2] +
                          i[3] * strides[3] + i[4] * strides[4] + i[5] * strides[5] +
                          i[6] * strides[6] + i[7] * strides[7];
        out[out_idx++] = in[flat_in];
    }}}}}}}}
}

static rknn3_init_extend make_rknn3_init_extend(const char* tag)
{
    static char selected_device_id[RKNN3_MAX_DEV_LEN] = {0};
    rknn3_init_extend init_extend;
    memset(&init_extend, 0, sizeof(init_extend));

    const char* env_device_id = getenv("RKNN3_DEVICE_ID");
    if (env_device_id && env_device_id[0] != '\0') {
        snprintf(selected_device_id, sizeof(selected_device_id), "%s", env_device_id);
        init_extend.device_id = selected_device_id;
        printf("[%s] use RKNN3_DEVICE_ID=%s\n", tag, selected_device_id);
        return init_extend;
    }

    rknn3_devices devices;
    memset(&devices, 0, sizeof(devices));
    int ret = rknn3_find_devices(&devices);
    if (ret == 0) {
        if (devices.n_devices > 1) {
            printf("[%s] multiple RKNN3 devices found, using device_id=%s\n",
                   tag, devices.devices[0].id);
            snprintf(selected_device_id, sizeof(selected_device_id), "%s", devices.devices[0].id);
            init_extend.device_id = selected_device_id;
        }
    }
    return init_extend;
}

static void dump_tensor_attr(rknn3_tensor_attr* attrs)
{
    std::string shape_str = "";
    for (int j = 0; j < attrs->n_dims; j++) {
      shape_str += std::to_string(attrs->shape[j]);
      if (j < attrs->n_dims - 1) {
        shape_str += ", ";
      }
    }
  
    std::string stride_str = "";
    for (int j = 0; j < attrs->n_stride; j++) {
      stride_str += std::to_string(attrs->stride[j]);
      if (j < attrs->n_stride - 1) {
        stride_str += ", ";
      }
    }
  
    printf("Tensor: name=%s, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%ld, layout=%s, dtype=%s, core_id=%d, "
           "qnt_type=%s\n",
           attrs->name, attrs->n_dims, shape_str.c_str(), stride_str.c_str(), attrs->aligned_size, rknn3_get_layout_string(attrs->layout),
           rknn3_get_type_string(attrs->dtype), attrs->core_id, rknn3_get_qnt_type_string(attrs->qnt_type));
}

#define MODEL_WIDTH 384 // 由于vision模型被裁剪了，需要手动添加这些参数配置
#define MODEL_HEIGHT 384
// TODO 配置


int init_internal_share(caiwei::context::VLMRKNN3Context* context, rknn3_context vlm, rknn3_context llm, uint32_t core_mask_vision, uint32_t core_mask_llm)
{
    int ret = -1;

    uint32_t core_num_vision = 0;
    uint32_t core_num_llm = 0;
    ret = rknn3_query(vlm, RKNN3_QUERY_CORE_NUMBER, &core_num_vision, sizeof(core_num_vision));
    if (ret < 0) {
        printf("rknn3_query failed! ret=%d\n", ret);
        return ret;
    }
    ret = rknn3_query(llm, RKNN3_QUERY_CORE_NUMBER, &core_num_llm, sizeof(core_num_llm));
    if (ret < 0) {
        printf("rknn3_query failed! ret=%d\n", ret);
        return ret;
    }

    uint32_t core_num_vision_ = 0;
    uint32_t core_num_llm_ = 0;
    for (int i = 0; i < 32; i++) {
        if (core_mask_vision & (1 << i))    core_num_vision_++;
    }
    for (int i = 0; i < 32; i++) {
        if (core_mask_llm & (1 << i))    core_num_llm_++;
    }
    if (core_num_vision_ != core_num_vision) {
        printf("the core_mask_vision = %x is not match the core_num_vision = %d!\n", core_mask_vision, core_num_vision);
        return -1;
    }
    if (core_num_llm_ != core_num_llm) {
        printf("the core_mask_llm = %x is not match the core_num_llm = %d!\n", core_mask_llm, core_num_llm);
        return -1;
    }

    rknn3_core_mem_size* core_mem_sizes_vision = (rknn3_core_mem_size*)malloc(sizeof(rknn3_core_mem_size) * core_num_vision);
    if (!core_mem_sizes_vision) {
        printf("Failed to allocate memory for core_mem_sizes_vision\n");
        return ret;
    }
    rknn3_core_mem_size* core_mem_sizes_llm = (rknn3_core_mem_size*)malloc(sizeof(rknn3_core_mem_size) * core_num_llm);
    if (!core_mem_sizes_llm) {
        printf("Failed to allocate memory for core_mem_sizes_llm\n");
        return ret;
    }
    ret = rknn3_query(vlm, RKNN3_QUERY_CORE_MEM_SIZE, core_mem_sizes_vision, sizeof(rknn3_core_mem_size) * core_num_vision);
    if (ret < 0) {
        printf("rknn3_query core memory size failed! ret=%d\n", ret);
        return ret;
    }
    ret = rknn3_query(llm, RKNN3_QUERY_CORE_MEM_SIZE, core_mem_sizes_llm, sizeof(rknn3_core_mem_size) * core_num_llm);
    if (ret < 0) {
        printf("rknn3_query core memory size failed! ret=%d\n", ret);
        return ret;
    }

    std::vector<int> llm_to_vision(core_num_llm);
    for (int i = 0; i < core_num_llm; i++)  { llm_to_vision[i] = -1; }

    int core_num_same = 0;
    for (int i = 0; i < core_num_vision; i++) {
        for (int j = 0; j < core_num_llm; j++) {
            if (core_mem_sizes_vision[i].core_id == core_mem_sizes_llm[j].core_id) {
                uint64_t internal_size = std::max(core_mem_sizes_vision[i].internal_size, core_mem_sizes_llm[j].internal_size);
                core_mem_sizes_vision[i].internal_size = internal_size;
                core_mem_sizes_llm[j].internal_size = internal_size;
                core_num_same ++;
                llm_to_vision[j] = i;
                break;
            }
        }
    }

    context->n_internal_mems = core_num_vision + core_num_llm - core_num_same;
    context->internal_mems.resize(context->n_internal_mems);
    rknn3_tensor_mem** internal_mems_vision = (rknn3_tensor_mem**)calloc(core_num_vision, sizeof(rknn3_tensor_mem*));
    if (!internal_mems_vision) {
        printf("Failed to allocate memory for internal_mems_vision array\n");
        return -1;
    }
    rknn3_tensor_mem** internal_mems_llm = (rknn3_tensor_mem**)calloc(core_num_llm, sizeof(rknn3_tensor_mem*));
    if (!internal_mems_llm) {
        printf("Failed to allocate memory for internal_mems_llm array\n");
        return -1;
    }

    int idx = 0;
    for (uint32_t i = 0; i < core_num_vision; i++) {
        internal_mems_vision[i] = rknn3_create_mem(vlm, core_mem_sizes_vision[i].internal_size, core_mem_sizes_vision[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!internal_mems_vision[i]) {
            return -1;
        }
        context->internal_mems[idx++] = internal_mems_vision[i];
    }
    for (uint32_t i = 0; i < core_num_llm; i++) {
        if (llm_to_vision[i] != -1) {
            internal_mems_llm[i] = internal_mems_vision[llm_to_vision[i]];
            continue;
        }
        internal_mems_llm[i] = rknn3_create_mem(vlm, core_mem_sizes_llm[i].internal_size, core_mem_sizes_llm[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!internal_mems_llm[i]) {
            return -1;
        }
        context->internal_mems[idx++] = internal_mems_llm[i];
    }

    ret = rknn3_set_internal_mem(vlm, internal_mems_vision, core_num_vision);
    if (ret < 0) {
        printf("rknn3_set_internal_mem failed! ret=%d\n", ret);
        return ret;
    }
    ret = rknn3_set_internal_mem(llm, internal_mems_llm, core_num_llm);
    if (ret < 0) {
        printf("rknn3_set_internal_mem failed! ret=%d\n", ret);
        return ret;
    }

    free(internal_mems_vision);
    free(internal_mems_llm);
    free(core_mem_sizes_vision);
    free(core_mem_sizes_llm);

    return ret;
}

bool caiwei::context::VLMRKNN3Context::load_vlm_model() {
    int ret;
    rknn3_config config;
    config.run_core_mask = 0xFF;
    config.user_mem_internal = 1; // 使用用户管理的internal内存
    rknn3_init_extend init_extend = make_rknn3_init_extend("VisionInit");
    ret = rknn3_init(&this->vlm_context, init_extend.device_id ? &init_extend : NULL);
    if (ret < 0)
    {
        printf("rknn_init fail ret=%d\n", ret);
        return ret;
    }
    ret = rknn3_load_model_from_path(this->vlm_context, this->vlm_model_path.c_str(), nullptr);
    if (ret < 0) {
        printf("rknn_load_model failed! ret=%d\n", ret);
        return ret;
    }
    ret = rknn3_model_init(this->vlm_context, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        return ret;
    }
    rknn3_input_output_num io_num;
    ret = rknn3_query(this->vlm_context, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return ret;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    this->input_attrs.resize(io_num.n_input);
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn3_query(this->vlm_context, RKNN3_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    this->output_attrs.resize(io_num.n_output);
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn3_query(this->vlm_context, RKNN3_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    this->deepstack_aligned_size = output_attrs[0].aligned_size;
    printf("deepstack_aligned_size = %d \n", this->deepstack_aligned_size);

    // Set to context
    this->pruned_version_flag = 0;
    this->inputs.resize(io_num.n_input);
    this->outputs.resize(io_num.n_output);
    this->input_size = io_num.n_input;
    this->output_size = io_num.n_output;
    for (int i = 0; i < io_num.n_input; i++) {
        this->inputs[i].mem  = rknn3_create_mem(this->vlm_context, input_attrs[i].aligned_size, input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        this->inputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        memcpy(this->inputs[i].attr, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
    }
    for (int i = 0; i < io_num.n_output; i++) {
        this->outputs[i].mem  = rknn3_create_mem(this->vlm_context, output_attrs[i].aligned_size, output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        this->outputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        memcpy(this->outputs[i].attr, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
    }

    if (input_attrs[0].layout == RKNN3_TENSOR_NCHW)
    {
        printf("model is NCHW input layout\n");
        this->model_channel = input_attrs[0].shape[1];
        this->model_height = input_attrs[0].shape[2];
        this->model_width = input_attrs[0].shape[3];
        printf("input image height=%d, input image width=%d, input image channel=%d\n",
            this->model_height, this->model_width, this->model_channel);
    }
    else if (input_attrs[0].layout == RKNN3_TENSOR_NHWC)
    {
        printf("model is NHWC input layout\n");
        this->model_channel = input_attrs[0].shape[3];
        this->model_height = input_attrs[0].shape[1];
        this->model_width = input_attrs[0].shape[2];
        printf("input image height=%d, input image width=%d, input image channel=%d\n",
            this->model_height, this->model_width, this->model_channel);

    }
    else 
    {   
        int feature_size = MODEL_WIDTH * MODEL_HEIGHT / 16 / 16;
        if(model_width != 0 && model_height != 0) {
            feature_size = model_width * model_height / 16 / 16;
        }
        if(input_attrs[0].shape[0] != feature_size){
            printf("Please note: model_width=%d and model_height=%d do not match the model's expected input dimensions. Please either update the default parameters or explicitly specify the correct model_width and model_height arguments in the command line.\n", model_width == 0 ? MODEL_WIDTH : model_width, model_height == 0 ? MODEL_HEIGHT : model_height);
            return -1;
        }
        this->pruned_version_flag = 1;
        this->model_channel = 3;
        // 由于vision模型被裁剪了，需要手动添加这些参数配置
        if(model_width != 0 && model_height != 0) {
            this->model_width = (int)model_width;
            this->model_height = (int)model_height;
        } else {
            this->model_width = (int)MODEL_WIDTH;
            this->model_height = (int)MODEL_HEIGHT;
        }
        printf("model is UNDEFINED!\n");
        printf("model_width=%d model_height=%d \n", this->model_width,this->model_height);
    }

    if (output_attrs[0].layout == RKNN3_TENSOR_UNDEFINED)
    {
        printf("model is UNDEFINED output layout\n");
        this->embeds_shape = this->outputs[0].attr->shape;
        this->embeds_ndims = this->outputs[0].attr->n_dims;
    }
    else
    {
        printf("model is not UNDEFINED output layout, model output error!\n");
        return false;
    }

    for(int i = 0; i < 3; i++) {
        deepstack_attrs[i].index = 2 + i; //deepstack的index为2、3、 4,可以通过rknn3_query查询所有input_attrs定位到deepstack的index
        ret = rknn3_query(this->context, RKNN3_QUERY_INPUT_ATTR, &(deepstack_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0) {
          printf("rknn_query fail! ret=%d\n", ret);
          return -1;
        }

        this->deepstack_tensor[i].mem = rknn3_create_mem(this->context, this->deepstack_aligned_size, deepstack_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!this->deepstack_tensor[i].mem) {
          printf("fail to create aux_input_tensor.mem!\n");
          return -1;
        }
    }
    printf("--> init internal share\n");
    ret = init_internal_share(this, this->vlm_context, this->context, 0xFF, 0xFF);
    if (ret < 0) {
        printf("qwen3_vl llm/vision internal memeory share fail! ret=%d\n", ret);
        return false;
    }
    return true;
}

bool caiwei::context::VLMRKNN3Context::vlm_run(float16* img_embeds, float16* deepstack_data0, float16* deepstack_data1, float16* deepstack_data2)
{
    int ret;
    float16* expand_data;
    // memset(&dst_img, 0, sizeof(image_buffer_t));

    // // Pre Process
    // dst_img.width     = vision_ctx->model_width;
    // dst_img.height    = vision_ctx->model_height;
    // dst_img.format    = IMAGE_FORMAT_RGB888;
    // dst_img.size      = get_image_size(&dst_img);
    // dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);


    // if (dst_img.virt_addr == NULL)
    // {
    //     printf("malloc buffer size:%d fail!\n", dst_img.size);
    //     goto out;
    // }

    // ret = convert_image(img, &dst_img, NULL, NULL, 0);
    // if (ret < 0)
    // {
    //     printf("convert_image fail! ret=%d\n", ret);
    //     goto out;
    // }

//     if(vision_ctx->pruned_version_flag == 1) {
//         expand_data = (float16*)malloc(dst_img.size * 2 * sizeof(float16)); //expand
//         normalize_and_to_nchw(dst_img.virt_addr, expand_data, vision_ctx->model_width, vision_ctx->model_height);
//         memcpy((char *)expand_data + dst_img.size * 2 , expand_data, dst_img.size * 2); //expand
//         transpose((float16 *)expand_data, (float16 *)(vision_ctx->inputs[0].mem->virt_addr), vision_ctx->model_width, vision_ctx->model_height);
//         free(expand_data);
//     } else {
//         // Set Input Data
//         memcpy(vision_ctx->inputs[0].mem->virt_addr, (uint8_t*)dst_img.virt_addr, dst_img.size);
//     }

//     // Sync Inputs
//     for (int i = 0; i < vision_ctx->io_num.n_input; i++)
//     {
//         ret = rknn3_mem_sync(vision_ctx->rknn_ctx, vision_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
//         if (ret != RKNN3_SUCCESS)
//         {
//             printf("rknn3_mem_sync input[%d] failed! ret=%d\n", i, ret);
//             goto out;
//         }
//     }
    
//     // Run
//     ret = rknn3_run(vision_ctx->rknn_ctx, vision_ctx->inputs, vision_ctx->io_num.n_input, vision_ctx->outputs, vision_ctx->io_num.n_output);
//     if (ret < 0)
//     {
//         printf("rknn_run fail! ret=%d\n", ret);
//         goto out;
//     }

//     // Sync Outputs
//     for (int i = 0; i < vision_ctx->io_num.n_output; i++)
//     {
//         ret = rknn3_mem_sync(vision_ctx->rknn_ctx, vision_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
//         if (ret != RKNN3_SUCCESS)
//         {
//             printf("rknn3_mem_sync output[%d] failed! ret=%d\n", i, ret);
//             goto out;
//         }
//     }

//     // Get Output
//     memcpy((uint8_t*)img_embeds, (uint8_t*)vision_ctx->outputs[0].mem->virt_addr, vision_ctx->outputs[0].mem->size);
//     memcpy((uint8_t*)deepstack_data0, (uint8_t*)vision_ctx->outputs[1].mem->virt_addr, vision_ctx->outputs[1].mem->size);
//     memcpy((uint8_t*)deepstack_data1, (uint8_t*)vision_ctx->outputs[2].mem->virt_addr, vision_ctx->outputs[2].mem->size);
//     memcpy((uint8_t*)deepstack_data2, (uint8_t*)vision_ctx->outputs[3].mem->virt_addr, vision_ctx->outputs[3].mem->size);

// out:
//     if (dst_img.virt_addr != NULL)
//     {
//         free(dst_img.virt_addr);
//     }

//     return ret;
}

#ifdef ENABLE_SPEEDUP
// Qwen3-VL feeds three deepstack tensors in addition to image_embed.
// Keep these tensors aligned with the SpeedUP output layout.
static bool update_qwen3_vl_deepstack(rknn_app_context_t* app_ctx,
                                       SpeedUPHandle handle,
                                       int n_images,
                                       int tokens_per_image)
{
    int output_count_per_input = speedup_get_output_count_per_input(handle);
    if (output_count_per_input <= 0 || output_count_per_input >= tokens_per_image) {
        return true;
    }

    for (int ds_idx = 0; ds_idx < 3; ds_idx++) {
        if (!app_ctx->llm.deepstack_tensor[ds_idx].mem) {
            return false;
        }

        float16* ds_base = (float16*)app_ctx->llm.deepstack_tensor[ds_idx].mem->virt_addr;
        size_t ds_size_per_image = app_ctx->vision.outputs[ds_idx + 1].mem->size / std::max(1, n_images);
        int ds_total_elems = (int)(ds_size_per_image / sizeof(float16));
        int ds_dim = ds_total_elems / tokens_per_image;
        if (ds_dim <= 0) continue;

        int global_write_idx = 0;
        for (int img_idx = 0; img_idx < n_images; img_idx++) {
            const int* keep_mask = speedup_get_keep_mask_for_image(handle, img_idx);
            if (!keep_mask) keep_mask = speedup_get_keep_mask(handle);
            if (!keep_mask) return false;
            float16* img_src = ds_base + img_idx * ds_total_elems;
            for (int tok = 0; tok < tokens_per_image; tok++) {
                if (keep_mask[tok]) {
                    memmove(&ds_base[global_write_idx * ds_dim],
                            &img_src[tok * ds_dim],
                            ds_dim * sizeof(float16));
                    global_write_idx++;
                }
            }
        }
    }
    return true;
}

#endif

std::vector<rknn3_llm_input> caiwei::context::VLMRKNN3Context::get_inputs(rknn3_session* session, const caiwei::text::CompletionsRequest& request) {
    // LLM Input
    rknn3_llm_multimodal_tensor tensor;
    tensor.name = "input_embeds";
    // Add image start tags to the prompt
    float16* img_embeds = NULL; // TODO 全局分配内存
    size_t embed_elems = 1;
    for (size_t i = 0; i < this->embeds_ndims; i++)
    {
      embed_elems *= this->embeds_shape[i];
    }
    img_embeds = (float16*)malloc((embed_elems) * sizeof(float16));
    std::string prompt_with_image;
    prompt_with_image = "<image> " + std::string("prompt");
    tensor.prompt = (prompt_with_image).c_str();
    tensor.image.image_embed = img_embeds;
    if(this->embeds_ndims == 2) {
        tensor.image.n_image_tokens = this->embeds_shape[0];
        tensor.image.n_image        = 1;
    } else {
        tensor.image.n_image_tokens = this->embeds_shape[1];
        tensor.image.n_image        = this->embeds_shape[0];
    }
    tensor.image.image_width   = this->model_width;
    tensor.image.image_height  = this->model_height;
    tensor.image.image_start   = "<|vision_start|>";
    tensor.image.image_end     = "<|vision_end|>";
    tensor.image.image_content = "<|image_pad|>";
    tensor.enable_thinking     = false;

    int ret;

    printf("--> inference qwen3_vl vision model\n");
    // int64_t start_us = getCurrentTimeUs();
    ret = this->vlm_run(img_embeds,
        (float16*)this->deepstack_tensor[0].mem->virt_addr,
        (float16*)this->deepstack_tensor[1].mem->virt_addr,
        (float16*)this->deepstack_tensor[2].mem->virt_addr);
    // perf->vision_latency = getCurrentTimeUs() - start_us;
    if (ret != 0) {
        printf("inference qwen3_vl vision model fail! ret=%d\n", ret);
        return {};
    }

    printf("--> inference qwen3_vl llm model\n");

#ifdef ENABLE_SPEEDUP
    if (!this->speedup) {
        printf("[SpeedUP] handle is NULL, run normal inference without SpeedUP\n");
    } else {
        int n_images = tensor.image.n_image > 0 ? (int)tensor.image.n_image : 1;
        int tokens_per_image = (int)tensor.image.n_image_tokens;
        int output_count_per_input = tokens_per_image;
        ret = speedup_prepare_rknn_multimodal_tensor(this->speedup,
                                                          img_embeds,
                                                          this->embeds_shape,
                                                          this->embeds_ndims,
                                                          &tensor,
                                                          this->model_width,
                                                          this->model_height,
                                                          speedup_ratio,
                                                          session,
                                                          &output_count_per_input);
        if (ret != 0) {
            printf("[SpeedUP] prepare failed! ret=%d\n", ret);
            return ret;
        }
        if (!update_qwen3_vl_deepstack(app_ctx, app_ctx->speedup,
                                        n_images, tokens_per_image)) {
            printf("[SpeedUP] deepstack update failed\n");
            return -1;
        }
    }
#endif

    std::vector<rknn3_llm_input> inputs(4);
    inputs[0].input_type = RKNN3_LLM_INPUT_MULTIMODAL;
    inputs[0].multimodal_input = tensor;
    for (int i = 0; i < 3; ++i)
    {
        inputs[i + 1].input_type = RKNN3_LLM_INPUT_AUX;
        inputs[i + 1].aux_input  = this->deepstack_tensor[i];
    }
    return inputs;
}


std::vector<rknn3_llm_input> caiwei::context::RerankingRKNN3Context::get_inputs(rknn3_session* session, const caiwei::text::CompletionsRequest& request) {
    rknn3_llm_tensor tensor;
    // LLM Input
    tensor.name = "input";
    tensor.embed = NULL;
    tensor.tokens = NULL;
    tensor.n_tokens = 0;
    tensor.enable_thinking = false;
    std::vector<rknn3_llm_input> inputs(1);
    inputs[0].input_type = RKNN3_LLM_INPUT_EMBED;
    inputs[0].llm_input  = tensor;
    return inputs;
}