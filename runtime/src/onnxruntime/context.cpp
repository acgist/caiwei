#include "caiwei/runtime/onnxruntime.hpp"

OrtLoggingLevel caiwei::context::onnxruntime_log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO;

caiwei::context::ONNXRuntimeContext::ONNXRuntimeContext(std::string path, const Ort::Env* env) : path(std::move(path)) {
    CW_LOG_I("创建ONNXRuntimeContext: %s", this->path.c_str());
    Ort::SessionOptions options;
    // options.DisableCpuMemArena();
    #ifdef ENABLE_CAIWEI_BACKEND_CUDA
    // options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    // options.SetLogSeverityLevel(static_cast<int>(caiwei::context::onnxruntime_log_level));
    // options.SetIntraOpNumThreads(1);
    // options.SetInterOpNumThreads(1);
    // options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    OrtCUDAProviderOptions cudaOptions;
    cudaOptions.device_id = caiwei::env::get_int("CAIWEI_CUDA_ID");
    options.AppendExecutionProvider_CUDA(cudaOptions);
    CW_LOG_I("ONNXRuntimeContext使用CUDA推理: %d", cudaOptions.device_id);
    #else
    options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    options.SetLogSeverityLevel(static_cast<int>(caiwei::context::onnxruntime_log_level));
    options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    options.SetInterOpNumThreads(std::thread::hardware_concurrency());
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    CW_LOG_I("ONNXRuntimeContext使用CPU推理");
    #endif
    #if CAIWEI_OS_WIN
    std::wstring wPath(this->path.begin(), this->path.end());
    this->session = new Ort::Session(*env, wPath.c_str(), options);
    #else
    this->session = new Ort::Session(*env, this->path.c_str(), options);
    #endif
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t inputNodeCount  = this->session->GetInputCount();
    const size_t outputNodeCount = this->session->GetOutputCount();
    for(size_t index = 0; index < inputNodeCount; ++index) {
        const Ort::AllocatedStringPtr name = this->session->GetInputNameAllocated(index, allocator);
        // TODO 长度
        char* node_name = new char[64];
        std::strcpy(node_name, name.get());
        this->input_node_names.push_back(node_name);
        CW_LOG_D("ONNXRuntimeContext输入节点: %" PRId64 " = %s", index, node_name);
    }
    for(size_t index = 0; index < outputNodeCount; ++ index) {
        const Ort::AllocatedStringPtr name = this->session->GetOutputNameAllocated(index, allocator);
        // TODO 长度
        char* node_name = new char[64];
        std::strcpy(node_name, name.get());
        this->output_node_names.push_back(node_name);
        CW_LOG_D("ONNXRuntimeContext输出节点: %" PRId64 " = %s", index, node_name);
    }
    this->run_options = new Ort::RunOptions(nullptr);
}

caiwei::context::ONNXRuntimeContext::~ONNXRuntimeContext() {
    CW_LOG_D("释放ONNXRuntimeContext: %s", this->path.c_str());
    if(this->run_options) {
        CW_LOG_D("释放ONNXRuntimeContext run_options");
        delete this->run_options;
        this->run_options = nullptr;
    }
    if(this->session) {
        CW_LOG_D("释放ONNXRuntimeContext session");
        delete this->session;
        this->session = nullptr;
    }
    for(auto ptr : this->input_node_names) {
        delete[] ptr;
    }
    this->input_node_names.clear();
    for(auto ptr : this->output_node_names) {
        delete[] ptr;
    }
    this->output_node_names.clear();
}

Ort::Value caiwei::context::ONNXRuntimeContext::run(float* blob, std::vector<int64_t>& out_dims, int batch) {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->input_node_dims[0] = batch;
    #ifdef ENABLE_CAIWEI_BACKEND_CUDA
    Ort::IoBinding io_binding(*this->session);
    auto memory_info = Ort::MemoryInfo("CudaPinned", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    const Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memory_info,
        blob,
        this->input_data_length * batch,
        this->input_node_dims.data(),
        this->input_node_dims.size()
    );
    io_binding.BindInput(this->input_node_names[0], inputTensor);
    Ort::MemoryInfo output_memory_info{"CudaPinned", OrtDeviceAllocator, 0, OrtMemTypeDefault};
    io_binding.BindOutput(this->output_node_names[0], output_memory_info);
    this->session->Run(
        *this->run_options,
        io_binding
    );
    auto outputTensor = io_binding.GetOutputValues();
    #else
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    const Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memory_info,
        blob,
        this->input_data_length * batch,
        this->input_node_dims.data(),
        this->input_node_dims.size()
    );
    auto outputTensor = this->session->Run(
        *this->run_options,
        this->input_node_names.data(),
        &inputTensor,
        this->input_node_names.size(),
        this->output_node_names.data(),
        this->output_node_names.size()
    );
    #endif
    auto& output = outputTensor.front();
    out_dims = output.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    return std::move(output);
}

std::shared_ptr<caiwei::context::ClsContext> caiwei::context::get_cls_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) {
    int w = caiwei::env::get_int("CAIWEI_CLS_W");
    int h = caiwei::env::get_int("CAIWEI_CLS_H");
    int top_k = caiwei::env::get_int("CAIWEI_CLS_TOP_K");
    int class_size = caiwei::env::get_int("CAIWEI_CLS_CLASS_SIZE");
    float confidence_threshold = caiwei::env::get_float("CAIWEI_CLS_CONFIDENCE_THRESHOLD");
    if (!std::filesystem::exists(info->path)) {
        CW_LOG_W("ClsContext模型无效: %s", info->path.c_str());
        return nullptr;
    }
    return std::make_shared<ClsONNXRuntimeContext>(info->path, w, h, top_k, class_size, confidence_threshold, runtime);
}

std::shared_ptr<caiwei::context::DetContext> caiwei::context::get_det_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) {
    int w = caiwei::env::get_int("CAIWEI_DET_W");
    int h = caiwei::env::get_int("CAIWEI_DET_H");
    int class_size = caiwei::env::get_int("CAIWEI_DET_CLASS_SIZE");
    float iou_threshold = caiwei::env::get_float("CAIWEI_DET_IOU_THRESHOLD");
    float confidence_threshold = caiwei::env::get_float("CAIWEI_DET_CONFIDENCE_THRESHOLD");
    if (!std::filesystem::exists(info->path)) {
        CW_LOG_W("DetContext模型无效: %s", info->path.c_str());
        return nullptr;
    }
    return std::make_shared<DetONNXRuntimeContext>(info->path, w, h, class_size, iou_threshold, confidence_threshold, runtime);
}

std::shared_ptr<caiwei::context::OBBContext> caiwei::context::get_obb_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::SegContext> caiwei::context::get_seg_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::SemContext> caiwei::context::get_sem_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::PoseContext> caiwei::context::get_pose_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::DepthContext> caiwei::context::get_depth_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::ASRContext> caiwei::context::get_asr_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::TTSContext> caiwei::context::get_tts_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::LLMContext> caiwei::context::get_llm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::VLMContext> caiwei::context::get_vlm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::MLLMContext> caiwei::context::get_mllm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::EmbeddingContext> caiwei::context::get_embedding_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::RerankingContext> caiwei::context::get_reranking_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::VLEmbeddingContext> caiwei::context::get_vl_embedding_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::VLRerankingContext> caiwei::context::get_vl_reranking_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::ONNXRuntimeRuntime> runtime) {
    return nullptr;
}
