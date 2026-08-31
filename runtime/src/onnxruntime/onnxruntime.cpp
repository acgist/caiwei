#include "onnxruntime.hpp"

OrtLoggingLevel caiwei::context::onnxruntime_log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO;

caiwei::context::ONNXRuntimeContext::ONNXRuntimeContext(std::string path, const Ort::Env* env) : path(path) {
    CW_LOG_I("创建ONNXRuntime: %s", path.c_str());
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
    CW_LOG_I("ONNXRuntime使用CUDA推理: %d", cudaOptions.device_id);
    #else
    options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    options.SetLogSeverityLevel(static_cast<int>(caiwei::context::onnxruntime_log_level));
    options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    options.SetInterOpNumThreads(std::thread::hardware_concurrency());
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    CW_LOG_I("ONNXRuntime使用CPU推理");
    #endif
    #if OS_WIN
    std::wstring wPath(path.begin(), path.end());
    this->session = new Ort::Session(*env, wPath.c_str(), options);
    #else
    this->session = new Ort::Session(*env, path.c_str(), options);
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
        CW_LOG_D("ONNXRuntime输入节点: %" PRId64 " = %s", index, node_name);
    }
    for(size_t index = 0; index < outputNodeCount; ++ index) {
        const Ort::AllocatedStringPtr name = this->session->GetOutputNameAllocated(index, allocator);
        // TODO 长度
        char* node_name = new char[64];
        std::strcpy(node_name, name.get());
        this->output_node_names.push_back(node_name);
        CW_LOG_D("ONNXRuntime输出节点: %" PRId64 " = %s", index, node_name);
    }
    this->run_options = new Ort::RunOptions(nullptr);
}

caiwei::context::ONNXRuntimeContext::~ONNXRuntimeContext() {
    CW_LOG_D("释放ONNXRuntime: %s", this->path.c_str());
    if(this->run_options) {
        CW_LOG_D("释放ONNXRuntime run_options");
        delete this->run_options;
        this->run_options = nullptr;
    }
    if(this->session) {
        CW_LOG_D("释放ONNXRuntime session");
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
