#include "caiwei/runtime/onnxruntime.hpp"

#include "caiwei/type.hpp"
#include "caiwei/image_tool.hpp"

#if CAIWEI_DEBUG
OrtLoggingLevel caiwei::context::onnxruntime_log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO;
#else
OrtLoggingLevel caiwei::context::onnxruntime_log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
#endif

caiwei::context::ONNXRuntimeContext::ONNXRuntimeContext(std::string path, int c, int h, int w, const Ort::Env* env) : path(std::move(path)), input_data_length(c * h * w) {
    this->input_node_dims.push_back(1);
    this->input_node_dims.push_back(c);
    this->input_node_dims.push_back(h);
    this->input_node_dims.push_back(w);
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
        CW_LOG_I("ONNXRuntimeContext输入节点: %" PRId64 " = %s", index, node_name);
        auto info = this->session->GetInputTypeInfo(index).GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        for (auto dim : shape) {
            CW_LOG_I("ONNXRuntimeContext输入节点维度: %" PRId64, dim);
        }
    }
    for(size_t index = 0; index < outputNodeCount; ++ index) {
        const Ort::AllocatedStringPtr name = this->session->GetOutputNameAllocated(index, allocator);
        // TODO 长度
        char* node_name = new char[64];
        std::strcpy(node_name, name.get());
        this->output_node_names.push_back(node_name);
        CW_LOG_I("ONNXRuntimeContext输出节点: %" PRId64 " = %s", index, node_name);
        auto info = this->session->GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        for (auto dim : shape) {
            CW_LOG_I("ONNXRuntimeContext输出节点维度: %" PRId64, dim);
        }
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

std::vector<Ort::Value> caiwei::context::ONNXRuntimeContext::run(int h, int w, const caiwei::media::ImageFrame& image) {
    if (this->image_width != image.width || this->image_height != image.height) {
        this->image_width  = image.width;
        this->image_height = image.height;
        caiwei::image::resize(image.width, image.height, w, h, this->dst_w, this->dst_h, this->pad_w, this->pad_h, this->scale);
        this->dst.resize(this->dst_w * this->dst_h * image.channels);
        this->pad.resize(          w *           h * image.channels, caiwei::image::DEFAULT_PADDING);
        this->hwc.resize(          w *           h * image.channels);
        this->chw.resize(          w *           h * image.channels);
    }
    caiwei::image::resize(image.data.data(), this->dst.data(), image.width, image.height, this->dst_w, this->dst_h);
    caiwei::image::padding(this->dst.data(), this->pad.data(), this->dst_w, this->dst_h, this->pad_w, this->pad_h, w, h);
    caiwei::type::i8_to_f32(this->pad.data(), w * h * image.channels, this->hwc.data(), 255.0F);
    caiwei::image::hwc_to_chw(this->hwc.data(), this->chw.data(), h, w, image.channels);
    return this->run(this->chw.data());
}

std::vector<Ort::Value> caiwei::context::ONNXRuntimeContext::run(float* blob, int batch) {
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
    std::vector<Ort::Value> ret;
    ret.reserve(outputTensor.size());
    for (auto iter = outputTensor.begin(); iter != outputTensor.end(); ++iter) {
        auto& out = *iter;
        ret.push_back(std::move(out));
    }
    return ret;
}
