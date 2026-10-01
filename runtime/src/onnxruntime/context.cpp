#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/type.hpp"
#include "caiwei/image_tool.hpp"
#include "caiwei/runtime/onnxruntime.hpp"

#if CAIWEI_DEBUG
OrtLoggingLevel caiwei::context::onnxruntime_log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO;
#else
OrtLoggingLevel caiwei::context::onnxruntime_log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
#endif

static void print_tensor_info(const char* title, size_t index, const char* name, const Ort::ShapeInferContext::Ints& shape);

caiwei::context::ONNXRuntimeContext::ONNXRuntimeContext(std::string path, int c, int h, int w, const Ort::Env* env)
  : path(std::move(path)), env(env) {
    this->input_node_dims.push_back({ 1, c, h, w });
    CW_LOG_I("创建ONNXRuntimeContext: %s", this->path.c_str());
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
        // TODO 释放崩溃
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

bool caiwei::context::ONNXRuntimeContext::load_model() {
    Ort::SessionOptions options;
    // options.DisableCpuMemArena();
    #ifdef ENABLE_CAIWEI_BACKEND_CUDA
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetLogSeverityLevel(static_cast<int>(caiwei::context::onnxruntime_log_level));
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::CUDAProviderOptions cudaOptions;
    std::unordered_map<std::string, std::string> opts = {
        {"device_id",     "0"         },
        {"gpu_mem_limit", "1073741824"},
    };
    cudaOptions.Update(opts);
//  cudaOptions.device_id     = caiwei::env::get_int("CAIWEI_CUDA_ID");
//  cudaOptions.gpu_mem_limit = caiwei::env::get_int("CAIWEI_CUDA_LIMIT");
    options.AppendExecutionProvider_CUDA_V2(*cudaOptions);
    CW_LOG_I("ONNXRuntimeContext使用CUDA推理: %d", caiwei::env::get_int("CAIWEI_CUDA_ID"));
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
    this->input_node_dims.resize(inputNodeCount);
    for(size_t index = 0; index < inputNodeCount; ++index) {
        const Ort::AllocatedStringPtr name = this->session->GetInputNameAllocated(index, allocator);
        // TODO
        // this->session->GetInputNames();
        // TODO 长度
        char* node_name = new char[64];
        std::strcpy(node_name, name.get());
        this->input_node_names.push_back(node_name);
        auto info = this->session->GetInputTypeInfo(index).GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        // TODO GPU shape - 0
        if (shape.size() == 0) {
            shape.assign(this->input_node_dims[index].begin(), this->input_node_dims[index].end());
        } else {
            this->input_node_dims[index].clear();
            for (auto dim : shape) {
                this->input_node_dims[index].push_back(dim);
            }
        }
        print_tensor_info("ONNXRuntimeContext输入节点", index, node_name, shape);
    }
    for(size_t index = 0; index < outputNodeCount; ++ index) {
        const Ort::AllocatedStringPtr name = this->session->GetOutputNameAllocated(index, allocator);
        // TODO 长度
        char* node_name = new char[64];
        std::strcpy(node_name, name.get());
        this->output_node_names.push_back(node_name);
        auto info = this->session->GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        // TODO GPU shape - 0
        print_tensor_info("ONNXRuntimeContext输出节点", index, node_name, shape);
    }
    this->run_options = new Ort::RunOptions(nullptr);
    this->memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    return true;
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
    return this->run(this->chw.data(), this->chw.size());
}

std::vector<Ort::Value> caiwei::context::ONNXRuntimeContext::run(float* blob, size_t size, int batch) {
    #ifdef ENABLE_CAIWEI_BACKEND_CUDA
    Ort::IoBinding io_binding(*this->session);
    std::vector<Ort::Value> input_tensors;
    for (int i = 0; i < this->input_node_names.size(); ++i) {
        this->input_node_dims[i][0] = batch;
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            blob,
            size,
            this->input_node_dims[i].data(),
            this->input_node_dims[i].size()
        );
        input_tensors.push_back(std::move(input_tensor));
        io_binding.BindInput(this->input_node_names[i], input_tensors[i]);
    }
    for (int i = 0; i < this->output_node_names.size(); ++i) {
        io_binding.BindOutput(this->output_node_names[i], memory_info);
    }
    this->session->Run(*this->run_options, io_binding);
    auto output_tensor = io_binding.GetOutputValues();
    #else
    std::vector<Ort::Value> input_tensors;
    for (int i = 0; i < this->input_node_names.size(); ++i) {
        this->input_node_dims[i][0] = batch;
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            blob,
            size,
            this->input_node_dims[i].data(),
            this->input_node_dims[i].size()
        );
        input_tensors.push_back(std::move(input_tensor));
    }
    auto output_tensor = this->session->Run(
        *this->run_options,
        this->input_node_names.data(),
        input_tensors.data(),
        this->input_node_names.size(),
        this->output_node_names.data(),
        this->output_node_names.size()
    );
    #endif
    std::vector<Ort::Value> ret;
    ret.reserve(output_tensor.size());
    for (auto iter = output_tensor.begin(); iter != output_tensor.end(); ++iter) {
        ret.push_back(std::move(*iter));
    }
    return ret;
}

static void print_tensor_info(const char* title, size_t index, const char* name, const Ort::ShapeInferContext::Ints& shape) {
    std::string shape_info = "(";
    if (shape.size() > 0) {
        shape_info += std::to_string(shape[0]);
        for (int i = 1; i < shape.size(); ++i) {
            shape_info += ", " + std::to_string(shape[i]);
        }
    } else {
        shape_info += "0";
    }
    shape_info += ")";
    CW_LOG_I("%s: %" PRId64 " = %s shape: %s", title, index, name, shape_info.c_str());
}
