#include <thread>
#include <unordered_map>

#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/type.hpp"
#include "caiwei/image_tool.hpp"
#include "caiwei/runtime/onnxruntime.hpp"

#if CAIWEI_DEBUG
OrtLoggingLevel caiwei::context::onnxruntime_log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE;
#else
OrtLoggingLevel caiwei::context::onnxruntime_log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO;
#endif

static void print_tensor_info(const char* title, const size_t index, const char* name, const std::vector<int64_t>& shape);

caiwei::context::ONNXRuntimeContext::ONNXRuntimeContext(
    std::string path,
    int c,
    int h,
    int w,
    const Ort::Env* env
) : path(std::move(path)), env(env) {
    CW_LOG_I("创建ONNXRuntimeContext: %s", this->path.c_str());
    this->input_node_dims.push_back({ 1, c, h, w });
}

caiwei::context::ONNXRuntimeContext::~ONNXRuntimeContext() {
    CW_LOG_D("释放ONNXRuntimeContext: %s", this->path.c_str());
    this->release();
}

void caiwei::context::ONNXRuntimeContext::release() {
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
    this->input_node_names.clear();
    this->output_node_names.clear();
}

bool caiwei::context::ONNXRuntimeContext::load_model() {
    if (this->env == nullptr) {
        CW_LOG_W("没有ONNXRuntime环境");
        return false;
    }
    this->release();
    Ort::SessionOptions options;
    options.SetLogSeverityLevel(static_cast<int>(caiwei::context::onnxruntime_log_level));
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    #if ENABLE_CAIWEI_BACKEND_CUDA
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    Ort::CUDAProviderOptions cudaOptions;
    const std::string cuda_id  = caiwei::env::get("CAIWEI_CUDA_ID");
    const std::string cuda_mem = std::to_string(1024ULL * 1024 * caiwei::env::get_int("CAIWEI_CUDA_MEM"));
    std::unordered_map<std::string, std::string> opts = {
        {"device_id",     cuda_id  },
        {"gpu_mem_limit", cuda_mem },
    };
    cudaOptions.Update(opts);
    options.AppendExecutionProvider_CUDA_V2(*cudaOptions);
    #else
    options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    options.SetIntraOpNumThreads(std::thread::hardware_concurrency() / 2);
    options.SetInterOpNumThreads(1);
    #endif
    #if CAIWEI_OS_WIN
    std::wstring path(this->path.begin(), this->path.end());
    this->session = new Ort::Session(*env, path.c_str(), options);
    #else
    this->session = new Ort::Session(*env, this->path.c_str(), options);
    #endif
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t input_node_count  = this->session->GetInputCount();
    const size_t output_node_count = this->session->GetOutputCount();
    if (input_node_count <= 0 || output_node_count <= 0) {
        CW_LOG_W("模型输入输出错误: %zu - %zu", input_node_count, output_node_count);
        this->release();
        return false;
    }
    this->input_node_dims.resize(input_node_count);
    for(size_t index = 0; index < input_node_count; ++index) {
        Ort::AllocatedStringPtr name = this->session->GetInputNameAllocated(index, allocator);
        auto info = this->session->GetInputTypeInfo(index).GetTensorTypeAndShapeInfo();
        #if ENABLE_CAIWEI_BACKEND_CUDA
        #else
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            CW_LOG_W("输入节点数据类型错误: %d", info.GetElementType());
            this->release();
            return false;
        }
        #endif
        auto& node_dims = this->input_node_dims[index];
        auto shape = info.GetShape();
        if (shape.size() == 0) {
            shape.assign(node_dims.begin(), node_dims.end());
        } else if (shape.size() == node_dims.size()) {
            for (size_t i = 0; i < shape.size(); ++i) {
                if (shape[i] != -1 && shape[i] != node_dims[i]) {
                    CW_LOG_W("输入节点维度错误: %" PRId64 " != %" PRId64, node_dims[i], shape[i]);
                    this->release();
                    return false;
                }
            }
        } else {
            CW_LOG_W("输入节点维度错误: %zu != %zu", shape.size(), node_dims.size());
            this->release();
            return false;
        }
        print_tensor_info("输入节点", index, name.get(), shape);
        this->input_node_names.push_back(std::move(name));
    }
    for(size_t index = 0; index < output_node_count; ++ index) {
        Ort::AllocatedStringPtr name = this->session->GetOutputNameAllocated(index, allocator);
        auto info = this->session->GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo();
        #if ENABLE_CAIWEI_BACKEND_CUDA
        #else
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            CW_LOG_W("输出节点数据类型错误: %d", info.GetElementType());
            this->release();
            return false;
        }
        #endif
        auto shape = info.GetShape();
        print_tensor_info("输出节点", index, name.get(), shape);
        this->output_node_names.push_back(std::move(name));
    }
    this->run_options = new Ort::RunOptions(nullptr);
    this->memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    return true;
}

std::vector<Ort::Value> caiwei::context::ONNXRuntimeContext::run(const int h, const int w, const caiwei::media::ImageFrame& image) {
    if (this->src_w != image.width || this->src_h != image.height || this->src_c != image.channels) {
        auto& node_dims = this->input_node_dims[0];
        if (node_dims[1] != image.channels || node_dims[2] != h || node_dims[3] != w) {
            CW_LOG_W("图片大小和模型输入大小不匹配");
            return {};
        }
        if (!caiwei::image::resize(image.width, image.height, w, h, this->dst_w, this->dst_h, this->pad_w, this->pad_h, this->scale)) {
            return {};
        }
        this->src_w = image.width;
        this->src_h = image.height;
        this->src_c = image.channels;
        this->dst.resize(static_cast<size_t>(this->dst_w) * this->dst_h * image.channels);
        this->pad.resize(static_cast<size_t>(          w) *           h * image.channels, caiwei::image::DEFAULT_PADDING);
        this->hwc.resize(static_cast<size_t>(          w) *           h * image.channels);
        this->chw.resize(static_cast<size_t>(          w) *           h * image.channels);
    }
    if (!caiwei::image::resize(image.data.data(), this->dst.data(), image.width, image.height, this->dst_w, this->dst_h, image.channels)) {
        return {};
    }
    caiwei::image::padding(this->dst.data(), this->pad.data(), this->dst_w, this->dst_h, this->pad_w, this->pad_h, w, h, image.channels);
    caiwei::image::hwc_to_chw_u8_f32(this->pad.data(), this->chw.data(), h, w, image.channels, 255.0F);
    return this->run(this->chw.data(), this->chw.size());
}

std::vector<Ort::Value> caiwei::context::ONNXRuntimeContext::run(float* blob, const size_t size, const int batch) {
    Ort::IoBinding io_binding(*this->session);
    std::vector<Ort::Value> input_tensors;
    for (size_t i = 0; i < this->input_node_names.size(); ++i) {
        this->input_node_dims[i][0] = batch;
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            blob,
            size,
            this->input_node_dims[i].data(),
            this->input_node_dims[i].size()
        );
        input_tensors.push_back(std::move(input_tensor));
        io_binding.BindInput(this->input_node_names[i].get(), input_tensors[i]);
    }
    for (size_t i = 0; i < this->output_node_names.size(); ++i) {
        io_binding.BindOutput(this->output_node_names[i].get(), memory_info);
    }
    this->session->Run(*this->run_options, io_binding);
    return io_binding.GetOutputValues();
}

static void print_tensor_info(const char* title, const size_t index, const char* name, const std::vector<int64_t>& shape) {
    std::string shape_info = "(";
    if (shape.empty()) {
        shape_info += "-";
    } else {
        shape_info += std::to_string(shape[0]);
        for (size_t i = 1; i < shape.size(); ++i) {
            shape_info += ", " + std::to_string(shape[i]);
        }
    }
    shape_info += ")";
    CW_LOG_I("%s: %zu = %s shape: %s", title, index, name, shape_info.c_str());
}
