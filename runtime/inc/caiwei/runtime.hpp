/**
 * 运行环境
 * 
 * session -> context -> runtime
 */
#ifndef CAIWEI_RUNTIME_RUNTIME_HPP
#define CAIWEI_RUNTIME_RUNTIME_HPP

#include <atomic>
#include <memory>

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN
#ifdef __arm__
typedef uint32_t rknn_context;
#else
typedef uint64_t rknn_context;
#endif
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
namespace Ort {

struct Env;

};
#endif

namespace caiwei  {
namespace runtime {

enum class Type {
    CANN,
    RKNN2,
    RKNN3,
    LLAMACPP,
    ONNXRUNTIME,
    NONE,
};

class Runtime {
private:
    caiwei::runtime::Type type;
    std::atomic_int32_t ref_count = 0;
public:
    Runtime(caiwei::runtime::Type type);
    virtual ~Runtime();
public:
    uint32_t ref();
    uint32_t unref();
};

#ifdef ENABLE_CAIWEI_RUNTIME_CANN
class CANNRuntime : public Runtime {
public:
    CANNRuntime();
    ~CANNRuntime();
};
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_RKNN
class RKNN2Runtime : public Runtime {
public:
    RKNN2Runtime();
    ~RKNN2Runtime();
};

class RKNN3Runtime : public Runtime {
public:
    RKNN3Runtime();
    ~RKNN3Runtime();
};
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_LLAMACPP
class LlamaCPPRuntime : public Runtime {
public:
    LlamaCPPRuntime();
    ~LlamaCPPRuntime();
};
#endif

#ifdef ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME
class ONNXRuntimeRuntime : public Runtime {
public:
    Ort::Env* env = nullptr;
public:
    ONNXRuntimeRuntime();
    ~ONNXRuntimeRuntime();
};
#endif

void init();
void stop();

template <typename R>
std::shared_ptr<R> get_runtime(caiwei::runtime::Type type);

} // namespace runtime
} // namespace caiwei

#endif // CAIWEI_RUNTIME_RUNTIME_HPP
