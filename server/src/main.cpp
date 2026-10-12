#include "caiwei/env.hpp"
#ifdef ENABLE_CAIWEI_REST
#include "caiwei/rest.hpp"
#endif
#include "caiwei/caiwei.hpp"

#if CAIWEI_OS_WIN
#include <cstdlib>
#endif
#include <cstring>

int main(int argc, char* argv[]) {
#if CAIWEI_OS_WIN
    system("chcp 65001");
#if CAIWEI_DEBUG
    caiwei::env::set("CAIWEI_CONTEXT_INFO", R"(
CLS,YOLO,yolo26n-cls,yolo26n-cls.onnx
DET,YOLO,yolo26n-det,yolo26n-det.onnx
SEG,YOLO,yolo26n-seg,yolo26n-seg.onnx
POSE,YOLO,yolo26n-pose,yolo26n-pose.onnx
ASR,QWEN,qwen3-asr,D:/tmp/model/llama.cpp/Qwen3-ASR-0.6B-Q4_K_M.gguf|D:/tmp/model/llama.cpp/mmproj-Qwen3-ASR-0.6b-Q4_K_M.gguf
LLM,QWEN,qwen3-llm,D:/tmp/model/llama.cpp/Qwen3-0.6B-Q8_0.gguf
VLM,QWEN,qwen3-vlm,D:/tmp/model/llama.cpp/Qwen3-VL-2B-Instruct-Q4_K_M.gguf|D:/tmp/model/llama.cpp/mmproj-Qwen3-VL-2B-Instruct-Q4_K_M.gguf
EMBEDDING,QWEN,qwen3-embedding,D:/tmp/model/llama.cpp/Qwen3-Embedding-0.6B-Q4_K_M.gguf
RERANKING,QWEN,qwen3-reranking,D:/tmp/model/llama.cpp/Qwen3-Reranker-0.6B-Q4_K_M.gguf
    )");
#endif
#endif
    if (argc >= 2 && std::strcmp(argv[1], "-h") == 0) {
        caiwei::env::print_all_env();
        return 0;
    }
    caiwei::init();
#ifdef ENABLE_CAIWEI_REST
    caiwei::rest::open();
#endif
    caiwei::stop();
    return 0;
}
