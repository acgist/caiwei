#include "test.hpp"

#include "caiwei/session.hpp"

[[maybe_unused]]
void test_completions_sync() {
    caiwei::text::CompletionsRequest request;
    request.model = "qwen3-llm";
    request.stream = false;
    request.enable_thinking = false;
    request.messages.push_back({.role = "user", .content = "1+1=?"});
    caiwei::session::CompletionsSession session(request);
    CW_LOG_I("completions: %s", session.get_sync().c_str());
}

[[maybe_unused]]
void test_completions_async() {
    caiwei::text::CompletionsRequest request;
    request.model = "qwen3-llm";
    request.stream = false;
    request.enable_thinking = false;
    request.messages.push_back({.role = "user", .content = "1+1=?"});
    caiwei::session::CompletionsSession session(request, [](const char* data, int /* length */) {
        CW_LOG_I("completions: %s", data);
        return true;
    });
    session.get();
}

[[maybe_unused]]
void test_embeddings() {
    caiwei::text::EmbeddingsRequest request;
    request.model = "qwen3-embedding";
    caiwei::session::EmbeddingsSession session(request);
    CW_LOG_I("embeddings: %s", session.get().c_str());
}

[[maybe_unused]]
void test_rerankings() {
    caiwei::text::RerankingsRequest request;
    request.model = "qwen3-reranking";
    caiwei::session::RerankingsSession session(request);
    CW_LOG_I("rerankings: %s", session.get().c_str());
}

[[maybe_unused]]
void test_video_play() {
    caiwei::session::VideoPlaySession session("./caiwei.mp4", "file", false, [](const char* /* data */, int length) {
        CW_LOG_I("video play: %d kb", length / 1024);
        return true;
    });
    session.get();
}

int main() {
    #if ENABLE_CAIWEI_RUNTIME_RKNN3
    caiwei::env::set("CAIWEI_CONTEXT_INFO", R"(
CLS,YOLO,yolo26n-cls,yolo26n-cls-rk3588-f16.rknn
DET,YOLO,yolo26n-det,yolo26n-det-rk3588-f16.rknn
ASR,QWEN,qwen3-asr,/data/model/Qwen3-ASR-0.6B/llm/Qwen3-ASR-0.6B.rknn|/data/model/Qwen3-ASR-0.6B/llm/Qwen3-ASR-0.6B.weight|/data/model/Qwen3-ASR-0.6B/llm/Qwen3-ASR-0.6B.embed.bin|/data/model/Qwen3-ASR-0.6B/llm/Qwen3-ASR-0.6B.tokenizer.gguf|/data/model/Qwen3-ASR-0.6B/asr/Qwen3-ASR-0.6B.rknn|/data/model/Qwen3-ASR-0.6B/asr/Qwen3-ASR-0.6B.weight
LLM,QWEN,qwen3-llm,/data/model/Qwen3-1.7B/Qwen3-1.7B.rknn|/data/model/Qwen3-1.7B/Qwen3-1.7B.weight|/data/model/Qwen3-1.7B/Qwen3-1.7B.embed.bin|/data/model/Qwen3-1.7B/Qwen3-1.7B.tokenizer.gguf
VLM,QWEN,qwen3-vlm,/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.rknn|/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.weight|/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.embed.bin|/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.tokenizer.gguf|/data/model/Qwen3-VL-2B-Instruct/vlm/Qwen3-VL-2B-Instruct.rknn|/data/model/Qwen3-VL-2B-Instruct/vlm/Qwen3-VL-2B-Instruct.weight
EMBEDDING,QWEN,qwen3-embedding,/data/model/Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B.rknn|/data/model/Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B.weight|/data/model/Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B.embed.bin|/data/model/Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B.tokenizer.gguf
RERANKING,QWEN,qwen3-reranking,/data/model/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B.rknn|/data/model/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B.weight|/data/model/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B.embed.bin|/data/model/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B.tokenizer.gguf
    )");
    #elif CAIWEI_OS_UNIX
    caiwei::env::set("CAIWEI_CONTEXT_INFO", R"(
CLS,YOLO,yolo26n-cls,yolo26n-cls.onnx
DET,YOLO,yolo26n-det,yolo26n-det.onnx
ASR,QWEN,qwen3-asr,/data/caiwei/deps/llama.cpp/Qwen3-ASR-0.6B/Qwen3-ASR-0.6B-Q4_K_M.gguf|/data/caiwei/deps/llama.cpp/Qwen3-ASR-0.6B/mmproj-Qwen3-ASR-0.6b-Q4_K_M.gguf
LLM,QWEN,qwen3-llm,/data/caiwei/deps/llama.cpp/Qwen3-1.7B/Qwen3-1.7B-Q4_K_M.gguf
VLM,QWEN,qwen3-vlm,/data/caiwei/deps/llama.cpp/Qwen3-VL-2B-InstructQwen3-VL-2B-Instruct-Q4_K_M.gguf|/data/caiwei/deps/llama.cpp/Qwen3-VL-2B-Instruct/mmproj-Qwen3-VL-2b-Instruct-Q4_K_M.gguf
EMBEDDING,QWEN,qwen3-embedding,/data/caiwei/deps/llama.cpp/Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B-Q4_K_M.gguf
RERANKING,QWEN,qwen3-reranking,/data/caiwei/deps/llama.cpp/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B-Q4_K_M.gguf
    )");
    #elif CAIWEI_OS_WIN
//     caiwei::env::set("CAIWEI_CONTEXT_INFO", R"(
// CLS,YOLO,yolo26n-cls,yolo26n-cls.onnx
// DET,YOLO,yolo26n-det,yolo26n-det.onnx
// ASR,QWEN,qwen3-asr,E:/model/llama.cpp/Qwen3-ASR-0.6B-Q4_K_M.gguf|E:/model/llama.cpp/mmproj-Qwen3-ASR-0.6b-Q4_K_M.gguf
// LLM,QWEN,qwen3-llm,E:/model/llama.cpp/Qwen3-1.7B-Q4_K_M.gguf
// VLM,QWEN,qwen3-vlm,E:/model/llama.cpp/Qwen3-VL-2B-Instruct-Q4_K_M.gguf|E:/model/llama.cpp/mmproj-Qwen3-VL-2b-Instruct-Q4_K_M.gguf
// EMBEDDING,QWEN,qwen3-embedding,E:/model/llama.cpp/Qwen3-Embedding-0.6B-Q4_K_M.gguf
// RERANKING,QWEN,qwen3-reranking,E:/model/llama.cpp/Qwen3-Reranker-0.6B-Q4_K_M.gguf
//     )");
    caiwei::env::set("CAIWEI_CONTEXT_INFO", R"(
CLS,YOLO,yolo26n-cls,yolo26n-cls.onnx
DET,YOLO,yolo26n-det,yolo26n-det.onnx
ASR,QWEN,qwen3-asr,D:/tmp/model/llama.cpp/Qwen3-ASR-0.6B-Q4_K_M.gguf|D:/tmp/model/llama.cpp/mmproj-Qwen3-ASR-0.6b-Q4_K_M.gguf
LLM,QWEN,qwen3-llm,D:/tmp/model/llama.cpp/Qwen3-1.7B-Q4_K_M.gguf
VLM,QWEN,qwen3-vlm,D:/tmp/model/llama.cpp/Qwen3-VL-2B-Instruct-Q4_K_M.gguf|D:/tmp/model/llama.cpp/mmproj-Qwen3-VL-2b-Instruct-Q4_K_M.gguf
EMBEDDING,QWEN,qwen3-embedding,D:/tmp/model/llama.cpp/Qwen3-Embedding-0.6B-Q4_K_M.gguf
RERANKING,QWEN,qwen3-reranking,D:/tmp/model/llama.cpp/Qwen3-Reranker-0.6B-Q4_K_M.gguf
    )");
    #else
    #endif
    caiwei::test::init_test();
    test_completions_sync();
    test_completions_async();
    test_embeddings();
    test_rerankings();
    test_video_play();
    caiwei::test::stop_test();
    return 0;
}
