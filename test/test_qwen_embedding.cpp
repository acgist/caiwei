#include "test.hpp"

#include "nlohmann/json.hpp"

#include <thread>
#include <vector>

[[maybe_unused]]
static void cosine_similarity(const std::vector<float>& a, const std::vector<float>& z) {
    if (a.empty() || z.empty() || a.size() != z.size()) {
        return;
    }
    double dot    = 0.0;
    double norm_a = 0.0;
    double norm_z = 0.0;
    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        dot    += (double) a[i] * z[i];
        norm_a += (double) a[i] * a[i];
        norm_z += (double) z[i] * z[i];
    }
    norm_a = std::sqrt(norm_a);
    norm_z = std::sqrt(norm_z);
    if (norm_a < 1e-8 || norm_z < 1e-8) {
        return;
    }
    const float ret = static_cast<float>(dot / (norm_a * norm_z));
    CW_LOG_I("余弦相似度: %.2f", ret);
}

[[maybe_unused]]
void test_embedding() {
    caiwei::text::EmbeddingsRequest request;
    request.model = "qwen3-embedding";
    // request.input = "北京";
    request.input = std::vector<std::string>{ "小狗", "猫咪", "苹果", "汽车" };
    auto ptr = caiwei::manager::get_context<caiwei::context::EmbeddingContext>("qwen3-embedding");
    if (!ptr) {
        CW_LOG_W("模型加载失败");
        return;
    }
    const auto ret = ptr->run(request);
    for (const auto& item : ret.result) {
        CW_LOG_I("embedding: %zu", item.size());
    }
    assert((ret.result.size() == std::get<std::vector<std::string>>(request.input).size()));
    cosine_similarity(ret.result[0], ret.result[1]);
    cosine_similarity(ret.result[0], ret.result[2]);
    cosine_similarity(ret.result[0], ret.result[3]);
    cosine_similarity(ret.result[1], ret.result[2]);
    cosine_similarity(ret.result[1], ret.result[3]);
    cosine_similarity(ret.result[2], ret.result[3]);
}

[[maybe_unused]]
void test_embedding_for() {
    caiwei::text::EmbeddingsRequest request;
    request.model = "qwen3-embedding";
    request.input = std::vector<std::string>{ "小狗", "猫咪", "苹果", "汽车" };
    auto ptr = caiwei::manager::get_context<caiwei::context::EmbeddingContext>("qwen3-embedding");
    if (!ptr) {
        CW_LOG_W("模型加载失败");
        return;
    }
    caiwei::test::caiwei_wait_enter();
    ptr->run(request);
    caiwei::test::caiwei_wait_enter();
    CAIWEI_FOR_EACH(100)
    ptr->run(request);
    CAIWEI_FOR_EACH_END
    caiwei::test::caiwei_wait_enter();
    ptr->run(request);
    caiwei::test::caiwei_wait_enter();
}

[[maybe_unused]]
void test_embedding_thread() {
    caiwei::text::EmbeddingsRequest request;
    request.model = "qwen3-embedding";
    request.input = std::vector<std::string>{ "小狗", "猫咪", "苹果", "汽车" };
    // 共享模型线程安全所有线程共享一个模型实例
    auto ptr = caiwei::manager::get_context<caiwei::context::EmbeddingContext>("qwen3-embedding");
    if (!ptr) {
        CW_LOG_W("模型加载失败");
        return;
    }
    const auto true_ret = ptr->run(request);
    std::vector<std::jthread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.push_back(std::jthread([&]() {
            CAIWEI_FOR_EACH(100)
            auto ret = ptr->run(request);
            assert(ret.result == true_ret.result);
            CAIWEI_FOR_EACH_END
        }));
    }
    for (auto& thread : threads) {
        thread.join();
    }
    caiwei::test::caiwei_wait_enter();
}

int main() {
    #if ENABLE_CAIWEI_RUNTIME_RKNN3
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "EMBEDDING,QWEN,qwen3-embedding,/data/model/Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B.rknn|/data/model/Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B.weight|/data/model/Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B.embed.bin|/data/model/Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B.tokenizer.gguf");
    #elif CAIWEI_OS_UNIX
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "EMBEDDING,QWEN,qwen3-embedding,/data/caiwei/deps/llama.cpp/Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B-Q4_K_M.gguf");
    #elif CAIWEI_OS_WIN
    // caiwei::env::set("CAIWEI_CONTEXT_INFO", "EMBEDDING,QWEN,qwen3-embedding,E:/model/llama.cpp/Qwen3-Embedding-0.6B-Q4_K_M.gguf");
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "EMBEDDING,QWEN,qwen3-embedding,D:/tmp/model/llama.cpp/Qwen3-Embedding-0.6B-Q4_K_M.gguf");
    #else
    #endif
    caiwei::test::init_test();
    test_embedding();
    test_embedding_for();
    // test_embedding_thread();
    // test_embedding_thread();
    caiwei::test::stop_test();
    return 0;
}