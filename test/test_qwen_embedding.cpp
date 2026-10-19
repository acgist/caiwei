#include "test.hpp"

#include "nlohmann/json.hpp"

#include <thread>
#include <vector>

static void diff(const std::vector<float>& a, const std::vector<float>& z) {
    float sum = 0.0F;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = a[i] - z[i];
        sum += d * d;
    }
    float euclidean_dist = std::sqrt(sum);
    CW_LOG_I("欧式距离: %f", euclidean_dist);
}

void test_embedding() {
    caiwei::text::EmbeddingsRequest request;
    request.model = "qwen3-embedding";
    // request.input = "北京";
    // request.input = std::vector<std::string>{ "苹果", "苹果", "橘子", "汽车" };
    // request.input = std::vector<std::string>{ "美国首都那座城市", "华盛顿", "中国首都是北京", "美国首都是广州" };
    request.input = std::vector<std::string>{ "广州今天的天气怎么样", "出门可以坐地铁", "广州正在下雨", "达州正在下雨" };
    auto ptr = caiwei::manager::get_context<caiwei::context::EmbeddingContext, caiwei::text::EmbeddingsRequest, caiwei::text::EmbeddingResult>("qwen3-embedding");
    if (!ptr) {
        return;
    }
    auto ret = ptr->run(request);
    for (const auto& item : ret.result) {
        std::printf("%s\n", nlohmann::json(item).dump().c_str());
    }
    diff(ret.result[0], ret.result[1]);
    diff(ret.result[0], ret.result[2]);
    diff(ret.result[0], ret.result[3]);
    std::fflush(stdout);
}

void test_embedding_for() {
    caiwei::text::EmbeddingsRequest request;
    request.model = "qwen3-embedding";
    request.input = std::vector<std::string>{ "广州今天的天气怎么样", "出门可以坐地铁", "广州正在下雨", "达州正在下雨" };
    auto ptr = caiwei::manager::get_context<caiwei::context::EmbeddingContext, caiwei::text::EmbeddingsRequest, caiwei::text::EmbeddingResult>("qwen3-embedding");
    if (!ptr) {
        return;
    }
    caiwei_wait_enter();
    ptr->run(request);
    caiwei_wait_enter();
    CAIWEI_FOR_EACH(100)
    ptr->run(request);
    CAIWEI_FOR_EACH_END
    caiwei_wait_enter();
    ptr->run(request);
    caiwei_wait_enter();
}

void test_embedding_thread() {
    caiwei::text::EmbeddingsRequest request;
    request.model = "qwen3-embedding";
    request.input = std::vector<std::string>{ "广州今天的天气怎么样", "出门可以坐地铁", "广州正在下雨", "达州正在下雨" };
    auto ptr = caiwei::manager::get_context<caiwei::context::EmbeddingContext, caiwei::text::EmbeddingsRequest, caiwei::text::EmbeddingResult>("qwen3-embedding");
    if (!ptr) {
        return;
    }
    auto true_ret = ptr->run(request);
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.push_back(std::thread([&]() {
            CAIWEI_FOR_EACH(100)
            auto ret = ptr->run(request);
            assert(ret.result == true_ret.result);
            CAIWEI_FOR_EACH_END
        }));
    }
    for (auto& thread : threads) {
        thread.join();
    }
    caiwei_wait_enter();
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
    // test_embedding_for();
    // test_embedding_thread();
    // test_embedding_thread();
    caiwei::test::stop_test();
    return 0;
}