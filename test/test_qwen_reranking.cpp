#include "test.hpp"

#include "nlohmann/json.hpp"

#include <thread>
#include <vector>

[[maybe_unused]]
void test_reranking() {
    caiwei::text::RerankingsRequest request;
    request.model = "qwen3-reranking";
    request.query = "广州今天的天气怎么样";
    request.documents = std::vector<std::string>{ "下雨打车出门", "广州正在下雨", "广州美食不错", "达州正在下雨" };
    auto ptr = caiwei::manager::get_context<caiwei::context::RerankingContext>("qwen3-reranking");
    if (!ptr) {
        CW_LOG_W("模型加载失败");
        return;
    }
    const auto ret = ptr->run(request);
    for (const auto& item : ret.result) {
        CW_LOG_I("reranking: %s", nlohmann::json(item).dump().c_str());
    }
}

[[maybe_unused]]
void test_reranking_for() {
    caiwei::text::RerankingsRequest request;
    request.model = "qwen3-reranking";
    request.query = "广州今天的天气怎么样";
    request.documents = std::vector<std::string>{ "下雨打车出门", "广州正在下雨", "广州美食不错", "达州正在下雨" };
    auto ptr = caiwei::manager::get_context<caiwei::context::RerankingContext>("qwen3-reranking");
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
void test_reranking_thread() {
    caiwei::text::RerankingsRequest request;
    request.model = "qwen3-reranking";
    request.query = "广州今天的天气怎么样";
    request.documents = std::vector<std::string>{ "下雨打车出门", "广州正在下雨", "广州美食不错", "达州正在下雨" };
    auto ptr = caiwei::manager::get_context<caiwei::context::RerankingContext>("qwen3-reranking");
    if (!ptr) {
        CW_LOG_W("模型加载失败");
        return;
    }
    const auto true_ret = ptr->run(request);
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
    caiwei::test::caiwei_wait_enter();
}

int main() {
    #if ENABLE_CAIWEI_RUNTIME_RKNN3
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "RERANKING,QWEN,qwen3-reranking,/data/model/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B.rknn|/data/model/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B.weight|/data/model/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B.embed.bin|/data/model/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B.tokenizer.gguf");
    #elif CAIWEI_OS_UNIX
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "RERANKING,QWEN,qwen3-reranking,/data/caiwei/deps/llama.cpp/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B-Q4_K_M.gguf");
    #elif CAIWEI_OS_WIN
    // caiwei::env::set("CAIWEI_CONTEXT_INFO", "RERANKING,QWEN,qwen3-reranking,E:/model/llama.cpp/Qwen3-Reranker-0.6B-Q4_K_M.gguf");
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "RERANKING,QWEN,qwen3-reranking,D:/tmp/model/llama.cpp/Qwen3-Reranker-0.6B-Q4_K_M.gguf");
    #else
    #endif
    caiwei::test::init_test();
    test_reranking();
    // test_reranking_for();
    // test_reranking_thread();
    // test_reranking_thread();
    caiwei::test::stop_test();
    return 0;
}