#include "test.hpp"

#include "nlohmann/json.hpp"

#include <vector>

void test_reranking() {
    caiwei::text::RerankingsRequest request;
    request.model = "qwen3-reranking";
    request.query = "苹果";
    request.documents = std::vector<std::string>{ "苹果", "橘子", "汽车" };
    // request.query = "美国首都那座城市";
    // request.documents = std::vector<std::string>{ "华盛顿", "中国首都是北京", "美国首都是广州" };
    // request.query = "广州今天的天气怎么样";
    // request.documents = std::vector<std::string>{ "出门可以坐地铁", "广州正在阴天", "广州美食不错", "达州正在下雨" };
    auto ptr = caiwei::manager::get_context<caiwei::context::RerankingContext, caiwei::text::RerankingsRequest, caiwei::text::RerankingResult>("qwen3-reranking");
    if (!ptr) {
        return;
    }
    // CAIWEI_FOR_EACH(100)
    // ptr->run(request);
    // CAIWEI_FOR_EACH_END
    auto ret = ptr->run(request);
    for (const auto& item : ret.result) {
        std::printf("%s\n", nlohmann::json(item).dump().c_str());
    }
    std::fflush(stdout);
}

int main() {
    #if ENABLE_CAIWEI_RUNTIME_RKNN3
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "RERANKING,QWEN,qwen3-reranking,/data/model/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B.rknn|/data/model/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B.weight|/data/model/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B.embed.bin|/data/model/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B.tokenizer.gguf");
    #elif CAIWEI_OS_WIN
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "RERANKING,QWEN,qwen3-reranking,E:/model/llama.cpp/Qwen3-Reranker-0.6B-Q8_0.gguf");
    // caiwei::env::set("CAIWEI_CONTEXT_INFO", "RERANKING,QWEN,qwen3-reranking,D:/tmp/model/llama.cpp/Qwen3-Reranker-0.6B-Q4_K_M.gguf");
    #else
    #endif
    caiwei::test::init_test();
    test_reranking();
    caiwei::test::stop_test();
    return 0;
}