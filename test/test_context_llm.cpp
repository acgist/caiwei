#include "test_runtime.hpp"

#include "nlohmann/json.hpp"

void test_llm() {
    caiwei::text::CompletionsRequest request;
    auto ptr = caiwei::context::get_context<caiwei::context::LLMContext, caiwei::text::CompletionsRequest, std::vector<std::string>>("qwen3-llm");
    auto result = ptr->run(request);
    for (const auto& value : result) {
        CW_LOG_D("value: %s", value.c_str());
    }
}

int main() {
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "LLM,QWEN,qwen3-llm,D:/tmp/model/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf");
    init_test();
    test_llm();
    stop_test();
    return 0;
}