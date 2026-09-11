#include "test.hpp"

#include "nlohmann/json.hpp"

#include <generator>

void test_llm() {
    caiwei::text::CompletionsRequest request;
    request.stream = true;
    request.enable_thinking = false;
    request.messages.push_back({.role = "system", .content = "帮助用户查询城市天气"});
    request.messages.push_back({.role = "user", .content = "北京"});
    request.tools = {
        caiwei::text::CompletionsRequestTool {
            .type = "function",
            .function = caiwei::text::CompletionsRequestToolFunction {
                .name = "get_weather",
                .description = "查询城市天气",
                .parameters = R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})"_json,
            },
        }
    };
    // {"name": "get_weather", "arguments": {"city": "北京"}}
    auto ptr = caiwei::manager::get_context<caiwei::context::LLMContext, caiwei::text::CompletionsRequest, std::generator<std::string>>("qwen3-llm");
    if (!ptr) {
        return;
    }
    CAIWEI_FOR_EACH(100)
    for (const std::string& value : ptr->run(request)) {
        std::printf("%s", value.c_str());
        std::fflush(stdout);
    }
    CAIWEI_FOR_EACH_END
}

int main() {
    #if ENABLE_CAIWEI_RUNTIME_RKNN3
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "LLM,QWEN,qwen3-llm,/data/model/Qwen3-4B.rknn|/data/model/Qwen3-4B.weight|/data/model/Qwen3-4B.embed.bin|/data/model/Qwen3-4B.tokenizer.gguf");
    #else
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "LLM,QWEN,qwen3-llm,D:/tmp/model/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf");
    #endif
    caiwei::test::init_test();
    test_llm();
    caiwei::test::stop_test();
    return 0;
}