#include "test.hpp"

#include "nlohmann/json.hpp"

#include <thread>
#include <vector>
#include <generator>

void test_llm() {
    caiwei::text::CompletionsRequest request;
    request.stream = true;
    request.enable_thinking = false;
    request.messages.push_back({.role = "user", .content = "解释一下碧螺萧萧"});
    // request.messages.push_back({.role = "system", .content = "帮助用户查询城市天气"});
    // request.messages.push_back({.role = "user", .content = "查询北京天气"});
    // request.tools = {
    //     caiwei::text::CompletionsRequestTool {
    //         .type = "function",
    //         .function = caiwei::text::CompletionsRequestToolFunction {
    //             .name = "get_weather",
    //             .description = "查询城市天气",
    //             .parameters = R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})"_json,
    //         },
    //     }
    // };
    // {"name": "get_weather", "arguments": {"city": "北京"}}
    auto ptr = caiwei::manager::get_context<caiwei::context::LLMContext>("qwen3-llm");
    if (!ptr) {
        return;
    }
    for (const auto& value : ptr->run(request)) {
        // -
    }
    request.max_completion_tokens = 2;
    for (const auto& value : ptr->run(request)) {
        // -
    }
}

void test_llm_for() {
    caiwei::text::CompletionsRequest request;
    request.stream = true;
    request.enable_thinking = false;
    request.messages.push_back({.role = "user", .content = "1+1=?"});
    auto ptr = caiwei::manager::get_context<caiwei::context::LLMContext>("qwen3-llm");
    if (!ptr) {
        return;
    }
    caiwei_wait_enter();
    for (const auto& value : ptr->run(request)) {
        // -
    }
    caiwei_wait_enter();
    CAIWEI_FOR_EACH(100)
    for (const auto& value : ptr->run(request)) {
        // -
    }
    CAIWEI_FOR_EACH_END
    caiwei_wait_enter();
    for (const auto& value : ptr->run(request)) {
        // -
    }
    caiwei_wait_enter();
}

void test_llm_thread() {
    caiwei::text::CompletionsRequest request;
    request.stream = true;
    request.enable_thinking = false;
    request.messages.push_back({.role = "user", .content = "1+1=?"});
    auto ptr = caiwei::manager::get_context<caiwei::context::LLMContext>("qwen3-llm");
    if (!ptr) {
        return;
    }
    std::vector<std::thread> threads;
    for (int i = 0; i < 2; ++i) {
        threads.push_back(std::thread([&]() {
            CAIWEI_FOR_EACH(100)
            for (const auto& value : ptr->run(request)) {
                // -
            }
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
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "LLM,QWEN,qwen3-llm,/data/model/Qwen3-1.7B/Qwen3-1.7B.rknn|/data/model/Qwen3-1.7B/Qwen3-1.7B.weight|/data/model/Qwen3-1.7B/Qwen3-1.7B.embed.bin|/data/model/Qwen3-1.7B/Qwen3-1.7B.tokenizer.gguf");
    #elif CAIWEI_OS_UNIX
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "LLM,QWEN,qwen3-llm,/data/caiwei/deps/llama.cpp/Qwen3-1.7B/Qwen3-1.7B-Q4_K_M.gguf");
    #elif CAIWEI_OS_WIN
    // caiwei::env::set("CAIWEI_CONTEXT_INFO", "LLM,QWEN,qwen3-llm,E:/model/llama.cpp/Qwen3-1.7B-Q4_K_M.gguf");
    caiwei::env::set("CAIWEI_CONTEXT_INFO", "LLM,QWEN,qwen3-llm,D:/tmp/model/llama.cpp/Qwen3-1.7B-Q4_K_M.gguf");
    #else
    #endif
    caiwei::test::init_test();
    test_llm();
    // test_llm_for();
    // test_llm_thread();
    // test_llm_thread();
    caiwei::test::stop_test();
    return 0;
}
