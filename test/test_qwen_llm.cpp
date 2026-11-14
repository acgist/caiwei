#include "test.hpp"

#include <thread>
#include <vector>

#include "nlohmann/json.hpp"

[[maybe_unused]]
void llm_run(caiwei::manager::LLMWrapper* ptr, const caiwei::text::CompletionsRequest& request) {
    ALWAYS_ASSERT(ptr != nullptr);
    std::string content;
    std::string thinking;
    std::string toolcall_name;
    std::string toolcall_arguments;
    for (const auto& value : ptr->run(request)) {
        if (value.toolcall) {
            toolcall_name      += value.toolcall_name;
            toolcall_arguments += value.toolcall_arguments;
        } else if (value.thinking) {
            thinking += value.token;
        } else {
            content += value.token;
        }
        if (!value.finish_reason.empty()) {
            CW_LOG_I("完成: %s", value.finish_reason.c_str());
        }
    }
    CW_LOG_I("content: %s", content.c_str());
    CW_LOG_I("thinking: %s", thinking.c_str());
    CW_LOG_I("toolcall_name: %s", toolcall_name.c_str());
    CW_LOG_I("toolcall_arguments: %s", toolcall_arguments.c_str());
}

[[maybe_unused]]
void test_llm() {
    caiwei::text::CompletionsRequest request;
    request.stream = true;
    request.enable_thinking = true;
    request.messages.push_back({.role = "user", .content = "解释一下碧螺萧萧"});
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
    auto ptr = caiwei::manager::get_context<caiwei::context::LLMContext>("qwen3-llm");
    if (!ptr) {
        return;
    }
    llm_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
    request.messages.clear();
    request.messages.push_back({.role = "user", .content = "解释一下碧螺萧萧"});
    request.enable_thinking = false;
    llm_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
    request.messages.clear();
    request.messages.push_back({.role = "user", .content = "查询北京天气"});
    llm_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
    request.tools.reset();
    request.messages.clear();
    request.messages.push_back({.role = "user", .content = "1+1=?"});
    llm_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
    request.max_completion_tokens = 2;
    llm_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
}

[[maybe_unused]]
void test_llm_for() {
    caiwei::text::CompletionsRequest request;
    request.stream = true;
    request.enable_thinking = false;
    request.messages.push_back({.role = "user", .content = "1+1=?"});
    auto ptr = caiwei::manager::get_context<caiwei::context::LLMContext>("qwen3-llm");
    if (!ptr) {
        return;
    }
    caiwei::test::caiwei_wait_enter();
    llm_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
    CAIWEI_FOR_EACH(100)
    llm_run(ptr.get(), request);
    CAIWEI_FOR_EACH_END
    caiwei::test::caiwei_wait_enter();
    llm_run(ptr.get(), request);
    caiwei::test::caiwei_wait_enter();
}

[[maybe_unused]]
void test_llm_thread() {
    caiwei::text::CompletionsRequest request;
    request.stream = true;
    request.enable_thinking = false;
    request.messages.push_back({.role = "user", .content = "1+1=?"});
    auto ptr = caiwei::manager::get_context<caiwei::context::LLMContext>("qwen3-llm");
    if (!ptr) {
        return;
    }
    std::vector<std::jthread> threads;
    for (int i = 0; i < 2; ++i) {
        threads.push_back(std::jthread([&]() {
            CAIWEI_FOR_EACH(100)
            llm_run(ptr.get(), request);
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
    test_llm_for();
    // test_llm_thread();
    // test_llm_thread();
    caiwei::test::stop_test();
    return 0;
}
