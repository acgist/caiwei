#include "test.hpp"

#include <variant>

[[maybe_unused]]
void test_json_to_completions() {
    auto request1 = caiwei::text::json_to_completions(R"({
        "model": "qwen3",
        "messages": [{
            "role": "system",
            "content": "碧螺萧萧"
        }]
    })");
    ALWAYS_ASSERT(request1.model == "qwen3");
    ALWAYS_ASSERT(request1.messages.size() == 1);
    auto request2 = caiwei::text::json_to_completions(R"({
        "model": "qwen3",
        "messages": [{
            "role": "system",
            "content": "碧螺萧萧"
        }, {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "测试"
            }, {
                "type": "audio_url",
                "audio_url": {
                    "url": "https://www.acgist.com/demo.mp3"
                }
            }]
        }, {
            "role": "tool",
            "tool_call_id": "123456",
            "tool_calls": [{
                "id": "123456",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"city\":\"广州\"}"
                }
            }]
        }],
        "seed": 12345,
        "temperature": 0.5,
        "enable_thinking": true,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询城市天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称"
                        }
                    },
                    "required": ["city"]
                }
            }
        }],
        "video_fps": 30,
        "media_url": "https://www.acgist.com/demo.mp4",
        "media_type": "http",
        "model_list": ["qwen3", "qwen3-vlm"]
    })");
    ALWAYS_ASSERT(request2.model == "qwen3");
    ALWAYS_ASSERT(request2.messages.size() == 3);
    ALWAYS_ASSERT(request2.messages[0].role == "system");
    ALWAYS_ASSERT(request2.messages[1].role == "user");
    ALWAYS_ASSERT(request2.seed == 12345);
    CW_LOG_I("repeat_penalty: %.2f", request2.repeat_penalty.value_or(0.0));
}

[[maybe_unused]]
void test_json_to_embeddings() {
    auto request1 = caiwei::text::json_to_embeddings(R"({
        "model": "qwen3-embedding",
        "input": "你好"
    })");
    ALWAYS_ASSERT(request1.model == "qwen3-embedding");
    std::visit([](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::string>) {
            ALWAYS_ASSERT(arg == "你好");
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            ALWAYS_ASSERT(arg == std::vector<std::string>{"你好"});
        }
    }, request1.input);
    auto request2 = caiwei::text::json_to_embeddings(R"({
        "model": "qwen3-embedding",
        "input": ["你好"]
    })");
    ALWAYS_ASSERT(request2.model == "qwen3-embedding");
    std::visit([](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::string>) {
            ALWAYS_ASSERT(arg == "你好");
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            ALWAYS_ASSERT(arg == std::vector<std::string>{"你好"});
        }
    }, request2.input);
}

[[maybe_unused]]
void test_json_to_rerankings() {
    auto request = caiwei::text::json_to_rerankings(R"({
        "model": "qwen3-reranking",
        "query": "你好",
        "documents": ["你好", "你好吗", "你好啊"]
    })");
    ALWAYS_ASSERT(request.model == "qwen3-reranking");
    ALWAYS_ASSERT((request.documents == std::vector<std::string>{"你好", "你好吗", "你好啊"}));
}

[[maybe_unused]]
void test_completions_response_to_json() {
    caiwei::text::CompletionsResponse response;
    response.id      = "123456";
    response.created = 1694502400;
    response.model   = "qwen3";
    response.choices = std::vector<caiwei::text::CompletionsResponseChoice> {
        {
            .index = 0,
            .finish_reason = "stop",
            .message = {
                .role = "assistant",
                .content = "你好"
            }
        },
    };
    const std::string json = caiwei::text::to_json(response);
    CW_LOG_I("completions_response: %s", json.c_str());
}

[[maybe_unused]]
void test_completions_chunk_to_json() {
    caiwei::text::CompletionsChunk chunk;
    chunk.id      = "123456";
    chunk.created = 1694502400;
    chunk.model   = "qwen3";
    chunk.choices = std::vector<caiwei::text::CompletionsChunkChoice> {
        {
            .index = 0,
            .finish_reason = "stop",
            .delta = {
                .role = "assistant",
                .content = "你好"
            }
        },
    };
    const std::string json = caiwei::text::to_json(chunk);
    CW_LOG_I("completions_chunk: %s", json.c_str());
}

[[maybe_unused]]
void test_embedding_response_to_json() {
    caiwei::text::EmbeddingsResponse response;
    response.model = "qwen3-embedding";
    response.data.push_back({.index = 0, .embedding = {0.5, 1.0}});
    response.data.push_back({.index = 1, .embedding = {0.5, 0.5}});
    response.usage = {
        .prompt_tokens = 100,
        .total_tokens  = 100
    };
    const std::string json = caiwei::text::to_json(response);
    CW_LOG_I("embedding_response: %s", json.c_str());
}

[[maybe_unused]]
void test_reranking_response_to_json() {
    caiwei::text::RerankingsResponse response;
    response.model = "qwen3-reranking";
    response.data.push_back({.index = 0, .score = 0.5});
    response.data.push_back({.index = 1, .score = 0.5});
    response.usage = {
        .prompt_tokens = 100,
        .total_tokens  = 100
    };
    const std::string json = caiwei::text::to_json(response);
    CW_LOG_I("reranking_response: %s", json.c_str());
}

[[maybe_unused]]
void test_image_data_json() {
    std::vector<caiwei::image::Cls> cls = {
        { 1, 0.8F },
        { 2, 0.5F }
    };
    std::vector<caiwei::image::Box> box = {
        { 0.1F, 0.2F, 0.3F, 0.4F, 1, 0.8F },
        { 0.5F, 0.6F, 0.7F, 0.8F, 2, 0.5F }
    };
    CW_LOG_I("cls: %s", caiwei::image::to_json(cls).c_str());
    CW_LOG_I("box: %s", caiwei::image::to_json(box).c_str());
}

int main() {
    caiwei::test::init_test();
    test_json_to_completions();
    test_json_to_embeddings();
    test_json_to_rerankings();
    test_completions_response_to_json();
    test_completions_chunk_to_json();
    test_embedding_response_to_json();
    test_reranking_response_to_json();
    test_image_data_json();
    caiwei::test::stop_test();
    return 0;
}
