#include "test.hpp"

[[maybe_unused]]
void test_json_to_completions() {
    auto request = caiwei::text::json_to_completions(R"({
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
            "content": "{\"city\":\"广州\"}"
        }],
        "seed": 12345,
        "temperature": 0.5,
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
        "enable_thinking": true,
        "video_fps": 30,
        "media_url": "https://www.acgist.com/demo.mp4",
        "media_type": "http",
        "model_list": ["qwen3", "qwen3-vlm"]
    })");
    CAIWEI_ASSERT(request.model == "qwen3");
    CAIWEI_ASSERT(request.messages.size() == 3);
    CAIWEI_ASSERT(request.messages[0].role == "system");
    CAIWEI_ASSERT(request.messages[1].role == "user");
    CAIWEI_ASSERT(request.seed == 12345);
    CAIWEI_ASSERT(request.repeat_penalty.has_value());
}

[[maybe_unused]]
void test_json_to_embeddings() {
    {
        auto request = caiwei::text::json_to_embeddings(R"({
            "model": "qwen3-embedding",
            "input": "你好"
        })");
        CAIWEI_ASSERT(request.model == "qwen3-embedding");
        CAIWEI_ASSERT(std::holds_alternative<std::string>(request.input));
        CAIWEI_ASSERT(std::get<std::string>(request.input) == "你好");
    }
    {
        auto request = caiwei::text::json_to_embeddings(R"({
            "model": "qwen3-embedding",
            "input": ["你好"]
        })");
        CAIWEI_ASSERT(request.model == "qwen3-embedding");
        CAIWEI_ASSERT(std::holds_alternative<std::vector<std::string>>(request.input));
        CAIWEI_ASSERT(std::get<std::vector<std::string>>(request.input) == std::vector<std::string>{"你好"});
    }
}

[[maybe_unused]]
void test_json_to_rerankings() {
    auto request = caiwei::text::json_to_rerankings(R"({
        "model": "qwen3-reranking",
        "query": "你好",
        "documents": ["你好", "你好吗", "你好啊"]
    })");
    CAIWEI_ASSERT(request.model == "qwen3-reranking");
    CAIWEI_ASSERT((request.documents == std::vector<std::string>{"你好", "你好吗", "你好啊"}));
}

[[maybe_unused]]
void test_completions_response_to_json() {
    caiwei::text::CompletionsResponse response;
    response.id = "123456";
    response.created = 1694502400;
    response.model = "qwen3";
    response.choices = std::vector<caiwei::text::CompletionsResponseChoice>{
        caiwei::text::CompletionsResponseChoice{
            .index = 0,
            .finish_reason = "stop",
            .message = caiwei::text::CompletionsResponseChoiceMessage{.role = "assistant", .content = "你好"}
        },
    };
    const std::string json = caiwei::text::to_json(response);
    CW_LOG_I("completions_response: %s", json.c_str());
    CAIWEI_ASSERT(
        json ==
        R"({"choices":[{"finish_reason":"stop","index":0,"message":{"content":"你好","role":"assistant"}}],"created":1694502400,"id":"123456","model":"qwen3","object":"chat.completion"})"
    );
}

[[maybe_unused]]
void test_completions_chunk_to_json() {
    caiwei::text::CompletionsChunk chunk;
    chunk.id = "123456";
    chunk.created = 1694502400;
    chunk.model = "qwen3";
    chunk.choices = std::vector<caiwei::text::CompletionsChunkChoice>{
        caiwei::text::CompletionsChunkChoice{
            .index = 0,
            .finish_reason = "stop",
            .delta = caiwei::text::CompletionsChunkDelta{.role = "assistant", .content = "你好"}
        },
    };
    const std::string json = caiwei::text::to_json(chunk);
    CW_LOG_I("completions_chunk: %s", json.c_str());
    CAIWEI_ASSERT(
        json ==
        R"({"choices":[{"delta":{"content":"你好","role":"assistant"},"finish_reason":"stop","index":0}],"created":1694502400,"id":"123456","model":"qwen3","object":"chat.completion.chunk"})"
    );
}

[[maybe_unused]]
void test_embedding_response_to_json() {
    caiwei::text::EmbeddingsResponse response;
    response.model = "qwen3-embedding";
    response.data.push_back({.index = 0, .embedding = {0.5F, 1.0F}});
    response.data.push_back({.index = 1, .embedding = {0.5F, 0.5F}});
    response.usage = {.prompt_tokens = 100, .total_tokens = 100};
    const std::string json = caiwei::text::to_json(response);
    CW_LOG_I("embedding_response: %s", json.c_str());
    CAIWEI_ASSERT(
        json ==
        R"({"data":[{"embedding":[0.5,1.0],"index":0,"object":"embedding"},{"embedding":[0.5,0.5],"index":1,"object":"embedding"}],"model":"qwen3-embedding","object":"list","usage":{"prompt_tokens":100,"total_tokens":100}})"
    );
}

[[maybe_unused]]
void test_reranking_response_to_json() {
    caiwei::text::RerankingsResponse response;
    response.model = "qwen3-reranking";
    response.data.push_back({.index = 0, .score = 0.5F});
    response.data.push_back({.index = 1, .score = 0.5F});
    response.usage = {.prompt_tokens = 100, .total_tokens = 100};
    const std::string json = caiwei::text::to_json(response);
    CW_LOG_I("reranking_response: %s", json.c_str());
    CAIWEI_ASSERT(
        json ==
        R"({"data":[{"index":0,"object":"reranking","score":0.5},{"index":1,"object":"reranking","score":0.5}],"model":"qwen3-reranking","object":"list","usage":{"prompt_tokens":100,"total_tokens":100}})"
    );
}

[[maybe_unused]]
void test_image_data_json() {
    // clang-format off
    std::vector<caiwei::media::Cls> cls = {
        {1, 0.8F},
        {2, 0.5F}
    };
    std::vector<caiwei::media::Box> box = {
        {0.1F, 0.2F, 0.3F, 0.4F, 1, 0.8F},
        {0.5F, 0.6F, 0.7F, 0.8F, 2, 0.5F}
    };
    // clang-format on
    auto cls_json = caiwei::media::to_json(cls);
    auto box_json = caiwei::media::to_json(box);
    CW_LOG_I("cls: %s", cls_json.c_str());
    CW_LOG_I("box: %s", box_json.c_str());
    CAIWEI_ASSERT(cls_json == R"([{"class_id":1,"score":0.800000011920929},{"class_id":2,"score":0.5}])");
    CAIWEI_ASSERT(
        box_json ==
        R"([{"class_id":1,"score":0.800000011920929,"x1":0.10000000149011612,"x2":0.30000001192092896,"y1":0.20000000298023224,"y2":0.4000000059604645},{"class_id":2,"score":0.5,"x1":0.5,"x2":0.699999988079071,"y1":0.6000000238418579,"y2":0.800000011920929}])"
    );
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
