#include "test.hpp"

#include "caiwei/text_tool.hpp"

[[maybe_unused]]
void test_chat_template() {
    caiwei::text::ChatTemplate chat_template;
    {
        // CAIWEI_FOR_EACH(100)
        chat_template.set_template("{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {%- if messages[0].content is string %}\n            {{- messages[0].content }}\n        {%- else %}\n            {%- for content in messages[0].content %}\n                {%- if 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' }}\n        {%- if messages[0].content is string %}\n            {{- messages[0].content }}\n        {%- else %}\n            {%- for content in messages[0].content %}\n                {%- if 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set image_count = namespace(value=0) %}\n{%- set video_count = namespace(value=0) %}\n{%- for message in messages %}\n    {%- if message.role == \"user\" %}\n        {{- '<|im_start|>' + message.role + '\\n' }}\n        {%- if message.content is string %}\n            {{- message.content }}\n        {%- else %}\n            {%- for content in message.content %}\n                {%- if content.type == 'image' or 'image' in content or 'image_url' in content %}\n                    {%- set image_count.value = image_count.value + 1 %}\n                    {%- if add_vision_id %}Picture {{ image_count.value }}: {% endif -%}\n                    <|vision_start|><|image_pad|><|vision_end|>\n                {%- elif content.type == 'video' or 'video' in content %}\n                    {%- set video_count.value = video_count.value + 1 %}\n                    {%- if add_vision_id %}Video {{ video_count.value }}: {% endif -%}\n                    <|vision_start|><|video_pad|><|vision_end|>\n                {%- elif 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role + '\\n' }}\n        {%- if message.content is string %}\n            {{- message.content }}\n        {%- else %}\n            {%- for content_item in message.content %}\n                {%- if 'text' in content_item %}\n                    {{- content_item.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and message.content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {%- if message.content is string %}\n            {{- message.content }}\n        {%- else %}\n            {%- for content in message.content %}\n                {%- if content.type == 'image' or 'image' in content or 'image_url' in content %}\n                    {%- set image_count.value = image_count.value + 1 %}\n                    {%- if add_vision_id %}Picture {{ image_count.value }}: {% endif -%}\n                    <|vision_start|><|image_pad|><|vision_end|>\n                {%- elif content.type == 'video' or 'video' in content %}\n                    {%- set video_count.value = video_count.value + 1 %}\n                    {%- if add_vision_id %}Video {{ video_count.value }}: {% endif -%}\n                    <|vision_start|><|video_pad|><|vision_end|>\n                {%- elif 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n", "", "");
        // CAIWEI_FOR_EACH_END
    }
    caiwei::text::CompletionsRequest request;
    request.extra_body = {
        .enable_thinking = true
    };
    request.messages.push_back({.role = "system", .content = "帮助用户查询城市天气"});
    request.messages.push_back({.role = "user", .content = std::vector<caiwei::text::CompletionsRequestMessageContentItem> {
        caiwei::text::CompletionsRequestMessageContentItem {
            .type = "text",
            .text = "测试"
        },
        caiwei::text::CompletionsRequestMessageContentItem {
            .type  = "audio",
            .audio = "测试"
        },
        caiwei::text::CompletionsRequestMessageContentItem {
            .type  = "image",
            .image = "测试"
        },
        caiwei::text::CompletionsRequestMessageContentItem {
            .type  = "image",
            .image = "测试"
        },
        caiwei::text::CompletionsRequestMessageContentItem {
            .type  = "video",
            .video = "测试"
        },
    }});
    request.messages.push_back({.role = "assistant", .content = "北京"});
    request.messages.push_back({.role = "user", .content = "北京"});
    request.messages.push_back({.role = "tool", .content = "北京"});
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
    const caiwei::text::SpecialToken special_token;
    {
        CAIWEI_FOR_EACH(100)
        chat_template.apply(special_token, request);
        CAIWEI_FOR_EACH_END
    }
    std::string prompt = chat_template.apply(special_token, request);
    CW_LOG_D("prompt: %s", prompt.c_str());
}

[[maybe_unused]]
void test_json_to_completions() {
    auto request1 = caiwei::text::json_to_completions(R"({
        "model": "qwen3",
        "messages": [{
            "role": "system",
            "content": "碧螺萧萧"
        }]
    })");
    assert(request1.model == "qwen3");
    assert(request1.messages.size() == 1);
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
                        "location": {
                            "type": "string",
                            "description": "城市名称"
                        }
                    },
                    "required": ["location"]
                }
            }
        }],
        "extra_body": {
            "video_fps": 30,
            "media_url": "https://www.acgist.com/demo.mp4",
            "media_type": "http",
            "model_list": ["qwen3", "qwen3-vlm"]
        }
    })");
    assert(request2.model == "qwen3");
    assert(request2.messages.size() == 3);
    assert(request2.messages[0].role == "system");
    assert(request2.messages[1].role == "user");
    assert(request2.seed == 12345);
    assert(!request2.repeat_penalty.has_value());
}

[[maybe_unused]]
void test_json_to_embeddings() {
    auto request1 = caiwei::text::json_to_embeddings(R"({
        "model": "qwen3-embedding",
        "input": "你好"
    })");
    assert(request1.model == "qwen3-embedding");
    std::visit([](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::string>) {
            assert(arg == "你好");
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            assert(arg == std::vector<std::string>{"你好"});
        }
    }, request1.input);
    auto request2 = caiwei::text::json_to_embeddings(R"({
        "model": "qwen3-embedding",
        "input": ["你好"]
    })");
    assert(request2.model == "qwen3-embedding");
    std::visit([](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::string>) {
            assert(arg == "你好");
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            assert(arg == std::vector<std::string>{"你好"});
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
    assert(request.model == "qwen3-reranking");
}

[[maybe_unused]]
void test_completions_response_to_json() {
    caiwei::text::CompletionsResponse response;
    response.created = 1694502400;
    response.id      = "123456";
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
    std::string json = caiwei::text::to_json(response);
    CW_LOG_D("test_completions_response_to_json: %s", json.c_str());
}

[[maybe_unused]]
void test_completions_chunk_to_json() {
    caiwei::text::CompletionsChunk chunk;
    chunk.created = 1694502400;
    chunk.id      = "123456";
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
    std::string json = caiwei::text::to_json(chunk);
    CW_LOG_D("test_completions_chunk_to_json: %s", json.c_str());
}

[[maybe_unused]]
void test_embedding_response_to_json() {
    caiwei::text::EmbeddingResponse response;
    response.model = "qwen3-embedding";
    response.data.push_back({.index = 0, .embedding = {0.5, 1.0}});
    response.data.push_back({.index = 1, .embedding = {0.5, 0.5}});
    response.usage = {
        .prompt_tokens = 100,
        .total_tokens  = 100
    };
    std::string json = caiwei::text::to_json(response);
    CW_LOG_D("test_embedding_response_to_json: %s", json.c_str());
}

[[maybe_unused]]
void test_reranking_response_to_json() {
    caiwei::text::RerankingResponse response;
    response.model = "qwen3-reranking";
    response.data.push_back({.index = 0, .score = 0.5});
    response.data.push_back({.index = 1, .score = 0.5});
    response.usage = {
        .prompt_tokens = 100,
        .total_tokens  = 100
    };
    std::string json = caiwei::text::to_json(response);
    CW_LOG_D("test_reranking_response_to_json: %s", json.c_str());
}

int main() {
    caiwei::test::init_test();
    // test_chat_template();
    test_json_to_completions();
    // test_json_to_embeddings();
    // test_json_to_rerankings();
    // test_completions_response_to_json();
    // test_completions_chunk_to_json();
    // test_embedding_response_to_json();
    // test_reranking_response_to_json();
    caiwei::test::stop_test();
    return 0;
}
