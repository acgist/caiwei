#include "test.hpp"

#include "llama.h"

#include "caiwei/tokenizer.hpp"

[[maybe_unused]]
void test_tokenizer() {
    caiwei::context::Tokenizer tokenizer;
    #if ENABLE_CAIWEI_RUNTIME_RKNN3
    if (!tokenizer.load("/data/model/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B.tokenizer.gguf")) {
        CW_LOG_W("模型加载失败");
        return;
    }
    #elif CAIWEI_OS_UNIX
    if (!tokenizer.load("/data/caiwei/deps/llama.cpp/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B-Q4_K_M.gguf")) {
        CW_LOG_W("模型加载失败");
        return;
    }
    #else
    if (!tokenizer.load("D:/tmp/model/llama.cpp/Qwen3-1.7B-Q4_K_M.gguf")) {
        CW_LOG_W("模型加载失败");
        return;
    }
    #endif
    const int32_t bos = tokenizer.get_bos();
    const int32_t eos = tokenizer.get_eos();
    const int32_t pad = tokenizer.get_pad();
    const std::string bos_piece = tokenizer.token_to_piece(bos);
    const std::string eos_piece = tokenizer.token_to_piece(eos);
    const std::string pad_piece = tokenizer.token_to_piece(pad);
    CW_LOG_I(
        "bos: %d = %s eos: %d = %s pad: %d = %s",
        bos, bos_piece.c_str(),
        eos, eos_piece.c_str(),
        pad, pad_piece.c_str()
    );
    assert(bos == tokenizer.piece_to_token(bos_piece));
    assert(eos == tokenizer.piece_to_token(eos_piece));
    assert(pad == tokenizer.piece_to_token(pad_piece));
    const int bthinking = tokenizer.piece_to_token("<think>");
    const int ethinking = tokenizer.piece_to_token("</think>");
    const int btoolcall = tokenizer.piece_to_token("<tool_call>");
    const int etoolcall = tokenizer.piece_to_token("</tool_call>");
    assert(bthinking == 151667);
    assert(ethinking == 151668);
    assert(btoolcall == 151657);
    assert(etoolcall == 151658);
    const std::string piece = tokenizer.token_to_piece(151644);
    assert(piece == "<|im_start|>");
    const std::string prompt = "<|im_start|>你好这是一个测试<think>think</think><tool_call>tool_call</tool_call><|im_end|>";
    std::vector<int32_t> tokens(64);
    const int length = tokenizer.tokenize(prompt.data(), static_cast<int32_t>(prompt.size()), tokens.data(), static_cast<int32_t>(tokens.size()));
    assert(length > 0);
    const std::string text = tokenizer.detokenize(tokens.data(), length);
    assert(text == "你好这是一个测试<think>think</think><tool_call>tool_call</tool_call>");
    CW_LOG_I("text: %s", tokenizer.detokenize(tokens.data(), length).c_str());
    CW_LOG_I("text: %s", tokenizer.detokenize(tokens.data(), length, true, false).c_str());
    CW_LOG_I("text: %s", tokenizer.detokenize(tokens.data(), length, false, true).c_str());
    CW_LOG_I("text: %s", tokenizer.detokenize(tokens.data(), length, true, true).c_str());
}

[[maybe_unused]]
void test_chat_template() {
    caiwei::text::ChatTemplate chat_template;
    chat_template.set_template("{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {%- if messages[0].content is string %}\n            {{- messages[0].content }}\n        {%- else %}\n            {%- for content in messages[0].content %}\n                {%- if 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' }}\n        {%- if messages[0].content is string %}\n            {{- messages[0].content }}\n        {%- else %}\n            {%- for content in messages[0].content %}\n                {%- if 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set image_count = namespace(value=0) %}\n{%- set video_count = namespace(value=0) %}\n{%- for message in messages %}\n    {%- if message.role == \"user\" %}\n        {{- '<|im_start|>' + message.role + '\\n' }}\n        {%- if message.content is string %}\n            {{- message.content }}\n        {%- else %}\n            {%- for content in message.content %}\n                {%- if content.type == 'image' or 'image' in content or 'image_url' in content %}\n                    {%- set image_count.value = image_count.value + 1 %}\n                    {%- if add_vision_id %}Picture {{ image_count.value }}: {% endif -%}\n                    <|vision_start|><|image_pad|><|vision_end|>\n                {%- elif content.type == 'video' or 'video' in content %}\n                    {%- set video_count.value = video_count.value + 1 %}\n                    {%- if add_vision_id %}Video {{ video_count.value }}: {% endif -%}\n                    <|vision_start|><|video_pad|><|vision_end|>\n                {%- elif 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role + '\\n' }}\n        {%- if message.content is string %}\n            {{- message.content }}\n        {%- else %}\n            {%- for content_item in message.content %}\n                {%- if 'text' in content_item %}\n                    {{- content_item.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and message.content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {%- if message.content is string %}\n            {{- message.content }}\n        {%- else %}\n            {%- for content in message.content %}\n                {%- if content.type == 'image' or 'image' in content or 'image_url' in content %}\n                    {%- set image_count.value = image_count.value + 1 %}\n                    {%- if add_vision_id %}Picture {{ image_count.value }}: {% endif -%}\n                    <|vision_start|><|image_pad|><|vision_end|>\n                {%- elif content.type == 'video' or 'video' in content %}\n                    {%- set video_count.value = video_count.value + 1 %}\n                    {%- if add_vision_id %}Video {{ video_count.value }}: {% endif -%}\n                    <|vision_start|><|video_pad|><|vision_end|>\n                {%- elif 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n", "", "");
    caiwei::text::CompletionsRequest request;
    request.enable_thinking = false;
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
        caiwei::text::CompletionsRequestMessageContentItem {
            .type  = "text",
            .text = "测试"
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
    const std::string prompt = chat_template.apply(special_token, request, "<audio>", "<image>", "<video>");
    CW_LOG_I("prompt: %s", prompt.c_str());
    CAIWEI_FOR_EACH(1000)
    chat_template.apply(special_token, request, "<audio>", "<image>", "<video>");
    CAIWEI_FOR_EACH_END
}

[[maybe_unused]] void print_name_arguments(caiwei::text::ToolcallSession& session) {
    std::string name = session.get_name();
    CW_LOG_I("name: %s", name.c_str());
    std::string arguments = session.get_arguments();
    CW_LOG_I("arguments: %s", arguments.c_str());
}

[[maybe_unused]]
void test_tool_call() {
    caiwei::text::ToolcallSession session;
    // -
    session.put_token(R"({"name": "get_weather", "arguments": {"city":"北京"}})");
    print_name_arguments(session);
    session.finish();
    print_name_arguments(session);
    // -
    session.reset();
    session.put_token(R"({"name":"get_weather","arguments":{"city":"北京"}})");
    print_name_arguments(session);
    session.finish();
    print_name_arguments(session);
    // -
    session.reset();
    session.put_token(R"({"arguments":{"city":"北京"},"name":"get_weather"})");
    print_name_arguments(session);
    session.finish();
    print_name_arguments(session);
    // -
    session.reset();
    session.put_token(R"({"nam)");
    print_name_arguments(session);
    session.put_token(R"(e": "g)");
    print_name_arguments(session);
    session.put_token(R"(et_weather", "arg)");
    print_name_arguments(session);
    session.put_token(R"(uments": {"city":")");
    print_name_arguments(session);
    session.put_token(R"(北)");
    print_name_arguments(session);
    session.put_token(R"(京)");
    print_name_arguments(session);
    session.put_token(R"("}})");
    session.finish();
    print_name_arguments(session);
    // -
    session.reset();
    session.put_token(R"({"nam)");
    print_name_arguments(session);
    session.put_token(R"(e":"g)");
    print_name_arguments(session);
    session.put_token(R"(et_weather","arg)");
    print_name_arguments(session);
    session.put_token(R"(uments":{"city":")");
    print_name_arguments(session);
    session.put_token(R"(北京"}})");
    session.finish();
    print_name_arguments(session);
}

int main() {
    caiwei::test::init_test();
    // test_tokenizer();
    // test_chat_template();
    test_tool_call();
    caiwei::test::stop_test();
    return 0;
}
