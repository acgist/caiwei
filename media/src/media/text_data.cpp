#include "caiwei/env.hpp"
#include "caiwei/text_data.hpp"

#include "nlohmann/json.hpp"

caiwei::text::Result::Result(bool thinking, bool toolcall, std::string token)
  : thinking(thinking)
  , toolcall(toolcall)
  , token(token) {
}

caiwei::text::Result::Result(bool thinking, bool toolcall, ResultToolcall* result_toolcall)
  : thinking(thinking)
  , toolcall(toolcall)
  , token(token)
  , result_toolcall(result_toolcall) {
}

caiwei::text::Result::Result(bool thinking, bool toolcall, std::string finish_reason, uint32_t prompt_tokens, uint32_t completion_tokens)
  : thinking(thinking)
  , toolcall(toolcall)
  , finish_reason(finish_reason)
  , prompt_tokens(prompt_tokens)
  , completion_tokens(completion_tokens) {
    this->total_tokens = this->prompt_tokens + this->completion_tokens;
}

caiwei::text::ResultToolcall::ResultToolcall() : toolcall_id(caiwei::env::id()), toolcall_index(0) {
    this->content.reserve(1024);
}

void caiwei::text::ResultToolcall::finish() {
    this->arguments += this->token;
    auto pos = this->arguments.rfind("}");
    if (pos != std::string::npos) {
        this->arguments = this->arguments.substr(0, pos);
    } else {
        // 异常情况
    }
    this->token.clear();
}

void caiwei::text::ResultToolcall::increment() {
    this->toolcall_id    =  caiwei::env::id();
    this->toolcall_index += 1;
}

void caiwei::text::ResultToolcall::put_token(std::string token) {
    this->token = token;
    if (this->name_return && this->arguments_return) {
        return;
    }
    this->content += token;
}

std::string caiwei::text::ResultToolcall::get_name() {
    if (this->name_return) {
        return "";
    }
    auto a = this->content.find(R"("name": ")");
    auto z = this->content.find(R"(", "arguments": )");
    if (a != std::string::npos && z != std::string::npos) {
        this->name_return = true;
        return this->content.substr(a + 9, z - a - 9);
    }
    return "";
}

std::string caiwei::text::ResultToolcall::get_arguments() {
    if (!this->name_return) {
        return "";
    }
    if (this->arguments_return) {
        this->token.swap(this->arguments);
        return std::move(this->token);
    }
    auto pos = this->content.find(R"("arguments": )");
    if (pos != std::string::npos) {
        this->arguments_return = true;
        return this->content.substr(pos + 13);
    }
    return "";
}

// TODO 需要手写

// void to_json(json& j, const CompletionsRequestToolChoiceVariant& v) {
//     std::visit([&](auto&& arg){ j = arg; }, v);
// }

// void from_json(const json& j, CompletionsRequestToolChoiceVariant& v) {
//     if(j.is_string()) {
//         v = j.get<std::string>();
//     } else if(j.is_object()) {
//         v = j.get<CompletionsRequestToolChoice>();
//     } else {
//         throw json::type_error::create(301, R"(tool_choice must be string("auto"/"none"/"required") or object)", j);
//     }
// }


// void to_json(json& j, const CompletionsRequest& v) {
//     j = json::object();
//     j["stream"] = v.stream;
//     j["top_p"] = v.top_p;
//     j["temperature"] = v.temperature;
//     j["presence_penalty"] = v.presence_penalty;
//     j["frequency_penalty"] = v.frequency_penalty;
//     j["n"] = v.n;
//     j["max_tokens"] = v.max_tokens;
//     j["model"] = v.model;

//     if(v.thinking) j["thinking"] = *v.thinking;
//     if(v.stop) j["stop"] = *v.stop;
//     if(v.stream_options) j["stream_options"] = *v.stream_options;
//     if(v.response_format) j["response_format"] = *v.response_format;
//     if(v.tool_choice) j["tool_choice"] = *v.tool_choice;
//     if(v.tools) j["tools"] = *v.tools;

//     j["messages"] = v.messages;

//     if(v.parallel_tool_calls) j["parallel_tool_calls"] = *v.parallel_tool_calls;
//     if(v.seed) j["seed"] = *v.seed;
//     if(v.user) j["user"] = *v.user;
// }

// void from_json(const json& j, CompletionsRequest& v) {
//     // 默认值已经在构造函数，json里不存在则保留默认
//     if(j.contains("stream")) j["stream"].get_to(v.stream);
//     if(j.contains("top_p")) j["top_p"].get_to(v.top_p);
//     if(j.contains("temperature")) j["temperature"].get_to(v.temperature);
//     if(j.contains("presence_penalty")) j["presence_penalty"].get_to(v.presence_penalty);
//     if(j.contains("frequency_penalty")) j["frequency_penalty"].get_to(v.frequency_penalty);
//     if(j.contains("n")) j["n"].get_to(v.n);
//     if(j.contains("max_tokens")) j["max_tokens"].get_to(v.max_tokens);
//     j.at("model").get_to(v.model);

//     if(j.contains("thinking")) v.thinking = j["thinking"].get<std::string>();
//     if(j.contains("stop")) v.stop = j["stop"].get<CompletionsRequestStop>();
//     if(j.contains("stream_options")) v.stream_options = j["stream_options"].get<CompletionsRequestStreamOptions>();
//     if(j.contains("response_format")) v.response_format = j["response_format"].get<CompletionsRequestResponseFormat>();
//     if(j.contains("tool_choice")) v.tool_choice = j["tool_choice"].get<CompletionsRequestToolChoiceVariant>();
//     if(j.contains("tools")) v.tools = j["tools"].get<std::vector<CompletionsRequestTool>>();

//     j.at("messages").get_to(v.messages);

//     if(j.contains("parallel_tool_calls")) v.parallel_tool_calls = j["parallel_tool_calls"].get<bool>();
//     if(j.contains("seed")) v.seed = j["seed"].get<int64_t>();
//     if(j.contains("user")) v.user = j["user"].get<std::string>();
// }
