#include "caiwei/env.hpp"
#include "caiwei/text_data.hpp"

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

void caiwei::text::ResultToolcall::reset() {
    this->toolcall_id    =  caiwei::env::id();
    this->toolcall_index += 1;
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

template<typename T>
inline static void set_opt(const nlohmann::json& data, std::optional<T>& opt, const std::string& key) {
    if (auto iter = data.find(key); iter != data.end()) {
        opt = iter->get<T>();
    }
}

namespace caiwei {
namespace text   {

void from_json(const nlohmann::json& json, CompletionsRequestMessageContentItemUrl& url) {
    set_opt(json, url.url, "url");
}

void from_json(const nlohmann::json& json, CompletionsRequestMessageContentItem& item) {
    set_opt(json, item.type, "type");
    set_opt(json, item.text, "text");
    set_opt(json, item.audio, "audio");
    set_opt(json, item.image, "image");
    set_opt(json, item.video, "video");
    set_opt(json, item.audio_url, "audio_url");
    set_opt(json, item.image_url, "image_url");
    set_opt(json, item.video_url, "video_url");
}

void from_json(const nlohmann::json& json, CompletionsRequestMessageContent& content) {
    if (json.is_string()) {
        content = json.get<std::string>();
    } else if (json.is_array()) {
        std::vector<CompletionsRequestMessageContentItem> vector = json;
        content = std::move(vector);
    } else {
        throw caiwei::env::MessageCodeException("输入格式错误");
    }
}

void from_json(const nlohmann::json& json, CompletionsRequestMessageToolCallFunction& function) {
    set_opt(json, function.name, "name");
    set_opt(json, function.arguments, "arguments");
}

void from_json(const nlohmann::json& json, CompletionsRequestMessageToolCall& tool_call) {
    set_opt(json, tool_call.id, "id");
    set_opt(json, tool_call.type, "type");
    set_opt(json, tool_call.function, "function");
}

void from_json(const nlohmann::json& json, CompletionsRequestMessage& message) {
    set_opt(json, message.role, "role");
    set_opt(json, message.name, "name");
    set_opt(json, message.tool_call_id, "tool_call_id");
    set_opt(json, message.content, "content");
    set_opt(json, message.tool_calls, "tool_calls");
}

void from_json(const nlohmann::json& json, CompletionsRequestToolFunction& function) {
    set_opt(json, function.name, "name");
    set_opt(json, function.description, "description");
    set_opt(json, function.parameters, "parameters");
}

void from_json(const nlohmann::json& json, CompletionsRequestTool& tool) {
    set_opt(json, tool.type, "type");
    set_opt(json, tool.function, "function");
}

void from_json(const nlohmann::json& json, CompletionsRequestExtraBody& extra_body) {
    set_opt(json, extra_body.video_fps, "video_fps");
    set_opt(json, extra_body.asr_frames, "asr_frames");
    set_opt(json, extra_body.vlm_frames, "vlm_frames");
    set_opt(json, extra_body.audio_queue_size, "audio_queue_size");
    set_opt(json, extra_body.video_queue_size, "video_queue_size");
    set_opt(json, extra_body.media_url, "media_url");
    set_opt(json, extra_body.media_type, "media_type");
    set_opt(json, extra_body.enable_thinking, "enable_thinking");
    set_opt(json, extra_body.model_list, "model_list");
}

}
}

caiwei::text::CompletionsRequest caiwei::text::json_to_completions(const std::string& json) {
    caiwei::text::CompletionsRequest request;
    nlohmann::json data = nlohmann::json::parse(json);
    request.stream = data.value("stream", false);
    request.model  = data.value("model", "");
    caiwei::env::check_empty(request.model, "模型不能为空");
    request.messages = data.value("messages", std::vector<CompletionsRequestMessage>());
    caiwei::env::check_bool(request.messages.empty(), "消息不能为空");
    set_opt(data, request.seed, "seed");
    set_opt(data, request.top_k, "top_k");
    set_opt(data, request.top_p, "top_p");
    set_opt(data, request.temperature, "temperature");
    set_opt(data, request.repeat_penalty, "repeat_penalty");
    set_opt(data, request.presence_penalty, "presence_penalty");
    set_opt(data, request.frequency_penalty, "frequency_penalty");
    set_opt(data, request.max_completion_tokens, "max_completion_tokens");
    set_opt(data, request.tools, "tools");
    set_opt(data, request.extra_body, "extra_body");
    request.id      = caiwei::env::id();
    request.created = caiwei::env::unix_timestamp();
    return request;
}

caiwei::text::EmbeddingsRequest caiwei::text::json_to_embeddings(const std::string& json) {
    EmbeddingsRequest request;
    nlohmann::json data = nlohmann::json::parse(json);
    request.model = data.value("model", "");
    caiwei::env::check_empty(request.model, "模型不能为空");
    nlohmann::json input_json = data.value("input", nlohmann::json());
    if (input_json.is_string()) {
        std::string input = input_json.get<std::string>();
        caiwei::env::check_empty(input, "输入不能为空");
        request.input = std::move(input);
    } else if (input_json.is_array()) {
        std::vector<std::string> input = input_json.get<std::vector<std::string>>();
        caiwei::env::check_bool(input.empty(), "输入不能为空");
        request.input = std::move(input);
    } else {
        throw caiwei::env::MessageCodeException("输入格式错误");
    }
    return request;
}

caiwei::text::RerankingsRequest caiwei::text::json_to_rerankings(const std::string& json) {
    caiwei::text::RerankingsRequest request;
    nlohmann::json data = nlohmann::json::parse(json);
    request.model = data.value("model", "");
    caiwei::env::check_empty(request.model, "模型不能为空");
    request.query = data.value("query", "");
    caiwei::env::check_empty(request.query, "查询不能为空");
    nlohmann::json documents_json = data.value("documents", nlohmann::json());
    if (documents_json.is_array()) {
        request.documents = documents_json.get<std::vector<std::string>>();
        caiwei::env::check_bool(request.documents.empty(), "文档不能为空");
    } else {
        throw caiwei::env::MessageCodeException("文档格式错误");
    }
    request.instruct = data.value("instruct", "");
    return request;
}

namespace caiwei {
namespace text   {

template<typename T>
void field_to_json(nlohmann::json& json, const char* key, const T& val) {
    json[key] = val;
}

template<typename T>
void field_to_json(nlohmann::json& json, const char* key, const std::optional<T>& val) {
    if (val.has_value()) {
        json[key] = val.value();
    }
}

void to_json(nlohmann::json& json, const CompletionsResponseChoiceMessageToolCallFunction& function) {
    field_to_json(json, "name",      function.name);
    field_to_json(json, "arguments", function.arguments);
}

void to_json(nlohmann::json& json, const CompletionsResponseChoiceMessageToolCall& tool_call) {
    field_to_json(json, "id",       tool_call.id);
    field_to_json(json, "type",     tool_call.type);
    field_to_json(json, "index",    tool_call.index);
    field_to_json(json, "function", tool_call.function);
}

void to_json(nlohmann::json& json, const CompletionsResponseChoiceMessage& message) {
    field_to_json(json, "role",              message.role);
    field_to_json(json, "content",           message.content);
    field_to_json(json, "refusal",           message.refusal);
    field_to_json(json, "reasoning_content", message.reasoning_content);
    field_to_json(json, "tool_calls",        message.tool_calls);
}

void to_json(nlohmann::json& json, const CompletionsResponseChoice& choice) {
    field_to_json(json, "index",         choice.index);
    field_to_json(json, "finish_reason", choice.finish_reason);
    field_to_json(json, "message",       choice.message);
}

void to_json(nlohmann::json& json, const CompletionsResponseUsage& usage) {
    field_to_json(json, "prompt_tokens",     usage.prompt_tokens);
    field_to_json(json, "completion_tokens", usage.completion_tokens);
    field_to_json(json, "total_tokens",      usage.total_tokens);
}

void to_json(nlohmann::json& json, const CompletionsChunkChoiceMessageToolCallFunction& function) {
    field_to_json(json, "name",      function.name);
    field_to_json(json, "arguments", function.arguments);
}

void to_json(nlohmann::json& json, const CompletionsChunkChoiceMessageToolCall& tool_call) {
    field_to_json(json, "id",       tool_call.id);
    field_to_json(json, "type",     tool_call.type);
    field_to_json(json, "index",    tool_call.index);
    field_to_json(json, "function", tool_call.function);
}

void to_json(nlohmann::json& json, const CompletionsChunkDelta& delta) {
    field_to_json(json, "role",              delta.role);
    field_to_json(json, "content",           delta.content);
    field_to_json(json, "refusal",           delta.refusal);
    field_to_json(json, "reasoning_content", delta.reasoning_content);
    field_to_json(json, "tool_calls",        delta.tool_calls);
}

void to_json(nlohmann::json& json, const CompletionsChunkChoice& choice) {
    field_to_json(json, "index",         choice.index);
    field_to_json(json, "finish_reason", choice.finish_reason);
    field_to_json(json, "delta",         choice.delta);
}

void to_json(nlohmann::json& json, const CompletionsChunkUsage& usage) {
    field_to_json(json, "prompt_tokens",     usage.prompt_tokens);
    field_to_json(json, "completion_tokens", usage.completion_tokens);
    field_to_json(json, "total_tokens",      usage.total_tokens);
}

void to_json(nlohmann::json& json, const EmbeddingResponseData& data) {
    field_to_json(json, "index",     data.index);
    field_to_json(json, "object",    data.object);
    field_to_json(json, "embedding", data.embedding);
}

void to_json(nlohmann::json& json, const EmbeddingResponseUsage& usage) {
    field_to_json(json, "prompt_tokens", usage.prompt_tokens);
    field_to_json(json, "total_tokens",  usage.total_tokens);
}

void to_json(nlohmann::json& json, const RerankingResponseData& data) {
    field_to_json(json, "index",  data.index);
    field_to_json(json, "object", data.object);
    field_to_json(json, "score",  data.score);
}

void to_json(nlohmann::json& json, const RerankingResponseUsage& usage) {
    field_to_json(json, "prompt_tokens", usage.prompt_tokens);
    field_to_json(json, "total_tokens",  usage.total_tokens);
}

}
}

std::string caiwei::text::to_json(const CompletionsResponse& response) {
    nlohmann::json json;
    json["created"] = response.created;
    json["id"]      = response.id;
    json["model"]   = response.model;
    json["object"]  = response.object;
    json["choices"] = response.choices;
    if (response.usage.has_value()) {
        json["usage"] = response.usage.value();
    }
    return json.dump();
}

std::string caiwei::text::to_json(const CompletionsChunk& chunk) {
    nlohmann::json json;
    json["created"] = chunk.created;
    json["id"]      = chunk.id;
    json["model"]   = chunk.model;
    json["object"]  = chunk.object;
    json["choices"] = chunk.choices;
    if (chunk.usage.has_value()) {
        json["usage"] = chunk.usage.value();
    }
    return json.dump();
}

std::string caiwei::text::to_json(const EmbeddingResponse& response) {
    nlohmann::json json;
    json["object"] = response.object;
    json["model"]  = response.model;
    json["data"]   = response.data;
    if (response.usage.has_value()) {
        json["usage"] = response.usage.value();
    }
    return json.dump();
}

std::string caiwei::text::to_json(const RerankingResponse& response) {
    nlohmann::json json;
    json["object"] = response.object;
    json["model"]  = response.model;
    json["data"]   = response.data;
    if (response.usage.has_value()) {
        json["usage"] = response.usage.value();
    }
    return json.dump();
}
