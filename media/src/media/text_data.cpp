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


std::string caiwei::text::completions_response(const CompletionsRequest& request, const std::string& finish_reason, std::string content, std::string thinking, std::string toolcall) {
    return "";
}

std::string caiwei::text::completions_chunk(const CompletionsRequest& request, const Result& result) {
    CompletionsChunk chunk;
    chunk.id      = request.id;
    chunk.model   = request.model;
    chunk.created = request.created;
    CompletionsChunkChoice choice;
    choice.index         = request.index;
    choice.finish_reason = result.finish_reason;
    choice.delta.role    = caiwei::text::ROLE_ASSISTANT;
    if (result.thinking) {
        choice.delta.reasoning_content = result.token;
    } else if (result.toolcall) {
        CompletionsChunkChoiceMessageToolCall tool_call;
        CompletionsChunkChoiceMessageToolCallFunction function;
        function.name      = result.result_toolcall->get_name();
        function.arguments = result.result_toolcall->get_arguments();
        tool_call.id    = result.result_toolcall->toolcall_id;
        tool_call.type  = "function";
        tool_call.index = result.result_toolcall->toolcall_index;
        tool_call.function = function;
        if (function.name.value_or("").empty() && function.arguments.value_or("").empty()) {
            return "";
        } else {
            return function.name.value_or("") + function.arguments.value_or("");
        }
    } else {
        choice.delta.content = result.token;
    }
    chunk.choices.push_back(choice);
    if (!result.finish_reason.empty()) {
        chunk.usage = CompletionsChunkUsage {
            .prompt_tokens     = result.prompt_tokens,
            .completion_tokens = result.completion_tokens,
            .total_tokens      = result.total_tokens,
        };
    }
    return result.token;
}

caiwei::text::EmbeddingRequest caiwei::text::json_to_embedding(const std::string& json) {
    EmbeddingRequest request;
    nlohmann::json data = nlohmann::json::parse(json);
    request.model = data.value("model", "");
    caiwei::env::check_empty(request.model, "模型不能为空");
    nlohmann::json input_json = data.value("input", nlohmann::json());
    if (input_json.is_string()) {
        std::string input = input_json.get<std::string>();
        caiwei::env::check_empty(input, "输入不能为空");
        request.input = input;
    } else if (input_json.is_array()) {
        std::vector<std::string> input = input_json.get<std::vector<std::string>>();
        caiwei::env::check_bool(input.empty(), "输入不能为空");
        request.input = input;
    } else {
        throw caiwei::env::MessageCodeException("输入格式错误");
    }
    return request;
}

caiwei::text::RerankingRequest caiwei::text::json_to_reranking(const std::string& json) {
    caiwei::text::RerankingRequest request;
    nlohmann::json data = nlohmann::json::parse(json);
    request.model = data.value("model", "");
    caiwei::env::check_empty(request.model, "模型不能为空");
    request.query = data.value("query", "");
    caiwei::env::check_empty(request.query, "查询不能为空");
    nlohmann::json documents_json = data.value("documents", nlohmann::json());
    if (documents_json.is_array()) {
        std::vector<std::string> documents = documents_json.get<std::vector<std::string>>();
        caiwei::env::check_bool(documents.empty(), "文档不能为空");
        request.documents = documents;
    } else {
        throw caiwei::env::MessageCodeException("文档格式错误");
    }
    request.instruct = data.value("instruct", "");
    return request;
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
