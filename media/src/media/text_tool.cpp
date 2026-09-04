#include "caiwei/text_tool.hpp"

#include "nlohmann/json.hpp"
#include "minja/chat-template.hpp"

caiwei::text::ChatTemplate::ChatTemplate() {
}

caiwei::text::ChatTemplate::~ChatTemplate() {
}

void caiwei::text::ChatTemplate::set_template(const std::string& template_text, const std::string& bos, const std::string& eos) {
    chat_template = std::make_unique<minja::chat_template>(template_text, bos, eos);
}

std::string caiwei::text::ChatTemplate::apply(const SpecialToken& special_token, const caiwei::text::CompletionsRequest& request) {
    nlohmann::json messages;
    nlohmann::json tools;
    nlohmann::json extra_context;
    if (!special_token.enable_thinking.empty()) {
        extra_context[special_token.enable_thinking] = request.enable_thinking.value_or(true);
    }
    for (const auto& message : request.messages) {
        nlohmann::json copy;
        if (message.role.has_value()) {
            copy["role"] = message.role.value_or("");
        }
        if (message.tool_call_id.has_value()) {
            copy["tool_call_id"] = message.tool_call_id.value_or("");
        }
        if (message.content.has_value()) {
            if (std::holds_alternative<std::string>(message.content.value())) {
                copy["content"] = std::get<std::string>(message.content.value());
            } else if (std::holds_alternative<std::vector<caiwei::text::CompletionsRequestMessageContentItem>>(message.content.value())) {
                const auto& items = std::get<std::vector<caiwei::text::CompletionsRequestMessageContentItem>>(message.content.value());
                nlohmann::json parts;
                for (const auto& item : items) {
                    if (item.type.has_value()) {
                        nlohmann::json part;
                        part["type"] = item.type.value_or("");
                        if (part["type"] == "text") {
                            part["text"] = item.text.value_or("");
                        }
                        parts.push_back(part);
                    }
                }
                copy["content"] = parts;
            }
        }
        if (message.tool_calls.has_value()) {
            const auto& items = message.tool_calls.value();
            nlohmann::json tool_calls;
            for (const auto& item : items) {
                nlohmann::json call;
                call["id"] = item.id.value_or("");
                if (item.function.has_value()) {
                    call["name"] = item.function.value().name.value_or("");
                    call["arguments"] = item.function.value().arguments.value_or("");
                }
                tool_calls.push_back(call);
            }
            copy["tool_calls"] = tool_calls;
        }
        messages.push_back(copy);
    }
    if (request.tools.has_value()) {
        const auto& items = request.tools.value();
        for (const auto& item : items) {
            nlohmann::json tool;
            if (item.function.has_value()) {
                tool["name"] = item.function.value().name.value_or("");
                tool["parameters"] = item.function.value().parameters.value_or(nlohmann::json());
                tool["description"] = item.function.value().description.value_or("");
            }
            tools.push_back(std::move(tool));
        }
    }
    return this->chat_template->apply(minja::chat_template_inputs{
        .messages              = messages,
        .tools                 = tools,
        .add_generation_prompt = true,
        .extra_context         = extra_context
    });
}

std::string caiwei::text::chunk_choice(const CompletionsRequest& request, const Result& result) {
    CompletionsChunk chunk;
    chunk.id      = request.id;
    chunk.model   = request.model;
    chunk.created = request.created;
    CompletionsChunkChoice choice;
    choice.index         = request.index;
    choice.finish_reason = result.finish_reason;
    choice.delta.role    = "assistant";
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

std::string caiwei::text::response_choice(const CompletionsRequest& request, const std::string& finish_reason, std::string content, std::string thinking, std::string toolcall) {
    return "";
}
