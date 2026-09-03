#include "caiwei/text.hpp"

#include "nlohmann/json.hpp"
#include "minja/chat-template.hpp"

caiwei::text::ChatTemplate::ChatTemplate() {
}

caiwei::text::ChatTemplate::~ChatTemplate() {
}

void caiwei::text::ChatTemplate::set_template(const std::string& template_text, const std::string& bos, const std::string& eos) {
    chat_template = std::make_unique<minja::chat_template>(template_text, bos, eos);
}

std::string caiwei::text::ChatTemplate::apply(const caiwei::text::CompletionsRequest& request) {
    nlohmann::json messages;
    nlohmann::json tools;
    nlohmann::json extra_context;
    // TODO 模型是否支持
    // TODO 使用配置变量
    if (request.enable_thinking.value_or(false)) {
        extra_context["enable_thinking"] = true;
    } else {
        extra_context["enable_thinking"] = false;
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
                for (const auto& item : items) {
                    nlohmann::json parts;
                    if (item.type.has_value()) {
                        nlohmann::json part;
                        part["type"] = item.type.value_or("");
                        if (part["type"] == "text") {
                            part["text"] = item.text.value_or("");
                        }
                        parts.push_back(part);
                    }
                    copy["content"] = parts;
                }
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
