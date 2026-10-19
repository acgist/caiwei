#include "caiwei/log.hpp"
#include "caiwei/text_tool.hpp"

#include "nlohmann/json.hpp"
#include "minja/chat-template.hpp"

std::string caiwei::text::get_reranking_prompt(const std::string& document, const caiwei::text::RerankingsRequest& request, const caiwei::text::SpecialToken& special_token) {
    std::string prompt;
    prompt
        .append(special_token.bos)
        .append("system\n")
        .append(special_token.system_prompt)
        .append(special_token.eos)
        .append("\n")
        .append(special_token.bos)
        .append("user\n")
        .append(special_token.instruction_key)
        .append(request.instruct.value_or(special_token.instruction_prompt))
        .append("\n")
        .append(special_token.query_key)
        .append(request.query)
        .append("\n")
        .append(special_token.document_key)
        .append(document)
        .append("\n")
        .append(special_token.eos)
        .append("\n")
        .append(special_token.bos)
        .append("assistant\n")
        .append(special_token.b_thinking)
        .append("\n\n")
        .append(special_token.e_thinking)
        .append("\n\n");
    return prompt;
}

caiwei::text::ChatTemplate::ChatTemplate() {
}

caiwei::text::ChatTemplate::~ChatTemplate() {
}

void caiwei::text::ChatTemplate::set_template(const std::string& template_text, const std::string& bos, const std::string& eos) {
    CW_LOG_I("set_template: %s", template_text.c_str());
    if (template_text.empty()) {
        return;
    }
    this->chat_template = std::make_unique<minja::chat_template>(template_text, bos, eos);
}

std::string caiwei::text::ChatTemplate::apply(
    const SpecialToken& special_token,
    const caiwei::text::CompletionsRequest& request,
    const std::string& audio_marker,
    const std::string& image_marker,
    const std::string& video_marker,
    bool add_vision_id
) const {
    nlohmann::json messages;
    nlohmann::json tools;
    nlohmann::json extra_context;
    if (!special_token.enable_thinking.empty()) {
        extra_context[special_token.enable_thinking] = request.enable_thinking.value_or(true);
    }
    if (add_vision_id) {
        extra_context["add_vision_id"] = true;
    }
    for (const auto& message : request.messages) {
        nlohmann::json copy;
        if (message.role.has_value()) {
            copy["role"] = message.role.value_or("");
        }
        if (message.tool_call_id.has_value()) {
            copy["tool_call_id"] = message.tool_call_id.value_or("");
        }
        int audio_count = 0;
        int image_count = 0;
        int video_count = 0;
        if (message.content.has_value()) {
            if (std::holds_alternative<std::string>(message.content.value())) {
                copy["content"] = std::get<std::string>(message.content.value());
            } else if (std::holds_alternative<std::vector<caiwei::text::CompletionsRequestMessageContentItem>>(message.content.value())) {
                nlohmann::json parts;
                const auto& items = std::get<std::vector<caiwei::text::CompletionsRequestMessageContentItem>>(message.content.value());
                int audio_index = 0;
                int image_index = 0;
                int video_index = 0;
                for (const auto& item : items) {
                    if (item.type.has_value()) {
                        nlohmann::json part;
                        const std::string& type = item.type.value_or("");
                        // part["type"] = type;
                        // 强制修改类型
                        part["type"] = "text";
                        if (type == "text") {
                            part["text"] = item.text.value_or("");
                        } else if (type == "audio" || type == "audio_url") {
                            audio_index++;
                            audio_count++;
                            part["text"] = audio_marker;
                        } else if (type == "image" || type == "image_url") {
                            image_index++;
                            image_count++;
                            std::string text;
                            if (add_vision_id) {
                                text += "Picture ";
                                text += std::to_string(image_count);
                                text += ": ";
                            }
                            text += image_marker;
                            part["text"] = std::move(text);
                        } else if (type == "video" || type == "video_url") {
                            video_index++;
                            video_count++;
                            std::string text;
                            if (add_vision_id) {
                                text += "Video ";
                                text += std::to_string(video_count);
                                text += ": ";
                            }
                            text += video_marker;
                            part["text"] = std::move(text);
                        } else {
                            CW_LOG_W("不支持的媒体类型: %s", type.c_str());
                        }
                        parts.emplace_back(std::move(part));
                    }
                }
                if (audio_index < message.audio_data.size()) {
                    for (int i = audio_index; i < message.audio_data.size(); i++) {
                        parts.emplace_back(nlohmann::json({
                            {"type", "text"},
                            {"text", audio_marker}
                        }));
                    }
                }
                if (image_index < message.image_data.size()) {
                    for (int i = image_index; i < message.image_data.size(); i++) {
                        parts.emplace_back(nlohmann::json({
                            {"type", "text"},
                            {"text", image_marker}
                        }));
                    }
                }
                if (video_index < message.video_data.size()) {
                    for (int i = video_index; i < message.video_data.size(); i++) {
                        parts.emplace_back(nlohmann::json({
                            {"type", "text"},
                            {"text", video_marker}
                        }));
                    }
                }
                copy["content"] = std::move(parts);
            } else {
                // -
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
