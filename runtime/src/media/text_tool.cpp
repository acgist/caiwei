#include "caiwei/log.hpp"
#include "caiwei/text_tool.hpp"

#include "nlohmann/json.hpp"

#include "minja/chat-template.hpp"

std::string caiwei::text::get_reranking_prompt(
    const std::string& document,
    const caiwei::text::RerankingsRequest& request,
    const caiwei::text::SpecialToken& special_token
) {
    std::string instruct = request.instruct.value_or(special_token.instruction_prompt);
    if (instruct.empty()) {
        instruct = special_token.instruction_prompt;
    }
    std::string prompt;
    prompt
        .append(special_token.bos)
        .append(caiwei::text::ROLE_SYSTEM)
        .append("\n")
        .append(special_token.system_prompt)
        .append(special_token.eos)
        .append("\n")
        .append(special_token.bos)
        .append(caiwei::text::ROLE_USER)
        .append("\n")
        .append(special_token.instruction_key)
        .append(instruct)
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
        .append(caiwei::text::ROLE_ASSISTANT)
        .append("\n")
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

void caiwei::text::ChatTemplate::reset() {
    this->chat_template = nullptr;
}

bool caiwei::text::ChatTemplate::set_template(const char* template_text, const std::string& bos, const std::string& eos) {
    if (template_text == nullptr || std::strlen(template_text) == 0) {
        CW_LOG_W("模板为空");
        return false;
    }
    CW_LOG_D("设置模板: %s - %s -> %s", bos.c_str(), eos.c_str(), template_text);
    try {
        this->chat_template = std::make_unique<minja::chat_template>(template_text, bos, eos);
    } catch (...) {
        CW_LOG_E("设置模板异常: %s", template_text);
    }
    return this->chat_template != nullptr;
}

std::string caiwei::text::ChatTemplate::apply(
    const SpecialToken& special_token,
    const caiwei::text::CompletionsRequest& request,
    const std::string& audio_marker,
    const std::string& image_marker,
    const std::string& video_marker,
    bool add_vision_id
) const {
    if (this->chat_template == nullptr) {
        CW_LOG_W("模板为空");
        return "";
    }
    nlohmann::json messages_json;
    nlohmann::json tools_json;
    nlohmann::json extra_context_json;
    int audio_count = 0;
    int image_count = 0;
    int video_count = 0;
    for (const auto& message : request.messages) {
        nlohmann::json message_json;
        message_json["role"] = message.role.value_or(caiwei::text::ROLE_USER);
        if (message.name.has_value()) {
            message_json["name"] = message.name.value_or("");
        }
        if (message.tool_call_id.has_value()) {
            message_json["tool_call_id"] = message.tool_call_id.value_or("");
        }
        if (message.content.has_value()) {
            if (std::holds_alternative<std::string>(message.content.value())) {
                message_json["content"] = std::get<std::string>(message.content.value());
            } else {
                int audio_index = 0;
                int image_index = 0;
                int video_index = 0;
                nlohmann::json message_content_items_json;
                const auto& message_content_items = std::get<std::vector<caiwei::text::CompletionsRequestMessageContentItem>>(message.content.value());
                for (const auto& message_content_item : message_content_items) {
                    if (message_content_item.type.has_value()) {
                        nlohmann::json message_content_item_json;
                        message_content_item_json["type"] = "text";
                        const std::string& type = message_content_item.type.value_or("");
                        if (type == "text") {
                            message_content_item_json["text"] = message_content_item.text.value_or("");
                        } else if (type == "audio" || type == "audio_url") {
                            audio_index++;
                            audio_count++;
                            message_content_item_json["text"] = audio_marker;
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
                            message_content_item_json["text"] = std::move(text);
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
                            message_content_item_json["text"] = std::move(text);
                        } else {
                            CW_LOG_W("不支持的媒体类型: %s", type.c_str());
                        }
                        message_content_items_json.emplace_back(std::move(message_content_item_json));
                    } else {
                        CW_LOG_W("没有设置消息类型");
                    }
                }
                for (size_t i = audio_index; i < message.audio_data.size(); ++i) {
                    CW_LOG_I("音频标签不够自动填充");
                    audio_count++;
                    message_content_items_json.emplace_back(nlohmann::json({
                        {"type", "text"},
                        {"text", audio_marker}
                    }));
                }
                for (size_t i = image_index; i < message.image_data.size(); ++i) {
                    CW_LOG_I("图片标签不够自动填充");
                    image_count++;
                    std::string text;
                    if (add_vision_id) {
                        text += "Picture ";
                        text += std::to_string(image_count);
                        text += ": ";
                    }
                    text += image_marker;
                    message_content_items_json.emplace_back(nlohmann::json({
                        {"type", "text"         },
                        {"text", std::move(text)}
                    }));
                }
                for (size_t i = video_index; i < message.video_data.size(); ++i) {
                    CW_LOG_I("视频标签不够自动填充");
                    video_count++;
                    std::string text;
                    if (add_vision_id) {
                        text += "Video ";
                        text += std::to_string(video_count);
                        text += ": ";
                    }
                    text += video_marker;
                    message_content_items_json.emplace_back(nlohmann::json({
                        {"type", "text"         },
                        {"text", std::move(text)}
                    }));
                }
                message_json["content"] = std::move(message_content_items_json);
            }
        } else {
            message_json["content"] = "";
        }
        if (message.tool_calls.has_value()) {
            nlohmann::json message_tool_calls_json;
            const auto& message_tool_calls = message.tool_calls.value();
            for (const auto& message_tool_call : message_tool_calls) {
                nlohmann::json message_tool_call_json;
                if (message_tool_call.id.has_value()) {
                    message_tool_call_json["id"] = message_tool_call.id.value_or("");
                }
                if (message_tool_call.type.has_value()) {
                    message_tool_call_json["type"] = message_tool_call.type.value_or("");
                }
                if (message_tool_call.function.has_value()) {
                    nlohmann::json message_tool_call_function_json;
                    const auto& message_tool_call_function = message_tool_call.function.value();
                    if (message_tool_call_function.name.has_value()) {
                        message_tool_call_function_json["name"] = message_tool_call_function.name.value_or("");
                    }
                    if (message_tool_call_function.arguments.has_value()) {
                        message_tool_call_function_json["arguments"] = message_tool_call_function.arguments.value_or("");
                    }
                    message_tool_call_json["function"] = std::move(message_tool_call_function_json);
                }
                message_tool_calls_json.push_back(message_tool_call_json);
            }
            message_json["tool_calls"] = std::move(message_tool_calls_json);
        }
        messages_json.push_back(std::move(message_json));
    }
    if (request.tools.has_value()) {
        const auto& tools = request.tools.value();
        for (const auto& tool : tools) {
            nlohmann::json tool_json;
            if (tool.type.has_value()) {
                tool_json["type"] = tool.type.value_or("");
            }
            if (tool.function.has_value()) {
                nlohmann::json tool_function_json;
                const auto& tool_function = tool.function.value();
                if (tool_function.name.has_value()) {
                    tool_function_json["name"] = tool_function.name.value_or("");
                }
                if (tool_function.description.has_value()) {
                    tool_function_json["description"] = tool_function.description.value_or("");
                }
                if (tool_function.parameters.has_value()) {
                    tool_function_json["parameters"] = tool_function.parameters.value_or(nlohmann::json());
                }
                tool_json["function"] = std::move(tool_function_json);
            }
            tools_json.push_back(std::move(tool_json));
        }
    }
    extra_context_json["add_vision_id"] = add_vision_id;
    if (special_token.enable_thinking.empty()) {
        extra_context_json["enable_thinking"] = request.enable_thinking.value_or(true);
    } else {
        extra_context_json[special_token.enable_thinking] = request.enable_thinking.value_or(true);
    }
    try {
        return this->chat_template->apply(minja::chat_template_inputs{
            .messages              = messages_json,
            .tools                 = tools_json,
            .add_generation_prompt = true,
            .extra_context         = extra_context_json
        });
    } catch(const std::exception& e) {
        CW_LOG_E("模板渲染异常: %s", e.what());
    } catch (...)  {
        CW_LOG_E("模板渲染异常");
    }
    return "";
}
