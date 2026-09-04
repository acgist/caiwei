/**
 * 文本工具
 */
#ifndef CAIWEI_MEDIA_TEXT_TOOL_HPP
#define CAIWEI_MEDIA_TEXT_TOOL_HPP

#include "caiwei/text_data.hpp"

#include <memory>

namespace minja {
class chat_template;
};

namespace caiwei {
namespace text   {

class ChatTemplate {
private:
    std::unique_ptr<minja::chat_template> chat_template { nullptr };
public:
    ChatTemplate();
    ~ChatTemplate();
public:
    void set_template(const std::string& template_text, const std::string& bos, const std::string& eos);
    std::string apply(const SpecialToken& special_token, const caiwei::text::CompletionsRequest& request);
};

std::string chunk_choice(const CompletionsRequest& request, const Result& result);
std::string response_choice(const CompletionsRequest& request, const std::string& finish_reason, std::string content, std::string thinking, std::string toolcall);

} // namespace text
} // namespace caiwei

#endif // CAIWEI_MEDIA_TEXT_TOOL_HPP
