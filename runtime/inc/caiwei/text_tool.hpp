/**
 * 文本工具
 */
#ifndef CAIWEI_RUNTIME_TEXT_TOOL_HPP
#define CAIWEI_RUNTIME_TEXT_TOOL_HPP

#include "caiwei/text_data.hpp"

#include <memory>

namespace minja {
class chat_template;
};

namespace caiwei {
namespace text   {

std::string get_reranking_prompt(const std::string& document, const caiwei::text::RerankingsRequest& request, const caiwei::text::SpecialToken& special_token);

class ChatTemplate {
private:
    std::unique_ptr<minja::chat_template> chat_template { nullptr };
public:
    ChatTemplate();
    ~ChatTemplate();
public:
    void set_template(const std::string& template_text, const std::string& bos, const std::string& eos);
    std::string apply(
        const SpecialToken& special_token,
        const caiwei::text::CompletionsRequest& request,
        const std::string& audio_marker = "",
        const std::string& image_marker = "",
        const std::string& video_marker = "",
        bool add_vision_id = true
    );
};

} // namespace text
} // namespace caiwei

#endif // CAIWEI_RUNTIME_TEXT_TOOL_HPP
