#ifndef CAIWEI_RUNTIME_TEXT_TOOL_HPP
#define CAIWEI_RUNTIME_TEXT_TOOL_HPP

#include <memory>

#include "caiwei/text_data.hpp"

namespace minja {

class chat_template;

} // minja

namespace caiwei {
namespace text   {

std::string get_reranking_prompt(
    const std::string& document,
    const caiwei::text::RerankingsRequest& request,
    const caiwei::text::SpecialToken& special_token
);

class ChatTemplate {
public:
    ChatTemplate();
    ~ChatTemplate();
private:
    std::unique_ptr<minja::chat_template> chat_template{ nullptr };
public:
    void set_template(const char* template_text, const std::string& bos, const std::string& eos);
    std::string apply(
        const SpecialToken& special_token,
        const caiwei::text::CompletionsRequest& request,
        const std::string& audio_marker = "",
        const std::string& image_marker = "",
        const std::string& video_marker = "",
        bool add_vision_id = false
    ) const;
};

} // text
} // caiwei

#endif // CAIWEI_RUNTIME_TEXT_TOOL_HPP
