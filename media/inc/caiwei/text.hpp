/**
 * 文本
 */
#ifndef CAIWEI_MEDIA_TEXT_HPP
#define CAIWEI_MEDIA_TEXT_HPP

#include "caiwei/text_data.hpp"

#include <memory>

namespace minja {
class chat_template;
};

namespace caiwei {
namespace text   {

class ChatTemplate {

private:
    std::unique_ptr<minja::chat_template> chat_template = nullptr;
public:
    ChatTemplate();
    ~ChatTemplate();
public:
    void set_template(const std::string& template_text, const std::string& bos, const std::string& eos);
    std::string apply(const caiwei::text::CompletionsRequest& request);

};

} // namespace text
} // namespace caiwei

#endif // CAIWEI_MEDIA_TEXT_HPP
