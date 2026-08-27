/**
 * 文本
 */
#ifndef CAIWEI_TEXT_HPP
#define CAIWEI_TEXT_HPP

#include <string>

namespace caiwei {
namespace text   {

enum class Role {

    USER,
    ASSISTANT,

};

enum class Type {

    TEXT,
    AUDIO,
    IMAGE,
    VIDEO,

};

class Content {

private:
    std::string text;
    std::string audio;
    std::string image;
    std::string video;

};

} // namespace text
} // namespace caiwei

#endif // CAIWEI_TEXT_HPP
