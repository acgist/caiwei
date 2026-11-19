#ifndef CAIWEI_RUNTIME_TOOL_HPP
#define CAIWEI_RUNTIME_TOOL_HPP

#include <cctype>
#include <string>
#include <algorithm>
#include <string_view>

namespace caiwei {
namespace tool {

inline constexpr const char* empty_char = " \t\r\n";

inline std::string strip(const std::string& value) {
    auto pos_a = value.find_first_not_of(empty_char);
    if (pos_a == std::string_view::npos) {
        return "";
    }
    auto pos_z = value.find_last_not_of(empty_char);
    return value.substr(pos_a, pos_z - pos_a + 1);
}

inline std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char v) {
        return static_cast<char>(tolower(v));
    });
    return value;
}

inline std::string to_upper(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char v) {
        return static_cast<char>(toupper(v));
    });
    return value;
}

inline bool equals_ignore_case(const std::string_view src, const std::string_view dst, size_t size = 0) {
    if (size == 0) {
        if (src.size() != dst.size()) {
            return false;
        }
        size = src.size();
    } else {
        if (size > src.size() || size > dst.size()) {
            return false;
        }
    }
    for (size_t i = 0; i < size; ++i) {
        if (tolower(static_cast<unsigned char>(src[i])) != tolower(static_cast<unsigned char>(dst[i]))) {
            return false;
        }
    }
    return true;
}

}  // namespace tool
}  // namespace caiwei

#endif  // CAIWEI_RUNTIME_TOOL_HPP
