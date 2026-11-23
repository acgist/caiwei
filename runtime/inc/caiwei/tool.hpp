#ifndef CAIWEI_RUNTIME_TOOL_HPP
#define CAIWEI_RUNTIME_TOOL_HPP

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

namespace caiwei {
namespace tool {

inline constexpr const std::string_view empty_char = " \t\r\n";

inline std::string strip(std::string_view value) {
    auto pos_a = value.find_first_not_of(empty_char);
    if (pos_a == std::string_view::npos) {
        return {};
    }
    auto pos_z = value.find_last_not_of(empty_char);
    return std::string(value.data() + pos_a, pos_z - pos_a + 1);
}

inline std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char v) {
        return static_cast<char>(std::tolower(v));
    });
    return value;
}

inline std::string to_upper(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char v) {
        return static_cast<char>(std::toupper(v));
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
        if (std::tolower(static_cast<unsigned char>(src[i])) != std::tolower(static_cast<unsigned char>(dst[i]))) {
            return false;
        }
    }
    return true;
}

} // namespace tool
} // namespace caiwei

#endif // CAIWEI_RUNTIME_TOOL_HPP
