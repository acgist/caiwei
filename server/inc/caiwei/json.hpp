#ifndef CAIWEI_SERVER_JSON_HPP
#define CAIWEI_SERVER_JSON_HPP

#include <utility>

#include "nlohmann/json.hpp"

namespace caiwei {
namespace json {

inline std::string buildResponse(nlohmann::json&& body = nlohmann::json::object()) {
    // clang-format off
    nlohmann::json response = {
        {"header", {
            {"code",    "0000"   },
            {"message", "success"},
        }},
        {"body", std::move(body)},
    };
    // clang-format on
    return response.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

inline std::string buildResponse(std::string code, std::string message) {
    // clang-format off
    nlohmann::json response = {
        {"header", {
            {"code",    std::move(code)   },
            {"message", std::move(message)},
        }}
    };
    // clang-format on
    return response.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

} // namespace json
} // namespace caiwei

#endif // CAIWEI_SERVER_JSON_HPP
