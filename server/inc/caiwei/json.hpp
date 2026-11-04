#ifndef CAIWEI_SERVER_JSON_HPP
#define CAIWEI_SERVER_JSON_HPP

#include "nlohmann/json.hpp"

namespace caiwei {
namespace json   {

inline std::string buildResponse(nlohmann::json body = {}) {
    nlohmann::json response = {
        {"header", {
            {"code",    "0000"   },
            {"message", "success"},
        }},
        {"body", std::move(body)},
    };
    return response.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

inline std::string buildResponse(std::string code, std::string message, nlohmann::json body = {}) {
    nlohmann::json response = {
        {"header", {
            {"code",    std::move(code)   },
            {"message", std::move(message)},
        }},
        {"body", std::move(body)},
    };
    return response.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

} // json
} // caiwei

#endif // CAIWEI_SERVER_JSON_HPP
