/**
 * JSON
 */
#ifndef CAIWEI_JSON_HPP
#define CAIWEI_JSON_HPP

#include <vector>

#include "json/json.hpp"

namespace caiwei {
namespace json   {

inline std::string buildResponse(nlohmann::json body = {}) {
    nlohmann::json response;
    nlohmann::json header;
    header["code"]     = "0000";
    header["message"]  = "成功";
    response["header"] = header;
    response["body"]   = body;
    return response.dump();
}

inline std::string buildResponse(const std::string& code, const std::string& message) {
    nlohmann::json response;
    nlohmann::json header;
    header["code"]     = code;
    header["message"]  = message;
    response["header"] = header;
    return response.dump();
}

} // json
} // caiwei

#endif // CAIWEI_JSON_HPP
