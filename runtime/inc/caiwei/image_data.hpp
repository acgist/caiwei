/**
 * 图片结构
 */
#ifndef CAIWEI_RUNTIME_IMAGE_DATA_HPP
#define CAIWEI_RUNTIME_IMAGE_DATA_HPP

#include <vector>
#include <string>
#include <cstdint>
#include <utility>

#include "nlohmann/json.hpp"

namespace caiwei {
namespace image  {

using Cls = std::pair<uint32_t, float>;

struct Box {
    float x1; // 左上角x百分比
    float y1; // 左上角y百分比
    float x2; // 右下角x百分比
    float y2; // 右下角y百分比
    int   class_id; // 类别ID
    float score;    // 置信度
};

std::string to_json(const Cls& cls);
std::string to_json(const Box& box);

template <typename T>
inline std::string to_json(const std::vector<T>& v) {
    nlohmann::json ret = nlohmann::json::array();
    for (const auto& x : v) {
        ret.push_back(to_json(x));
    }
    return ret.dump();
}

} // namespace image
} // namespace caiwei

#endif // CAIWEI_RUNTIME_IMAGE_DATA_HPP
