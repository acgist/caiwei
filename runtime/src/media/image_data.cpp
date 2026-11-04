#include "nlohmann/json.hpp"

#include "caiwei/image_data.hpp"

namespace caiwei {
namespace image  {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Cls,                 class_id, score);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Box, x1, y1, x2, y2, class_id, score);

}
}

std::string caiwei::image::to_json(const caiwei::image::Cls& cls) {
    return nlohmann::json(cls).dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

std::string caiwei::image::to_json(const caiwei::image::Box& box) {
    return nlohmann::json(box).dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

std::string caiwei::image::to_json(const std::vector<Cls>& cls) {
    nlohmann::json ret = cls;
    return ret.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

std::string caiwei::image::to_json(const std::vector<Box>& box) {
    nlohmann::json ret = box;
    return ret.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}
