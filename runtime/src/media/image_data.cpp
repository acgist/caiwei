#include "caiwei/image_data.hpp"

#include "nlohmann/json.hpp"

namespace caiwei {
namespace image  {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Cls,                 class_id, score);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Box, x1, y1, x2, y2, class_id, score);

}
}

std::string caiwei::image::to_json(const caiwei::image::Cls& cls) {
    return nlohmann::json(cls).dump();
}

std::string caiwei::image::to_json(const caiwei::image::Box& box) {
    return nlohmann::json(box).dump();
}

std::string caiwei::image::to_json(const std::vector<Cls>& cls) {
    nlohmann::json ret = cls;
    return ret.dump();
}

std::string caiwei::image::to_json(const std::vector<Box>& box) {
    nlohmann::json ret = box;
    return ret.dump();
}
