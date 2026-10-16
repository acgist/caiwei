#include "caiwei/image_data.hpp"

#include "nlohmann/json.hpp"

namespace caiwei {
namespace image  {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Box, x1, y1, x2, y2, class_id, score);

}
}

std::string caiwei::image::to_json(const caiwei::image::Cls& cls) {
    nlohmann::json ret;
    ret["class_id"] = cls.first;
    ret["score"]    = cls.second;
    return ret.dump();
}

std::string caiwei::image::to_json(const caiwei::image::Box& box) {
    return nlohmann::json(box).dump();
}
