#ifndef CAIWEI_RUNTIME_IMAGE_DATA_HPP
#define CAIWEI_RUNTIME_IMAGE_DATA_HPP

#include <string>
#include <vector>
#include <algorithm>

namespace caiwei {
namespace image  {

struct Cls {
    int   class_id = 0;
    float score    = 0.0F;
};

struct Box {
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
    int   class_id = 0;
    float score    = 0.0F;
};

std::string to_json(const Cls& cls);
std::string to_json(const Box& box);
std::string to_json(const std::vector<Cls>& cls);
std::string to_json(const std::vector<Box>& box);

inline Box to_box(float box_x, float box_y, float box_w, float box_h, int class_id, float score) {
    return caiwei::image::Box(
        std::clamp(box_x - box_w / 2.0F, 0.0F, 1.0F),
        std::clamp(box_y - box_h / 2.0F, 0.0F, 1.0F),
        std::clamp(box_x + box_w / 2.0F, 0.0F, 1.0F),
        std::clamp(box_y + box_h / 2.0F, 0.0F, 1.0F),
        class_id,
        score
    );
}

inline Box to_box_end2end(float box_x1, float box_y1, float box_x2, float box_y2, int class_id, float score) {
    return caiwei::image::Box(
        std::clamp(box_x1, 0.0F, 1.0F),
        std::clamp(box_y1, 0.0F, 1.0F),
        std::clamp(box_x2, 0.0F, 1.0F),
        std::clamp(box_y2, 0.0F, 1.0F),
        class_id,
        score
    );
}

} // image
} // caiwei

#endif // CAIWEI_RUNTIME_IMAGE_DATA_HPP
