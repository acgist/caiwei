#ifndef CAIWEI_RUNTIME_IMAGE_DATA_HPP
#define CAIWEI_RUNTIME_IMAGE_DATA_HPP

#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>

namespace caiwei {
namespace image  {

struct Cls {
    int   class_id;
    float score;
};

struct Box {
    float x1;
    float y1;
    float x2;
    float y2;
    int   class_id;
    float score;
};

std::string to_json(const Cls& cls);
std::string to_json(const Box& box);
std::string to_json(const std::vector<Cls>& cls);
std::string to_json(const std::vector<Box>& box);

inline Box to_box(float box_x, float box_y, float box_w, float box_h, int max_class, float max_score) {
    return caiwei::image::Box(
        std::clamp(box_x - box_w / 2.0F, 0.0F, 1.0F),
        std::clamp(box_y - box_h / 2.0F, 0.0F, 1.0F),
        std::clamp(box_x + box_w / 2.0F, 0.0F, 1.0F),
        std::clamp(box_y + box_h / 2.0F, 0.0F, 1.0F),
        max_class,
        max_score
    );
}

inline Box to_box_end2end(float box_x1, float box_y1, float box_x2, float box_y2, int max_class, float max_score) {
    return caiwei::image::Box(
        std::clamp(box_x1, 0.0F, 1.0F),
        std::clamp(box_y1, 0.0F, 1.0F),
        std::clamp(box_x2, 0.0F, 1.0F),
        std::clamp(box_y2, 0.0F, 1.0F),
        max_class,
        max_score
    );
}

} // image
} // caiwei

#endif // CAIWEI_RUNTIME_IMAGE_DATA_HPP
