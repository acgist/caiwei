#ifndef CAIWEI_RUNTIME_IMAGE_DATA_HPP
#define CAIWEI_RUNTIME_IMAGE_DATA_HPP

#include <string>
#include <vector>
#include <cstdint>

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

} // image
} // caiwei

#endif // CAIWEI_RUNTIME_IMAGE_DATA_HPP
