/**
 * 图片结构
 */
#ifndef CAIWEI_MEDIA_IMAGE_DATA_HPP
#define CAIWEI_MEDIA_IMAGE_DATA_HPP

#include <cstdint>

namespace caiwei {
namespace image  {

struct Box {
    float x1; // 左上角x百分比
    float y1; // 左上角y百分比
    float x2; // 右下角x百分比
    float y2; // 右下角y百分比
    int   class_id; // 类别ID
    float score;    // 置信度
};

struct Mask {
    int h;
    int w;
    std::vector<float> mask;
};

struct Seg {
    Box  box;
    Mask mask;
};

struct Point {
    float x; // x百分比
    float y; // y百分比
    float score; // 置信度
};

using PosePoint = std::vector<Point>;

struct Pose {
    Box box;
    PosePoint point;
};

} // namespace image
} // namespace caiwei

#endif // CAIWEI_IMAGE_DATA_HPP
