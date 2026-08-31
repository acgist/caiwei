/**
 * YOLO视觉结构
 * 
 * YOLO letterbox
 * 缩放居中
 * 默认填充: 114
 */
#ifndef CAIWEI_CONTEXT_YOLO_DATA_HPP
#define CAIWEI_CONTEXT_YOLO_DATA_HPP

namespace caiwei {
namespace yolo   {

struct Point {
    float x; // x百分比
    float y; // y百分比
};

struct Rect {
    float x; // 左上角x百分比
    float y; // 左上角y百分比
    float w; // 宽度百分比
    float h; // 高度百分比
};

struct RectCenter {
    float cx; // 中心x百分比
    float cy; // 中心y百分比
    float  w; // 宽度百分比
    float  h; // 高度百分比
};

struct Box {
    float x1; // 左上角x百分比
    float y1; // 左上角y百分比
    float x2; // 右下角x百分比
    float y2; // 右下角y百分比
    int   class_id; // 类别ID
    float score;    // 置信度
};

struct Area {

};

} // namespace context
} // namespace caiwei

#endif // CAIWEI_CONTEXT_YOLO_DATA_HPP
