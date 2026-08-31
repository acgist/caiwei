/**
 * YOLO视觉
 */
#ifndef CAIWEI_CONTEXT_YOLO_HPP
#define CAIWEI_CONTEXT_YOLO_HPP

#include "caiwei/yolo_data.hpp"
#include "caiwei/media_data.hpp"

namespace caiwei {
namespace yolo   {

Box        rect_to_box        (const Rect& rect, int class_id, float score);
RectCenter rect_to_rect_center(const Rect& rect);
Box  rect_center_to_box (const RectCenter& rect, int class_id, float score);
Rect rect_center_to_rect(const RectCenter& rect);

/**
 * 缩放居中
 * 默认填充: 114
 */
void letterbox(const caiwei::media::ImageFrame& frame);

} // namespace context
} // namespace caiwei

#endif // CAIWEI_CONTEXT_YOLO_HPP
