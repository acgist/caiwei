/**
 * 图像
 */
#ifndef CAIWEI_MEDIA_IMAGE_HPP
#define CAIWEI_MEDIA_IMAGE_HPP

#include "caiwei/image_data.hpp"
#include "caiwei/media_data.hpp"

namespace caiwei {
namespace image  {

Box        rect_to_box        (const Rect& rect, int class_id, float score);
RectCenter rect_to_rect_center(const Rect& rect);
Box  rect_center_to_box (const RectCenter& rect, int class_id, float score);
Rect rect_center_to_rect(const RectCenter& rect);

/**
 * 缩放居中
 * 默认填充: 114
 */
void letterbox(const caiwei::media::ImageFrame& frame);

} // namespace image
} // namespace caiwei

#endif // CAIWEI_MEDIA_IMAGE_HPP
