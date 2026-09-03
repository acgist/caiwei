#include "caiwei/image.hpp"

#include <algorithm>

caiwei::image::Box caiwei::image::rect_to_box(const Rect& rect, int class_id, float score) {
    return Box(
        std::max(0.0F, rect.x),
        std::max(0.0F, rect.y),
        std::min(1.0F, rect.x + rect.w),
        std::min(1.0F, rect.y + rect.h),
        class_id,
        score
    );
}

caiwei::image::RectCenter caiwei::image::rect_to_rect_center(const Rect& rect) {
    return RectCenter(
        rect.x + rect.w / 2.0F,
        rect.y + rect.h / 2.0F,
        rect.w,
        rect.h
    );
}

caiwei::image::Box caiwei::image::rect_center_to_box(const RectCenter& rect, int class_id, float score) {
    return Box(
        std::min(0.0F, rect.cx - rect.w / 2.0F),
        std::min(0.0F, rect.cy - rect.h / 2.0F),
        std::min(1.0F, rect.cx + rect.w / 2.0F),
        std::min(1.0F, rect.cy + rect.h / 2.0F),
        class_id,
        score
    );
}

caiwei::image::Rect caiwei::image::rect_center_to_rect(const RectCenter& rect) {
    return Rect(
        rect.cx - rect.w / 2.0F,
        rect.cy - rect.h / 2.0F,
        rect.w,
        rect.h
    );
}
