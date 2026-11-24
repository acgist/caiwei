#include "caiwei/media_data.hpp"
#include "caiwei/log.hpp"

#include <climits>

#include "stb/stb_image.h"

bool caiwei::media::ImageFrame::load_from_file(const char* path) {
    if (path == nullptr) {
        return false;
    }
    int width, height, channels;
    auto* data = stbi_load(path, &width, &height, &channels, STBI_rgb);
    if (data == nullptr) {
        CW_LOG_W("图片加载失败: %s = %s", path, stbi_failure_reason());
        return false;
    }
    channels = IMAGE_CHANNELS;
    this->data.assign(data, data + static_cast<size_t>(width) * height * channels);
    stbi_image_free(data);
    // clang-format off
    this->width    = width;
    this->height   = height;
    this->channels = channels;
    // clang-format on
    return true;
}

bool caiwei::media::ImageFrame::load_from_memory(const uint8_t* buffer, const size_t size) {
    if (size > INT_MAX) {
        CW_LOG_W("图片加载失败: %zu", size);
        return false;
    }
    if (buffer == nullptr || size == 0) {
        CW_LOG_W("图片加载失败: 数据为空");
        return false;
    }
    int width, height, channels;
    auto* data = stbi_load_from_memory(buffer, static_cast<int>(size), &width, &height, &channels, STBI_rgb);
    if (data == nullptr) {
        CW_LOG_W("图片加载失败: %zu = %s", size, stbi_failure_reason());
        return false;
    }
    channels = IMAGE_CHANNELS;
    this->data.assign(data, data + static_cast<size_t>(width) * height * channels);
    stbi_image_free(data);
    // clang-format off
    this->width    = width;
    this->height   = height;
    this->channels = channels;
    // clang-format on
    return true;
}

bool caiwei::media::ImageFrame::check() const {
    // clang-format off
    return
    this->width  > 0 &&
    this->height > 0 &&
    this->channels == IMAGE_CHANNELS &&
    this->data.size() == static_cast<size_t>(this->width) * this->height * this->channels;
    // clang-format on
}
