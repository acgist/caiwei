#include <climits>

#include "caiwei/log.hpp"
#include "caiwei/media_data.hpp"

#include "stb/stb_image.h"

caiwei::media::Frame::Frame(uint32_t size) {
    this->data.reserve(size);
}

caiwei::media::AudioFrame::AudioFrame(uint32_t size) : Frame(size) {
}

caiwei::media::ImageFrame::ImageFrame(uint32_t size) : Frame(size) {
}

caiwei::media::VideoFrame::VideoFrame(uint32_t size) : ImageFrame(size) {
}

bool caiwei::media::ImageFrame::load_from_file(const char* path) {
    if (path == nullptr) {
        return false;
    }
    int width, height, channels;
    auto* data = stbi_load(path, &width, &height, &channels, STBI_rgb);
    if (data == nullptr) {
        CW_LOG_W("图片加载失败: %s", path);
        return false;
    }
    channels = IMAGE_CHANNELS;
    this->data.assign(data, data + static_cast<size_t>(width) * height * channels);
    stbi_image_free(data);
    this->width    = width;
    this->height   = height;
    this->channels = channels;
    return true;
}

bool caiwei::media::ImageFrame::load_from_memory(const uint8_t* buffer, const size_t size) {
    if (buffer == nullptr || size == 0 || size > INT_MAX) {
        CW_LOG_W("图片加载失败: 数据为空");
        return false;
    }
    int width, height, channels;
    auto* data = stbi_load_from_memory(buffer, static_cast<int>(size), &width, &height, &channels, STBI_rgb);
    if (data == nullptr) {
        CW_LOG_W("图片加载失败: %zu", size);
        return false;
    }
    channels = IMAGE_CHANNELS;
    this->data.assign(data, data + static_cast<size_t>(width) * height * channels);
    stbi_image_free(data);
    this->width    = width;
    this->height   = height;
    this->channels = channels;
    return true;
}
