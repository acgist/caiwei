/**
 * 媒体结构
 */
#ifndef CAIWEI_MEDIA_MEDIA_DATA_HPP
#define CAIWEI_MEDIA_MEDIA_DATA_HPP

#include <cstdio>
#include <vector>
#include <cstdint>

namespace caiwei {
namespace media  {

enum class MediaType {
    AUDIO,
    IMAGE,
    VIDEO,
};

class Frame {
public:
    uint32_t             data_length;
    std::vector<uint8_t> data;
public:
    Frame(uint32_t size);
    virtual ~Frame();
};

class AudioFrame : public Frame {
public:
    uint64_t msec;
    uint64_t frames;
    uint32_t samples;
public:
    AudioFrame(uint32_t size);
};

class ImageFrame : public Frame {
public:
    uint32_t width;
    uint32_t height;
    uint32_t channels;
public:
    ImageFrame(uint32_t size);
};

class VideoFrame : public ImageFrame {
public:
    uint64_t msec;
    uint64_t frames;
public:
    VideoFrame(uint32_t size);
};

} // namespace media
} // namespace caiwei

#endif // CAIWEI_MEDIA_MEDIA_DATA_HPP
