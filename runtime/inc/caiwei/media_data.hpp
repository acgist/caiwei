#ifndef CAIWEI_RUNTIME_MEDIA_DATA_HPP
#define CAIWEI_RUNTIME_MEDIA_DATA_HPP

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
    std::vector<uint8_t> data;
public:
    Frame(uint32_t size);
    Frame() noexcept = default;
    virtual ~Frame() = default;
    Frame(const Frame& result) noexcept = default;
    Frame& operator=(const Frame& result) noexcept = default;
    Frame(Frame&& result) noexcept = default;
    Frame& operator=(Frame&& result) noexcept = default;
};

class AudioFrame : public Frame {
public:
    uint64_t msec;
    uint64_t frames;
    uint32_t samples;
public:
    AudioFrame(uint32_t size);
    AudioFrame() noexcept = default;
};

class ImageFrame : public Frame {
public:
    uint32_t width;
    uint32_t height;
    uint32_t channels;
public:
    ImageFrame(uint32_t size);
    ImageFrame() noexcept = default;
};

class VideoFrame : public ImageFrame {
public:
    uint64_t msec;
    uint64_t frames;
public:
    VideoFrame(uint32_t size);
    VideoFrame() noexcept = default;
};

} // media
} // caiwei

#endif // CAIWEI_RUNTIME_MEDIA_DATA_HPP
