#ifndef CAIWEI_RUNTIME_MEDIA_DATA_HPP
#define CAIWEI_RUNTIME_MEDIA_DATA_HPP

#include <cstdio>
#include <vector>
#include <cstdint>

namespace caiwei {
namespace media  {

inline constexpr const int AUDIO_CHANNELS     = 1;
inline constexpr const int AUDIO_SAMPLE_RATE  = 16000;
inline constexpr const int IMAGE_WIDTH        = 640;
inline constexpr const int IMAGE_HEIGHT       = 360;
inline constexpr const int IMAGE_CHANNELS     = 3;
inline constexpr const int AUDIO_MIN_CHANNELS = 1;
inline constexpr const int AUDIO_MAX_CHANNELS = 2;
inline constexpr const int AUDIO_MIN_SAMPLE_RATE = 16000;
inline constexpr const int AUDIO_MAX_SAMPLE_RATE = 48000;
inline constexpr const int VIDEO_MIN_FPS = 1;
inline constexpr const int VIDEO_MAX_FPS = 120;
inline constexpr const int VIDEO_MIN_WIDTH = 128;
inline constexpr const int VIDEO_MAX_WIDTH = 2560;
inline constexpr const int VIDEO_MIN_HEIGHT = 0;
inline constexpr const int VIDEO_MAX_HEIGHT = 2560;

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
public:
    bool load_from_file(const char* path);
    bool load_from_memory(const uint8_t* buffer, const size_t size);
};

class VideoFrame : public ImageFrame {
public:
    uint64_t frames;
public:
    VideoFrame(uint32_t size);
    VideoFrame() noexcept = default;
};

} // media
} // caiwei

#endif // CAIWEI_RUNTIME_MEDIA_DATA_HPP
