#ifndef CAIWEI_RUNTIME_MEDIA_DATA_HPP
#define CAIWEI_RUNTIME_MEDIA_DATA_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#ifndef QWEN_ASR_N_FFT
#define QWEN_ASR_N_FFT 400
#endif
#ifndef QWEN_ASR_HOP_LENGTH
#define QWEN_ASR_HOP_LENGTH 160
#endif
#ifndef QWEN_ASR_DOWN_SAMPLE_RATE
#define QWEN_ASR_DOWN_SAMPLE_RATE 8
#endif

#ifndef YOLO_BOX_SIZE
#define YOLO_BOX_SIZE 4
#endif

namespace caiwei {
namespace media {

// clang-format off
inline constexpr const int IMAGE_WIDTH        = 640;
inline constexpr const int IMAGE_HEIGHT       = 360;
inline constexpr const int IMAGE_CHANNELS     = 3;
inline constexpr const int AUDIO_CHANNELS     = 1;
inline constexpr const int AUDIO_SAMPLE_RATE  = 16000;
inline constexpr const int AUDIO_MIN_CHANNELS = 1;
inline constexpr const int AUDIO_MAX_CHANNELS = 2;
inline constexpr const int AUDIO_MIN_SAMPLE_RATE = 16000;
inline constexpr const int AUDIO_MAX_SAMPLE_RATE = 48000;
inline constexpr const int VIDEO_MIN_FPS    = 1;
inline constexpr const int VIDEO_MAX_FPS    = 120;
inline constexpr const int VIDEO_MIN_WIDTH  = 128;
inline constexpr const int VIDEO_MAX_WIDTH  = 2560;
inline constexpr const int VIDEO_MIN_HEIGHT = 96;
inline constexpr const int VIDEO_MAX_HEIGHT = 2560;
// clang-format on

inline constexpr uint8_t DEFAULT_PADDING = 114;

// clang-format off
inline constexpr const char* sdp_file_path   = "./sdp/";
inline constexpr const char* sdp_file_suffix = ".sdp";
// clang-format on

enum class MediaType {
    AUDIO,
    IMAGE,
    VIDEO,
};

class Frame {
public:
    std::vector<uint8_t> data;
public:
    Frame() noexcept = default;
    virtual ~Frame() = default;
    Frame(const Frame& result) = default;
    Frame& operator=(const Frame& result) = default;
    Frame(Frame&& result) noexcept = default;
    Frame& operator=(Frame&& result) noexcept = default;
};

class AudioFrame : public Frame {
public:
    // clang-format off
    uint64_t frames  = 0;
    uint32_t samples = 0;
    // clang-format on
};

class ImageFrame : public Frame {
public:
    // clang-format off
    uint32_t width    = 0;
    uint32_t height   = 0;
    uint32_t channels = 0;
    // clang-format on
public:
    bool load_from_file(const char* path);
    bool load_from_memory(const uint8_t* buffer, const size_t size);
    bool check() const;
};

class VideoFrame : public ImageFrame {
public:
    uint64_t frames = 0;
};

struct Cls {
    // clang-format off
    int   class_id = 0;
    float score    = 0.0F;
    // clang-format on
};

struct Box {
    // clang-format off
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
    int   class_id = 0;
    float score    = 0.0F;
    // clang-format on
};

} // namespace media
} // namespace caiwei

#endif // CAIWEI_RUNTIME_MEDIA_DATA_HPP
