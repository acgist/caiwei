#include "caiwei/media_tool.hpp"
#include "caiwei/env.hpp"
#include "caiwei/log.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

#if ENABLE_CAIWEI_BACKEND_RKNN
#include "rga/im2d.h"
#define RKNN_WIDTH_ALIGN 16
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STBIR_DEFAULT_FILTER_DOWNSAMPLE STBIR_FILTER_BOX
#define STB_IMAGE_RESIZE2_IMPLEMENTATION
#include "stb/stb_image_resize2.h"

#if ENABLE_CAIWEI_TEST
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#endif

#if defined(__aarch64__)
#include "arm_neon.h"
#endif
#if CAIWEI_OS_WIN
#include <corecrt_math_defines.h>
#endif

#include "nlohmann/json.hpp"
#include "pocketfft_hdronly/pocketfft_hdronly.h"

extern "C" {
#include "libavdevice/avdevice.h"
#include "libavformat/avformat.h"
#include "libavformat/avio.h"
#include "libavutil/pixfmt.h"
#include "libavutil/samplefmt.h"
}

#if CAIWEI_DEBUG
static void print_all_decoder();
static void print_all_encoder();
static void print_all_protocols();
#endif
static void delete_old_sdp_file();

static std::atomic<bool> ffmpeg_init(false);

static void log_mel_spectrogram(
    // clang-format off
    const float* audio_data,
    const int    audio_length,
    const float* mel_filters,
    const int    n_fft,
    const int    hop_length,
    const int    n_mels,
    std::vector<float>& mel_spec
    // clang-format on
);

caiwei::media::AudioInfo::AudioInfo(int channels, int sample_rate, int format) :
    channels(channels),
    sample_rate(sample_rate),
    format(format) {
    if (channels < AUDIO_MIN_CHANNELS || channels > AUDIO_MAX_CHANNELS) {
        CW_LOG_W("音频通道错误: %d", channels);
    }
    if (sample_rate < AUDIO_MIN_SAMPLE_RATE || sample_rate > AUDIO_MAX_SAMPLE_RATE) {
        CW_LOG_W("音频采样率错误: %d", sample_rate);
    }
    if (format < AV_SAMPLE_FMT_U8 || format >= AV_SAMPLE_FMT_NB) {
        CW_LOG_W("音频格式错误: %d", format);
    } else {
        this->bytes_per_sample = av_get_bytes_per_sample((AVSampleFormat) format);
    }
}

caiwei::media::VideoInfo::VideoInfo(int width, int height, int format) : fps(0), width(width), height(height), format(format) {
    if (width < VIDEO_MIN_WIDTH || width > VIDEO_MAX_WIDTH || height < VIDEO_MIN_HEIGHT || height > VIDEO_MAX_HEIGHT) {
        CW_LOG_W("视频尺寸错误: %d x %d", width, height);
    }
    if (format < AV_PIX_FMT_YUV420P || format >= AV_PIX_FMT_NB) {
        CW_LOG_W("视频格式错误: %d", format);
    }
}

caiwei::media::VideoInfo::VideoInfo(int fps, int width, int height, int format) :
    fps(fps),
    width(width),
    height(height),
    format(format) {
    if (fps < VIDEO_MIN_FPS || fps > VIDEO_MAX_FPS) {
        CW_LOG_W("视频帧率错误: %d", fps);
    }
    if (width < VIDEO_MIN_WIDTH || width > VIDEO_MAX_WIDTH || height < VIDEO_MIN_HEIGHT || height > VIDEO_MAX_HEIGHT) {
        CW_LOG_W("视频尺寸错误: %d x %d", width, height);
    }
    if (format < AV_PIX_FMT_YUV420P || format >= AV_PIX_FMT_NB) {
        CW_LOG_W("视频格式错误: %d", format);
    }
}

namespace caiwei {
namespace media {

// clang-format off
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Cls,                 class_id, score);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Box, x1, y1, x2, y2, class_id, score);
// clang-format on

} // namespace media
} // namespace caiwei

std::string caiwei::media::to_json(const caiwei::media::Cls& cls) {
    return nlohmann::json(cls).dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

std::string caiwei::media::to_json(const caiwei::media::Box& box) {
    return nlohmann::json(box).dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

std::string caiwei::media::to_json(const std::vector<Cls>& cls) {
    nlohmann::json ret = cls;
    return ret.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

std::string caiwei::media::to_json(const std::vector<Box>& box) {
    nlohmann::json ret = box;
    return ret.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

bool caiwei::media::resize(
    // clang-format off
    const uint8_t* __restrict src,
          uint8_t* __restrict dst,
    const int src_w,
    const int src_h,
    const int dst_w,
    const int dst_h,
    const int channels
    // clang-format on
) {
    if (src_h == dst_h && src_w == dst_w) {
        std::copy_n(src, static_cast<size_t>(src_w) * src_h * channels, dst);
        return true;
    }
    if (channels == IMAGE_CHANNELS) {
        if (stbir_resize_uint8_linear(src, src_w, src_h, 0, dst, dst_w, dst_h, 0, STBIR_RGB) == nullptr) {
            return false;
        }
    } else {
        CW_LOG_W("不支持的图片通道: %d", channels);
        return false;
    }
    return true;
}

#if ENABLE_CAIWEI_BACKEND_RKNN
bool caiwei::media::resize(
    // clang-format off
    const uint8_t* src,
          uint8_t* dst,
    const int src_w,
    const int src_h,
    const int w,
    const int h,
    const int dst_w,
    const int dst_h,
    const int pad_w,
    const int pad_h,
    const int channels
    // clang-format on
) {
    RgaSURF_FORMAT image_type;
    if (channels == IMAGE_CHANNELS) {
        image_type = RK_FORMAT_RGB_888;
    } else {
        CW_LOG_W("不支持的图片通道: %d", channels);
        return false;
    }
    if (src_h == dst_h && src_w == dst_w && pad_w == 0 && pad_h == 0) {
        std::copy_n(src, static_cast<size_t>(src_w) * src_h * channels, dst);
        return true;
    }
    if (src_w % RKNN_WIDTH_ALIGN != 0) {
        CW_LOG_D("图片宽度必须是%d的倍数: %d", RKNN_WIDTH_ALIGN, src_w);
        thread_local std::vector<uint8_t> v;
        v.resize(static_cast<size_t>(dst_w) * dst_h * channels);
        if (!caiwei::media::resize(src, v.data(), src_w, src_h, dst_w, dst_h, channels)) {
            return false;
        }
        if (!caiwei::media::padding(v.data(), dst, dst_w, dst_h, pad_w, pad_h, w, h, channels)) {
            return false;
        }
        return true;
    }
    // clang-format off
    rga_buffer_t src_img = wrapbuffer_virtualaddr((void*) src, src_w, src_h, image_type);
    rga_buffer_t dst_img = wrapbuffer_virtualaddr(dst,         w,     h,     image_type);
    rga_buffer_t pat_img = wrapbuffer_virtualaddr(nullptr,     0,     0,     0         );
    im_rect src_rect = { 0,     0,     src_w, src_h };
    im_rect dst_rect = { pad_w, pad_h, dst_w, dst_h };
    im_rect pat_rect = { 0,     0,     0,     0     };
    // clang-format on
    const auto status = improcess(src_img, dst_img, pat_img, src_rect, dst_rect, pat_rect, IM_SYNC);
    if (status != IM_STATUS_SUCCESS) {
        CW_LOG_W("图片处理失败: %d", status);
        return false;
    } else {
        return true;
    }
}
#endif

bool caiwei::media::load_mel_filters(const int n_mels, const int n_fft, std::vector<float>& mel_filters) {
    if (n_mels <= 0 || n_fft <= 1) {
        CW_LOG_W("读取mel_filters参数错误: %d - %d", n_mels, n_fft);
        return false;
    }
    const std::string mel_filters_file = caiwei::env::get("CAIWEI_MEL_FILTERS");
    std::ifstream stream(mel_filters_file);
    if (!stream.is_open()) {
        CW_LOG_W("读取mel_filters文件失败: %s", mel_filters_file.c_str());
        return false;
    }
    const int n_fft_bins = n_fft / 2 + 1;
    const int mel_filters_size = n_mels * n_fft_bins;
    mel_filters.resize(mel_filters_size);
    int index = 0;
    float value;
    stream.imbue(std::locale::classic());
    while (index < mel_filters_size && stream >> value) {
        mel_filters[index++] = value;
    }
    return index == mel_filters_size;
}

bool caiwei::media::log_mel_spectrogram(
    // clang-format off
    const float* audio_data,
    const int    audio_length,
    const float* mel_filters,
    const int    n_fft,
    const int    hop_length,
    const int    n_mels,
    const int    max_audio_length,
    std::vector<float>& mel_spec
    // clang-format on
) {
    if (max_audio_length < n_fft) {
        CW_LOG_W("音频数据小于窗口: %d - %d", max_audio_length, n_fft);
        return false;
    }
    if (audio_length < max_audio_length) {
        CW_LOG_W("音频数据长度不足: %d - %d", audio_length, max_audio_length);
        return false;
    }
    if (audio_data == nullptr || mel_filters == nullptr || n_fft <= 1 || hop_length <= 0 || n_mels <= 0) {
        CW_LOG_W("音频数据参数错误");
        return false;
    }
    mel_spec.resize(static_cast<size_t>(n_mels) * max_audio_length / hop_length, 0.0F);
    ::log_mel_spectrogram(audio_data, max_audio_length, mel_filters, n_fft, hop_length, n_mels, mel_spec);
    return true;
}

void caiwei::media::init() {
    if (!ffmpeg_init) {
        int ret = avformat_network_init();
        if (ret == 0) {
            CW_LOG_I("FFmpeg网络加载成功: %d", ret);
        } else {
            CW_LOG_W("FFmpeg网络加载失败: %d", ret);
        }
        avdevice_register_all();
        ffmpeg_init = true;
    }
#if CAIWEI_DEBUG
    print_all_decoder();
    print_all_encoder();
    print_all_protocols();
#endif
    delete_old_sdp_file();
}

void caiwei::media::stop() {
    if (ffmpeg_init) {
        int ret = avformat_network_deinit();
        if (ret == 0) {
            CW_LOG_I("FFmpeg网络卸载成功: %d", ret);
        } else {
            CW_LOG_W("FFmpeg网络卸载失败: %d", ret);
        }
        ffmpeg_init = false;
    }
}

#if CAIWEI_DEBUG
static void print_all_decoder() {
    void* iter = nullptr;
    const AVCodec* codec = nullptr;
    while ((codec = av_codec_iterate(&iter))) {
        if (av_codec_is_decoder(codec)) {
            if (codec->type == AVMEDIA_TYPE_AUDIO) {
                CW_LOG_D("ffmpeg audio decoder: %-16s = %s", codec->name, codec->long_name);
            } else if (codec->type == AVMEDIA_TYPE_VIDEO) {
                CW_LOG_D("ffmpeg video decoder: %-16s = %s", codec->name, codec->long_name);
            } else {
                // -
            }
        }
    }
}

static void print_all_encoder() {
    void* iter = nullptr;
    const AVCodec* codec = nullptr;
    while ((codec = av_codec_iterate(&iter))) {
        if (av_codec_is_encoder(codec)) {
            if (codec->type == AVMEDIA_TYPE_AUDIO) {
                CW_LOG_D("ffmpeg audio encoder: %-16s = %s", codec->name, codec->long_name);
            } else if (codec->type == AVMEDIA_TYPE_VIDEO) {
                CW_LOG_D("ffmpeg video encoder: %-16s = %s", codec->name, codec->long_name);
            } else {
                // -
            }
        }
    }
}

static void print_all_protocols() {
    void* iter = nullptr;
    const char* name = nullptr;
    while ((name = avio_enum_protocols(&iter, 0)) != nullptr) {
        CW_LOG_D("ffmpeg protocol: %s", name);
    }
}
#endif

static void delete_old_sdp_file() {
    std::error_code error_code_exists;
    if (!std::filesystem::exists(caiwei::media::sdp_file_path, error_code_exists)) {
        return;
    }
    std::error_code error_code_iter;
    auto iter = std::filesystem::directory_iterator(caiwei::media::sdp_file_path, error_code_iter);
    if (error_code_iter) {
        CW_LOG_W("遍历旧的SDP文件失败: %s = %s", caiwei::media::sdp_file_path, error_code_iter.message().c_str());
        return;
    }
    for (auto& entry : iter) {
        std::error_code error_code_file;
        if (!entry.is_regular_file(error_code_file)) {
            continue;
        }
        if (entry.path().extension() == ".sdp") {
            std::error_code error_code_remove;
            if (std::filesystem::remove(entry.path(), error_code_remove)) {
                CW_LOG_I("删除旧的SDP文件成功: %s", entry.path().string().c_str());
            } else {
                if (error_code_remove) {
                    CW_LOG_W(
                        "删除旧的SDP文件失败: %s = %s", entry.path().string().c_str(), error_code_remove.message().c_str()
                    );
                } else {
                    CW_LOG_W("删除旧的SDP文件失败: %s", entry.path().string().c_str());
                }
            }
        }
    }
}

static void hann_window(std::vector<float>& window, const int window_length) {
    for (int i = 0; i < window_length; ++i) {
        window[i] = 0.5 * (1 - std::cos(2 * M_PI * i / (window_length - 1)));
    }
}

static void reflect_pad(const float* begin, const float* end, std::vector<float>& pad_audio, const int pad_width) {
    std::copy(begin, end, pad_audio.begin() + pad_width);
    std::reverse_copy(begin, begin + pad_width, pad_audio.begin());
    std::reverse_copy(end - pad_width, end, pad_audio.end() - pad_width);
}

static void stfts(
    // clang-format off
    const float* audio_data,
    const int    audio_length,
    const int    hop_length,
    const float* window,
    const int    window_length,
    std::complex<float>* stft_result,
    const int n_stfts
    // clang-format on
) {
    const int n_fft_bins = window_length / 2 + 1;
    std::vector<float> input(window_length, 0.0F);
    std::vector<std::complex<float>> output(n_fft_bins);
    // clang-format off
    const pocketfft::shape_t  shape         = { static_cast<size_t>(window_length) };
    const pocketfft::stride_t stride_input  = { sizeof(float)                      };
    const pocketfft::stride_t stride_output = { sizeof(std::complex<float>)        };
    // clang-format on
    for (int i = 0; i < n_stfts; ++i) {
        const int pos = i * hop_length;
        const int n_audio_length = std::clamp(audio_length - pos, 0, window_length);
        int j = 0;
#if defined(__aarch64__)
        for (; j + 4 <= n_audio_length; j += 4) {
            // clang-format off
            const float32x4_t audio_input  = vld1q_f32(audio_data + pos + j);
            const float32x4_t window_input = vld1q_f32(window + j);
            const float32x4_t audio_output = vmulq_f32(audio_input, window_input);
            // clang-format on
            vst1q_f32(input.data() + j, audio_output);
        }
#endif
        for (; j < n_audio_length; ++j) {
            input[j] = audio_data[pos + j] * window[j];
        }
        for (; j < window_length; ++j) {
            input[j] = 0.0F;
        }
        pocketfft::r2c(shape, stride_input, stride_output, 0, true, input.data(), output.data(), 1.0F);
        std::memcpy(stft_result + i * n_fft_bins, output.data(), sizeof(std::complex<float>) * n_fft_bins);
    }
}

static void
compute_power(const std::complex<float>* stft_result, const int n_stfts, const int n_fft_bins, std::vector<float>& power) {
    int k = 0;
    for (int i = 0; i < n_fft_bins; ++i) {
        for (int j = 0; j < n_stfts - 1; ++j) {
            const std::complex<float>& value = stft_result[i * n_stfts + j];
            power[k] = value.real() * value.real() + value.imag() * value.imag();
            ++k;
        }
    }
}

static void matmul(
    // clang-format off
    const float* a_data,
    const float* b_data,
    std::vector<float>& c_data,
    const int rows_a,
    const int cols_a,
    const int cols_b
    // clang-format on
) {
    std::fill(c_data.begin(), c_data.end(), 0.0F);
    for (int i = 0; i < rows_a; ++i) {
        float* c = c_data.data() + i * cols_b;
        for (int k = 0; k < cols_a; ++k) {
            int j = 0;
            const float a = a_data[i * cols_a + k];
            if (a == 0.0F) {
                continue;
            }
            const float* b = b_data + k * cols_b;
#if defined(__aarch64__)
            const float32x4_t va = vdupq_n_f32(a);
            for (; j + 4 <= cols_b; j += 4) {
                const float32x4_t vb = vld1q_f32(b + j);
                const float32x4_t vc = vld1q_f32(c + j);
                vst1q_f32(c + j, vmlaq_f32(vc, va, vb));
            }
#endif
            for (; j < cols_b; ++j) {
                c[j] += a * b[j];
            }
        }
    }
}

static void clamp_and_log_max(std::vector<float>& mel_spec, const int rows, const int cols) {
    // clang-format off
    const float min_val        = 1e-10F;
    const float shift_value    = 4.0;
    const float scaling_factor = 1.0 / 4.0;
    // clang-format on
    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < rows * cols; ++i) {
        mel_spec[i] = std::log10f(std::max(min_val, mel_spec[i]));
        max_val = std::max(max_val, mel_spec[i]);
    }
    float threshold = max_val - 8.0;
    for (int i = 0; i < rows * cols; ++i) {
        mel_spec[i] = (std::max(mel_spec[i], threshold) + shift_value) * scaling_factor;
    }
}

static void log_mel_spectrogram(
    // clang-format off
    const float* audio_data,
    const int    audio_length,
    const float* mel_filters,
    const int    n_fft,
    const int    hop_length,
    const int    n_mels,
    std::vector<float>& mel_spec
    // clang-format on
) {
    const int n_stfts = audio_length / hop_length + 1;
    const int pad_size = audio_length + n_fft;
    const int n_fft_bins = n_fft / 2 + 1;
    const int rows_a = n_mels;
    const int cols_a = n_fft_bins;
    const int cols_b = n_stfts - 1;
    std::vector<float> window(n_fft);
    std::vector<float> pad_audio(pad_size);
    std::vector<std::complex<float>> stfts_result_o(n_fft_bins * n_stfts);
    std::vector<std::complex<float>> stfts_result_t(n_fft_bins * n_stfts);
    std::vector<float> power(n_fft_bins * (n_stfts - 1));
    hann_window(window, n_fft);
    reflect_pad(audio_data, audio_data + audio_length, pad_audio, n_fft / 2);
    stfts(pad_audio.data(), pad_size, hop_length, window.data(), n_fft, stfts_result_o.data(), n_stfts);
    caiwei::media::transpose(stfts_result_o.data(), stfts_result_t.data(), n_stfts, n_fft_bins);
    compute_power(stfts_result_t.data(), n_stfts, n_fft_bins, power);
    matmul(mel_filters, power.data(), mel_spec, rows_a, cols_a, cols_b);
    clamp_and_log_max(mel_spec, rows_a, cols_b);
}
