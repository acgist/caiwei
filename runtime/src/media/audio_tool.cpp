#include "caiwei/log.hpp"
#include "caiwei/audio_tool.hpp"

#include <cmath>
#include <limits>
#include <vector>
#include <complex>
#include <cstring>
#include <algorithm>

#if defined(__aarch64__)
#include "arm_neon.h"
#endif

#if CAIWEI_OS_WIN
#include <corecrt_math_defines.h>
#endif

#include "pocketfft_hdronly/pocketfft_hdronly.h"

#define MEL_FILTERS_PATH "mel_128_filters.txt"

void caiwei::audio::load_mel_filters(
    const int n_mels,
    const int n_fft,
    std::vector<float>& mel_filters
) {
    FILE* file = fopen(MEL_FILTERS_PATH, "r");
    if (file == nullptr) {
        CW_LOG_W("读取mel_filters文件失败: %s", MEL_FILTERS_PATH);
        return;
    }
    const int n_mels_filters = n_fft / 2 + 1;
    const int size           = n_mels * n_mels_filters;
    mel_filters.resize(size);
    auto* data = mel_filters.data();
    int line = 0;
    while (line < size && fscanf(file, "%f", &data[line]) == 1) {
        line++;
    }
    fclose(file);
}

static void hann_window(std::vector<float>& window, int length) {
    for (int i = 0; i < length; ++i) {
        window[i] = 0.5 * (1 - std::cos(2 * M_PI * i / (length - 1)));
    }
}

static void reflect_pad(const std::vector<float>& audio, std::vector<float>& pad_audio, const int pad_width) {
    std::copy(audio.begin(), audio.end(), pad_audio.begin() + pad_width);
    std::reverse_copy(audio.begin(), audio.begin() + pad_width, pad_audio.begin());
    std::reverse_copy(audio.end() - pad_width, audio.end(), pad_audio.end() - pad_width);
}

static void stfts(
    const std::vector<float>& audio,
    const int audio_length,
    const int window_length,
    const int hop_length,
    const std::vector<float>& window,
    std::complex<float>* stft_result,
    const int n_stfts
) {
    std::vector<float> input(window_length, 0.0F);
    std::vector<std::complex<float>> output(window_length / 2 + 1);
    pocketfft::shape_t  shape      = { static_cast<size_t>(window_length) };
    pocketfft::stride_t stride_in  = { sizeof(float)                      };
    pocketfft::stride_t stride_out = { sizeof(std::complex<float>)        };
    for (int i = 0; i < n_stfts; ++i) {
        int pos = i * hop_length;
        #if defined(__aarch64__)
        for (int j = 0; j < window_length - 3; j += 4) {
            if (pos + j < audio_length) {
                float32x4_t in  = vld1q_f32(audio.data() + pos + j);
                float32x4_t win = vld1q_f32(window.data() + j);
                float32x4_t out = vmulq_f32(in, win);
                vst1q_f32(input.data() + j, out);
            } else {
                vst1q_f32(input.data() + j, vdupq_n_f32(0.0F));
            }
        }
        for (int j = window_length - window_length % 4; j < window_length; ++j) {
            if (pos + j < audio_length) {
                input[j] = audio[pos + j] * window[j];
            } else {
                input[j] = 0.0F;
            }
        }
        #else
        for (int j = 0; j < window_length; ++j) {
            if (pos + j < audio_length) {
                input[j] = audio[pos + j] * window[j];
            } else {
                input[j] = 0.0F;
            }
        }
        #endif
        pocketfft::r2c(shape, stride_in, stride_out, 0, true, input.data(), output.data(), 1.0F);
        std::memcpy(stft_result + i * (window_length / 2 + 1), output.data(), sizeof(std::complex<float>) * (window_length / 2 + 1));
    }
}

static float compute_magnitude(const std::complex<float>& value) {
    return value.real() * value.real() + value.imag() * value.imag();
}

static void compute_magnitudes(const std::complex<float>* stft_result, const int n_mels_filters, const int n_stfts, std::vector<float>& magnitudes) {
    int k = 0;
    for (int i = 0; i < n_mels_filters; ++i) {
        for (int j = 0; j < n_stfts - 1; ++j) {
            magnitudes[k] = compute_magnitude(stft_result[i * n_stfts + j]);
            ++k;
        }
    }
}

static void clamp_and_log_max(std::vector<float>& mel_spec, const int rows, const int cols) {
    const float min_val        = 1e-10;
    const float shift_value    = 4.0;
    const float scaling_factor = 1.0 / 4.0;
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

static void transpose(const std::complex<float>* input, const int rows, int const cols, std::complex<float>* output) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const int input_index  = i * cols + j;
            const int output_index = j * rows + i;
            output[output_index] = input[input_index];
        }
    }
}

#if defined(__aarch64__)
static void matmul(const float* A, const float* B, std::vector<float>& C, const int ROWS_A, const int COLS_A, const int COLS_B) {
    std::fill(C.begin(), C.end(), 0.0F);
    for (int i = 0; i < ROWS_A; ++i) {
        float* c = &C[i * COLS_B];
        for (int k = 0; k < COLS_A; ++k) {
            const float a = A[i * COLS_A + k];
            if (a == 0.0F) {
                continue;
            }
            const float* b = &B[k * COLS_B];
            const float32x4_t va = vdupq_n_f32(a);
            int j = 0;
            for (; j <= COLS_B - 4; j += 4) {
                const float32x4_t vb = vld1q_f32(b + j);
                const float32x4_t vc = vld1q_f32(c + j);
                vst1q_f32(c + j, vmlaq_f32(vc, va, vb));
            }
            for (; j < COLS_B; j++) {
                c[j] += a * b[j];
            }
        }
    }
}
#else
static void matmul(const float* A, const float* B, std::vector<float>& C, const int ROWS_A, const int COLS_A, const int COLS_B) {
    std::fill(C.begin(), C.end(), 0.0F);
    for (int i = 0; i < ROWS_A; ++i) {
        float* c = &C[i * COLS_B];
        for (int k = 0; k < COLS_A; ++k) {
            const float a = A[i * COLS_A + k];
            if (a == 0.0F) {
                continue;
            }
            const float* b = &B[k * COLS_B];
            for (int j = 0; j < COLS_B; ++j) {
                c[j] += a * b[j];
            }
        }
    }
}
#endif

static void log_mel_spectrogram(
    const float* audio_data,
    const int    audio_length,
    const float* filters,
    const int    n_fft,
    const int    hop_length,
    const int    n_mels,
    std::vector<float>& mel_spec
) {
    const int n_stfts = audio_length / hop_length + 1;
    const int pad_size = audio_length + n_fft;
    const int n_mels_filters = n_fft / 2 + 1;
    const int ROWS_A = n_mels;
    const int COLS_A = n_mels_filters;
    const int COLS_B = n_stfts - 1;
    std::vector<float> window(n_fft);
    std::vector<float> audio(audio_data, audio_data + audio_length);
    std::vector<float> pad_audio(pad_size);
    std::vector<std::complex<float>> stfts_result(n_mels_filters * n_stfts);
    std::vector<std::complex<float>> stfts_result_t(n_mels_filters * n_stfts);
    std::vector<float> magnitudes(n_mels_filters * (n_stfts - 1));
    hann_window(window, n_fft);
    reflect_pad(audio, pad_audio, n_fft / 2);
    stfts(pad_audio, pad_size, n_fft, hop_length, window, stfts_result.data(), n_stfts);
    transpose(stfts_result.data(), n_stfts, n_mels_filters, stfts_result_t.data());
    compute_magnitudes(stfts_result_t.data(), n_mels_filters, n_stfts, magnitudes);
    matmul(filters, magnitudes.data(), mel_spec, ROWS_A, COLS_A, COLS_B);
    clamp_and_log_max(mel_spec, ROWS_A, COLS_B);
}

void caiwei::audio::log_mel_spectrogram(
    const float* audio,
    const int    audio_length,
    const float* mel_filters,
    const int    n_fft,
    const int    hop_length,
    const int    n_mels,
    const int    max_audio_length,
    std::vector<float>& mel_spec
) {
    if (audio_length < max_audio_length) {
        return;
    }
    std::vector<float> audio_data(max_audio_length);
    std::copy(audio, audio + max_audio_length, audio_data.begin());
    ::log_mel_spectrogram(audio_data.data(), max_audio_length, mel_filters, n_fft, hop_length, n_mels, mel_spec);
}
