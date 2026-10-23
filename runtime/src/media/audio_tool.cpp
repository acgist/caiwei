#include "caiwei/log.hpp"
#include "caiwei/audio_tool.hpp"

#include <cmath>
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

static void hann_window(std::vector<float>& window, int length) {
    for (int i = 0; i < length; ++i) {
        window[i] = 0.5 * (1 - std::cos(2 * M_PI * i / (length - 1)));
    }
}

static void reflect_pad(const std::vector<float>& audio, std::vector<float>& pad_audio, int pad_width) {
    std::copy(audio.begin(), audio.end(), pad_audio.begin() + pad_width);
    std::reverse_copy(audio.begin(), audio.begin() + pad_width, pad_audio.begin());
    std::reverse_copy(audio.end() - pad_width, audio.end(), pad_audio.end() - pad_width);
}

static void stfts(
    const std::vector<float>& audio,
    int audio_length,
    int window_length,
    int hop_length,
    const std::vector<float>& window,
    std::complex<float>* stft_result,
    int num_frames
) {
    std::vector<float> input(window_length, 0.0F);
    std::vector<std::complex<float>> output(window_length / 2 + 1);
    pocketfft::shape_t  shape      = { static_cast<size_t>(window_length) };
    pocketfft::stride_t stride_in  = { sizeof(float)                      };
    pocketfft::stride_t stride_out = { sizeof(std::complex<float>)        };
    for (int i = 0; i < num_frames; ++i) {
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

static void compute_magnitudes(std::complex<float>* stft_result, int num_mel_filters, int num_frames, std::vector<float>& magnitudes) {
    int k = 0;
    for (int i = 0; i < num_mel_filters; ++i) {
        for (int j = 0; j < num_frames - 1; ++j) {
            magnitudes[k] = compute_magnitude(stft_result[i * num_frames + j]);
            ++k;
        }
    }
}

static void clamp_and_log_max(std::vector<float>& mel_spec, int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    const float min_val        = 1e-10;
    const float shift_value    = 4.0;
    const float scaling_factor = 1.0 / 4.0;
    float max_val = mel_spec[0];
    for (int i = 0; i < rows * cols; ++i) {
        mel_spec[i] = std::log10f(std::max(min_val, mel_spec[i]));
        if (mel_spec[i] > max_val) {
            max_val = mel_spec[i];
        }
    }
    float threshold = max_val - 8.0;
    for (int i = 0; i < rows * cols; ++i) {
        mel_spec[i] = (std::max(mel_spec[i], threshold) + shift_value) * scaling_factor;
    }
}

static void transpose(std::complex<float>* input, int input_rows, int input_cols, std::complex<float>* output) {
    for (int i = 0; i < input_rows; ++i) {
        for (int j = 0; j < input_cols; ++j) {
            const int input_index  = i * input_cols + j;
            const int output_index = j * input_rows + i;
            output[output_index] = input[input_index];
        }
    }
}

#if defined(__aarch64__)
static void matmul(const float* A, const float* B, std::vector<float>& C, int ROWS_A, int COLS_A, int COLS_B) {
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
static void matmul(const float* A, const float* B, std::vector<float>& C, int ROWS_A, int COLS_A, int COLS_B) {
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
    int audio_length,
    int cur_num_frames_of_stfts,
    const float* filters,
    int n_fft,
    int hop_length,
    int n_mels,
    std::vector<float>& mel_spec
) {
    const int mels_filters_size = n_fft / 2 + 1;
    std::vector<float> window(n_fft);
    hann_window(window, n_fft);
    std::vector<float> audio(audio_data, audio_data + audio_length);
    const int pad_size = audio_length + n_fft;
    std::vector<float> pad_audio(pad_size);
    reflect_pad(audio, pad_audio, n_fft / 2);
    std::vector<std::complex<float>> stfts_result(mels_filters_size * cur_num_frames_of_stfts);
    stfts(pad_audio, pad_size, n_fft, hop_length, window, stfts_result.data(), cur_num_frames_of_stfts);
    std::vector<std::complex<float>> stfts_result_t(mels_filters_size * cur_num_frames_of_stfts);
    transpose(stfts_result.data(), cur_num_frames_of_stfts, mels_filters_size, stfts_result_t.data());
    std::vector<float> magnitudes(mels_filters_size * (cur_num_frames_of_stfts - 1));
    compute_magnitudes(stfts_result_t.data(), mels_filters_size, cur_num_frames_of_stfts, magnitudes);
    const int ROWS_A = n_mels;
    const int COLS_A = mels_filters_size;
    const int COLS_B = cur_num_frames_of_stfts - 1;
    matmul(filters, magnitudes.data(), mel_spec, ROWS_A, COLS_A, COLS_B);
    clamp_and_log_max(mel_spec, ROWS_A, COLS_B);
}

void caiwei::audio::log_mel_spectrogram(
    const float* audio,
    int audio_length,
    const float* mel_filters,
    int n_fft,
    int hop_length,
    int n_mels,
    int max_audio_length,
    std::vector<float>& mel_spec,
    int* actual_length
) {
    if (audio == nullptr || mel_filters == nullptr) {
        CW_LOG_W("参数错误");
        return;
    }
    std::vector<float> audio_data(max_audio_length);
    std::copy(audio, audio + max_audio_length, audio_data.begin());
    int cur_num_frames_of_stfts = max_audio_length / hop_length + 1;
    ::log_mel_spectrogram(audio_data.data(), max_audio_length, cur_num_frames_of_stfts, mel_filters, n_fft, hop_length, n_mels, mel_spec);
    *actual_length = max_audio_length / hop_length;
}
