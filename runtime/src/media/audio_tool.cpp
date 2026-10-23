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

static void hann_window(std::vector<float> &window, int length)
{
    for (int i = 0; i < length; i++)
    {
        window[i] = 0.5 * (1 - cos(2 * M_PI * i / (length - 1)));
    }
}

static void reflect_pad(const std::vector<float> &audio, std::vector<float> &padded_audio, int pad_width)
{
    std::copy(audio.begin(), audio.end(), padded_audio.begin() + pad_width);
    std::reverse_copy(audio.begin(), audio.begin() + pad_width, padded_audio.begin());
    std::reverse_copy(audio.end() - pad_width, audio.end(), padded_audio.end() - pad_width);
}

#if defined(__aarch64__)
static void stfts(const std::vector<float> &audio, int audio_length, int window_length, int hop_length, const std::vector<float> &window, std::complex<float> *stft_result, int num_frames)
{
    std::vector<float> input(window_length, 0.0f);
    std::vector<std::complex<float>> output(window_length / 2 + 1);
    pocketfft::shape_t shape = {static_cast<size_t>(window_length)};
    pocketfft::stride_t stride_in = {sizeof(float)};
    pocketfft::stride_t stride_out = {sizeof(std::complex<float>)};
    for (int i = 0; i < num_frames; i++)
    {
        int start = i * hop_length;

        for (int j = 0; j < window_length - 3; j += 4)
        {
            if (start + j < audio_length)
            {
                float32x4_t in = vld1q_f32(audio.data() + start + j);
                float32x4_t win = vld1q_f32(window.data() + j);
                float32x4_t out = vmulq_f32(in, win);
                vst1q_f32(input.data() + j, out);
            }
            else
            {
                vst1q_f32(input.data() + j, vdupq_n_f32(0.0f));
            }
        }

        for (int j = window_length - window_length % 4; j < window_length; j++)
        {
            if (start + j < audio_length)
            {
                input[j] = audio[start + j] * window[j];
            }
            else
            {
                input[j] = 0.0f;
            }
        }

        pocketfft::r2c(shape, stride_in, stride_out, 0, true, input.data(), output.data(), 1.0f);
        memcpy(stft_result + i * (window_length / 2 + 1), output.data(), sizeof(std::complex<float>) * (window_length / 2 + 1));
    }
}
#else
static void stfts(const std::vector<float> &audio, int audio_length, int window_length, int hop_length, const std::vector<float> &window, std::complex<float> *stft_result, int num_frames)
{
    std::vector<float> input(window_length, 0.0f);
    std::vector<std::complex<float>> output(window_length / 2 + 1);
    pocketfft::shape_t shape = {static_cast<size_t>(window_length)};
    pocketfft::stride_t stride_in = {sizeof(float)};
    pocketfft::stride_t stride_out = {sizeof(std::complex<float>)};

    for (int i = 0; i < num_frames; i++)
    {
        int start = i * hop_length;

        for (int j = 0; j < window_length; j++)
        {
            if (start + j < audio_length)
            {
                input[j] = audio[start + j] * window[j];
            }
            else
            {
                input[j] = 0.0f;
            }
        }

        pocketfft::r2c(shape, stride_in, stride_out, 0, true, input.data(), output.data(), 1.0f);
        memcpy(stft_result + i * (window_length / 2 + 1), output.data(), sizeof(std::complex<float>) * (window_length / 2 + 1));
    }
}
#endif

static float compute_magnitude(const std::complex<float> &value)
{
    return value.real() * value.real() + value.imag() * value.imag();
}

static void compute_magnitudes(std::complex<float> *stft_result, int num_mel_filters, int num_frames, std::vector<float> &magnitudes)
{
    int k = 0;
    for (int i = 0; i < num_mel_filters; i++)
    {
        for (int j = 0; j < num_frames - 1; j++)
        {
            magnitudes[k] = compute_magnitude(stft_result[i * num_frames + j]);
            k++;
        }
    }
}

static void clamp_and_log_max(std::vector<float> &mel_spec, int rows, int cols)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }
    float min_val = 1e-10;
    float scaling_factor = 1.0 / 4.0;
    float shift_value = 4.0;

    float max_val = mel_spec[0];
    for (int i = 0; i < rows * cols; ++i)
    {
        float value = mel_spec[i];
        value = (value < min_val) ? min_val : value;
        mel_spec[i] = log10f(value);

        if (mel_spec[i] > max_val)
            max_val = mel_spec[i];
    }

    float threshold = max_val - 8.0;
    for (int i = 0; i < rows * cols; ++i)
    {
        mel_spec[i] = (std::max(mel_spec[i], threshold) + shift_value) * scaling_factor;
    }
}

static void transpose(std::complex<float> *input, int input_rows, int input_cols, std::complex<float> *output)
{
    for (int i = 0; i < input_rows; ++i)
    {
        for (int j = 0; j < input_cols; ++j)
        {
            int input_index = i * input_cols + j;
            int output_index = j * input_rows + i;

            output[output_index] = input[input_index];
        }
    }
}

#if defined(__aarch64__)
void matmul(const float* A, float* B, std::vector<float>& C, int ROWS_A, int COLS_A, int COLS_B) {
    for (int i = 0; i < ROWS_A; i++) {
        float* c_row = &C[i * COLS_B];
        std::fill(c_row, c_row + COLS_B, 0.0f);
        for (int k = 0; k < COLS_A; k++) {
            const float a = A[i * COLS_A + k];
            if (a == 0.0f) {
                continue;
            }
            const float* b_row = &B[k * COLS_B];
            const float32x4_t va = vdupq_n_f32(a);
            int j = 0;
            for (; j <= COLS_B - 4; j += 4) {
                const float32x4_t vb = vld1q_f32(b_row + j);
                const float32x4_t vc = vld1q_f32(c_row + j);
                vst1q_f32(c_row + j, vmlaq_f32(vc, va, vb));
            }
            for (; j < COLS_B; j++) {
                c_row[j] += a * b_row[j];
            }
        }
    }
}
#else
static void matmul(const float *A, float *B, std::vector<float> &C, int ROWS_A, int COLS_A, int COLS_B)
{
    std::fill(C.begin(), C.end(), 0.0f);

    for (int i = 0; i < ROWS_A; i++) {
        float* c_row = &C[i * COLS_B];
        for (int k = 0; k < COLS_A; k++) {
            float a_val = A[i * COLS_A + k];
            float* b_row = &B[k * COLS_B];
            for (int j = 0; j < COLS_B; j++) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}
#endif

static void log_mel_spectrogram(const float *audio_data, int audio_length, int cur_num_frames_of_stfts, const float *filters, int n_fft, int hop_length, int n_mels, std::vector<float> &mel_spec)
{
    int mels_filters_size = n_fft / 2 + 1;

    std::vector<float> window(n_fft);
    hann_window(window, n_fft);

    std::vector<float> audio(audio_data, audio_data + audio_length);
    int padded_size = audio_length + n_fft;
    std::vector<float> padded_audio(padded_size);
    reflect_pad(audio, padded_audio, n_fft / 2);

    std::vector<std::complex<float>> stfts_result(mels_filters_size * cur_num_frames_of_stfts);
    stfts(padded_audio, audio_length + n_fft, n_fft, hop_length, window, stfts_result.data(), cur_num_frames_of_stfts);

    std::vector<std::complex<float>> stfts_result_t(mels_filters_size * cur_num_frames_of_stfts);
    transpose(stfts_result.data(), cur_num_frames_of_stfts, mels_filters_size, stfts_result_t.data());

    std::vector<float> magnitudes(mels_filters_size * (cur_num_frames_of_stfts - 1));
    compute_magnitudes(stfts_result_t.data(), mels_filters_size, cur_num_frames_of_stfts, magnitudes);

    int ROWS_A = n_mels;
    int COLS_A = mels_filters_size;
    int COLS_B = cur_num_frames_of_stfts - 1;
    matmul(filters, magnitudes.data(), mel_spec, ROWS_A, COLS_A, COLS_B);

    clamp_and_log_max(mel_spec, ROWS_A, COLS_B);

}

void caiwei::audio::log_mel_spectrogram(const float *audio, int audio_length, const float *mel_filters, int n_fft, int hop_length, int n_mels, int max_audio_length, std::vector<float> &mel_spec, int *actual_length)
{
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
