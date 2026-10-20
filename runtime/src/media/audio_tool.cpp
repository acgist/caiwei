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

static void pad_mel_spec(const std::vector<float> input, int rows_input, int cols_input, std::vector<float> &output, int cols_output)
{
    for (int i = 0; i < rows_input; ++i)
    {
        std::copy(input.begin() + i * cols_input, input.begin() + (i + 1) * cols_input, output.begin() + i * cols_output);
    }
}

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
        int end = start + window_length;
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
        int end = start + window_length;

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

void transpose(std::complex<float> *input, int input_rows, int input_cols, std::complex<float> *output)
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
void matmul(const float *A, float *B, std::vector<float> &C, int ROWS_A, int COLS_A, int COLS_B)
{
    int k_start = COLS_A, k_end = 0;
    for (auto i = 0; i < ROWS_A; i++)
    {
        for (auto k = 0; k < COLS_A; k++)
        {
            if (A[i * COLS_A + k] != 0.0f)
            {
                k_start = k;
                break;
            }
        }
        for (auto k = COLS_A - 1; k > 0; k--)
        {
            if (A[i * COLS_A + k] != 0.0f)
            {
                k_end = k;
                break;
            }
        }

        int k_diff = k_end - k_start + 1;
        if (k_diff % 4)
        {
            k_diff = 4 - (k_diff % 4);
            if (k_start > k_diff)
                k_start -= k_diff;
            else if (k_end < (COLS_A - k_diff))
                k_end += k_diff;
        }
        k_diff = (k_end - k_start) % 4;
        for (auto j = 0; j < COLS_B; j++)
        {
            float32x4_t v_tot = vdupq_n_f32(0.0f);
            for (auto k = k_start; k <= k_end - 3; k += 4)
            {
                float32x4_t vq_mat1 = vld1q_f32(&A[i * COLS_A + k]);
                float32x4_t vq_mat2 = vld1q_f32(&B[k * COLS_B + j]);
                v_tot = vmlaq_f32(v_tot, vq_mat1, vq_mat2);
            }

            float32x2_t tmp = vadd_f32(vget_high_f32(v_tot), vget_low_f32(v_tot));
            C[i * COLS_B + j] = vget_lane_f32(tmp, 0) + vget_lane_f32(tmp, 1);
        }
    }
}
#else
void matmul(const float *A, float *B, std::vector<float> &C, int ROWS_A, int COLS_A, int COLS_B)
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
    if (audio_length >= max_audio_length)
    {
        std::vector<float> audio_data(max_audio_length);
        std::copy(audio, audio + max_audio_length, audio_data.begin());
        int cur_num_frames_of_stfts = max_audio_length / hop_length + 1;
        ::log_mel_spectrogram(audio_data.data(), max_audio_length, cur_num_frames_of_stfts, mel_filters, n_fft, hop_length, n_mels, mel_spec);
        *actual_length = max_audio_length / hop_length;
    }
    else
    {
        int cur_num_frames_of_stfts = audio_length / hop_length + 1;
        int mel_spec_rows = n_mels;
        int mel_spec_cols = cur_num_frames_of_stfts - 1;
        int mel_spec_cols_pad = max_audio_length / hop_length;
        std::vector<float> cur_mel_spec(mel_spec_rows * mel_spec_cols, 0.0F);
        ::log_mel_spectrogram(audio, audio_length, cur_num_frames_of_stfts, mel_filters, n_fft, hop_length, n_mels, cur_mel_spec);
        pad_mel_spec(cur_mel_spec, mel_spec_rows, mel_spec_cols, mel_spec, mel_spec_cols_pad);
        *actual_length = mel_spec_cols;
    }
}
