/**
 * 音频工具
 */
#ifndef CAIWEI_RUNTIME_AUDIO_TOOL_HPP
#define CAIWEI_RUNTIME_AUDIO_TOOL_HPP

#include <vector>

namespace caiwei {
namespace audio  {

int read_mel_filters(const char *fileName, float *data, int max_lines);

void audio_preprocess(const float *audio, int audio_length, float *mel_filters, int n_fft, int hop_length, int n_mels, int max_audio_length, std::vector<float> &x_mel, int *actual_len);

}
}

# endif // CAIWEI_RUNTIME_AUDIO_TOOL_HPP
