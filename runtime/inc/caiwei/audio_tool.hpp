#ifndef CAIWEI_RUNTIME_AUDIO_TOOL_HPP
#define CAIWEI_RUNTIME_AUDIO_TOOL_HPP

#include <vector>

namespace caiwei {
namespace audio  {

void log_mel_spectrogram(
    const float* audio,
    int audio_length,
    const float* mel_filters,
    int n_fft,
    int hop_length,
    int n_mels,
    int max_audio_length,
    std::vector<float>& mel_spec,
    int* actual_length
);

} // audio
} // caiwei

# endif // CAIWEI_RUNTIME_AUDIO_TOOL_HPP
