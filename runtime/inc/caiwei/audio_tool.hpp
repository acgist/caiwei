#ifndef CAIWEI_RUNTIME_AUDIO_TOOL_HPP
#define CAIWEI_RUNTIME_AUDIO_TOOL_HPP

#include <vector>

#ifndef N_FFT
#define N_FFT 400
#endif
#ifndef HOP_LENGTH
#define HOP_LENGTH 160
#endif
#ifndef DOWN_SAMPLE_RATE
#define DOWN_SAMPLE_RATE 8
#endif

namespace caiwei {
namespace audio  {

bool load_mel_filters(
    const int n_mels,
    const int n_fft,
    std::vector<float>& mel_filters
);

bool log_mel_spectrogram(
    const float* audio,
    int audio_length,
    const float* mel_filters,
    int n_fft,
    int hop_length,
    int n_mels,
    int max_audio_length,
    std::vector<float>& mel_spec
);

} // audio
} // caiwei

#endif // CAIWEI_RUNTIME_AUDIO_TOOL_HPP
