#include "test.hpp"

#include "caiwei/type.hpp"
#include "caiwei/audio_tool.hpp"
#include "caiwei/image_tool.hpp"

extern "C" {
#include "libavcodec/avcodec.h"
}

#define N_FFT 400
#define HOP_LENGTH 160
#define MEL_FILTERS_PATH "mel_128_filters.txt"

[[maybe_unused]]
void test_log_mel_spectrogram() {
    caiwei::media::AudioFrame audio_frame;
    caiwei::media::MediaDemuxer media_demuxer("file", "caiwei.mp3");
    // caiwei::media::MediaDemuxer media_demuxer("file", "caiwei_noise.mp3");
    // caiwei::media::MediaDemuxer media_demuxer("file", "caiwei_small.mp3");
    media_demuxer.open(
        caiwei::media::AudioInfo(1, 16000, AV_SAMPLE_FMT_S16),
        caiwei::media::VideoInfo(640, 0, AV_PIX_FMT_RGB24),
        [&audio_frame](caiwei::media::AudioFrame& frame) {
            audio_frame.data.insert(audio_frame.data.end(), frame.data.begin(), frame.data.end());
            return true;
        },
        [](caiwei::media::VideoFrame& frame) {
            return true;
        }
    );
    // audio_frame.data.resize(0);
    // audio_frame.data.resize((N_FFT + HOP_LENGTH) * 2);
    const int n_mels = 128;
    std::vector<float> mel_filters{};
    caiwei::audio::load_mel_filters(n_mels, N_FFT, mel_filters);
    const int n_frames = audio_frame.data.size() * sizeof(uint8_t) / sizeof(int16_t);
    int n_frame = floor((n_frames - N_FFT) / HOP_LENGTH ) + 1;
    std::vector<float> audio_data(n_frames);
    const int16_t* audio_data_ptr = reinterpret_cast<const int16_t*>(audio_frame.data.data());
    std::transform(audio_data_ptr, audio_data_ptr + n_frames, audio_data.data(), [](int16_t v) {
        return static_cast<float>(v) / 32768.0F;
    });
    std::vector<float> padding_feature(n_mels * n_frame, 0.0F);
    caiwei::audio::log_mel_spectrogram(audio_data.data(), audio_data.size(), mel_filters.data(), N_FFT, HOP_LENGTH, n_mels, n_frame * HOP_LENGTH, padding_feature);
    stbi_write_hdr("./log_mel_spectrogram.hdr", n_frame, n_mels, 1, padding_feature.data());
}

int main(int argc, char* argv[]) {
    caiwei::test::init_test();
    test_log_mel_spectrogram();
    caiwei::test::stop_test();
    return 0;
}
