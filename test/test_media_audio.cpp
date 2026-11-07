#include "test.hpp"

#include "caiwei/type.hpp"

#define N_FFT 400
#define HOP_LENGTH 160
#define MEL_FILTERS_PATH "mel_128_filters.txt"

[[maybe_unused]]
void test_log_mel_spectrogram() {
    std::vector<uint8_t> audio_data;
    caiwei::media::MediaDemuxer media_demuxer("file", "caiwei.mp3");
    media_demuxer.open(
        caiwei::media::AUDIO_INFO_DECODE,
        caiwei::media::VIDEO_INFO_DECODE,
        [&audio_data](caiwei::media::AudioFrame& frame) {
            audio_data.insert(audio_data.end(), frame.data.begin(), frame.data.end());
            return true;
        },
        [](caiwei::media::VideoFrame& /* frame */) {
            return true;
        }
    );
    const int n_mels = 128;
    std::vector<float> mel_filters{};
    if (!caiwei::audio::load_mel_filters(n_mels, N_FFT, mel_filters)) {
        CW_LOG_W("读取mel_filters文件失败: %s", MEL_FILTERS_PATH);
        return;
    }
    const int n_frames = audio_data.size() * sizeof(uint8_t) / sizeof(int16_t);
    if (n_frames < (N_FFT + HOP_LENGTH) * 2) {
        CW_LOG_W("音频数据长度不足: %d", n_frames);
        return;
    }
    const int n_frame = std::floor((n_frames - N_FFT) / HOP_LENGTH ) + 1;
    std::vector<float> audio_data_f32(n_frames);
    const int16_t* audio_data_ptr = reinterpret_cast<const int16_t*>(audio_data.data());
    std::transform(audio_data_ptr, audio_data_ptr + n_frames, audio_data_f32.data(), [](int16_t v) {
        return static_cast<float>(v) / 32768.0F;
    });
    std::vector<float> audio_feature(n_mels * n_frame, 0.0F);
    caiwei::audio::log_mel_spectrogram(
        audio_data_f32.data(),
        audio_data_f32.size(),
        mel_filters.data(),
        N_FFT,
        HOP_LENGTH,
        n_mels,
        n_frame * HOP_LENGTH,
        audio_feature
    );
    stbi_write_hdr("./log_mel_spectrogram.hdr", n_frame, n_mels, 1, audio_feature.data());
}

int main() {
    caiwei::test::init_test();
    test_log_mel_spectrogram();
    caiwei::test::stop_test();
    return 0;
}
