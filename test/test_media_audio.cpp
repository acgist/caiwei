#include "test.hpp"

#include "stb/stb_image_write.h"

[[maybe_unused]]
static void test_log_mel_spectrogram() {
    std::vector<uint8_t> audio_data;
    caiwei::media::MediaDemuxer media_demuxer("file", "caiwei.mp3");
    bool success = media_demuxer.open(
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
    CAIWEI_ASSERT(success);
    const int n_mels = 128;
    std::vector<float> mel_filters {};
    CAIWEI_ASSERT(caiwei::media::load_mel_filters(n_mels, QWEN_ASR_N_FFT, mel_filters));
    const int n_frames = audio_data.size() / sizeof(int16_t);
    CAIWEI_ASSERT(n_frames >= (QWEN_ASR_N_FFT + QWEN_ASR_HOP_LENGTH) * 2);
    const int n_frame = (n_frames - QWEN_ASR_N_FFT) / QWEN_ASR_HOP_LENGTH + 1;
    std::vector<float> audio_data_f32(n_frames);
    const int16_t* audio_data_ptr = reinterpret_cast<const int16_t*>(audio_data.data());
    std::transform(audio_data_ptr, audio_data_ptr + n_frames, audio_data_f32.data(), [](int16_t v) {
        return static_cast<float>(v) / 32768.0F;
    });
    std::vector<float> audio_feature(n_mels * n_frame, 0.0F);
    const bool mel_success = caiwei::media::log_mel_spectrogram(
        audio_data_f32.data(),
        audio_data_f32.size(),
        mel_filters.data(),
        QWEN_ASR_N_FFT,
        QWEN_ASR_HOP_LENGTH,
        n_mels,
        n_frame * QWEN_ASR_HOP_LENGTH,
        audio_feature
    );
    CAIWEI_ASSERT(mel_success);
    CAIWEI_ASSERT(stbi_write_hdr("./log_mel_spectrogram.hdr", n_frame, n_mels, 1, audio_feature.data()) != 0);
}

int main() {
    caiwei::test::init_test();
    test_log_mel_spectrogram();
    caiwei::test::stop_test();
    return 0;
}
