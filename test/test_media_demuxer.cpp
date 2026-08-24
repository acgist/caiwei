#include "caiwei/media.hpp"
#include "caiwei/log.hpp"

int main() {
    caiwei::media::MediaDemuxer media_demuxer("file", "D://download/264.mp4", [](const caiwei::media::AudioFrame& frame) {
        std::printf("audioFrame: %" PRId64 " %" PRId64 " %" PRId32 "\n", frame.msec, frame.frames, frame.data_length);
        return true;
    }, [](const caiwei::media::VideoFrame& frame) {
        std::printf("videoFrame: %" PRId64 " %" PRId64 " %" PRId32 " %" PRId32 "x%" PRId32 "\n", frame.msec, frame.frames, frame.data_length, frame.width, frame.height);
        return true;
    });
    media_demuxer.open();
    return 0;
}