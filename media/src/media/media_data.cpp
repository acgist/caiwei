#include "caiwei/media_data.hpp"

caiwei::media::Frame::Frame(uint32_t size) {
    this->data.resize(size);
}

caiwei::media::Frame::~Frame() {
}

caiwei::media::AudioFrame::AudioFrame(uint32_t size) : Frame(size) {
}

caiwei::media::ImageFrame::ImageFrame(uint32_t size) : Frame(size) {
}

caiwei::media::VideoFrame::VideoFrame(uint32_t size) : ImageFrame(size) {
}
