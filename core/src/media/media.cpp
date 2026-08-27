#include "caiwei/media.hpp"

caiwei::media::Frame::Frame(uint32_t size) {
    this->data.resize(size);
}

caiwei::media::AudioInfo::AudioInfo(int channel, int sample_rate, int format) {
    this->channel     = channel;
    this->sample_rate = sample_rate;
    this->format      = format;
}

caiwei::media::VideoInfo::VideoInfo(int width, int height, int format) {
    this->width  = width;
    this->height = height;
    this->format = format;
}

caiwei::media::VideoInfo::VideoInfo(int fps, int width, int height, int format) {
    this->fps    = fps;
    this->width  = width;
    this->height = height;
    this->format = format;
}
