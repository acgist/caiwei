#include "caiwei/media_tool.hpp"

caiwei::media::AudioInfo::AudioInfo(int channels, int sample_rate, int format) {
    this->channels    = channels;
    this->sample_rate = sample_rate;
    this->format      = format;
}

caiwei::media::AudioInfo::AudioInfo(int channels, int sample_rate, int bytes_per_sample, int format) {
    this->channels         = channels;
    this->sample_rate      = sample_rate;    
    this->bytes_per_sample = bytes_per_sample;
    this->format           = format;
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
