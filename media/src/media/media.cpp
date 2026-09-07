#include "caiwei/media.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE2_IMPLEMENTATION

#include "stb/stb_image.h"
#include "stb/stb_image_resize2.h"

caiwei::media::AudioInfo::AudioInfo(int channels, int sample_rate, int format) {
    this->channels    = channels;
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
