#include "caiwei/media_tool.hpp"

#include <filesystem>

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

bool caiwei::media::check_file_security(const std::string& file, std::initializer_list<std::string> suffix) {
    std::filesystem::path path(file);
    if (!std::filesystem::exists(path)) {
        return false;
    }
    if (!std::filesystem::is_regular_file(path)) {
        return false;
    }
    if (!path.has_extension()) {
        return false;
    }
    // 可以添加安全目录校验更加安全
    auto iter = std::find(suffix.begin(), suffix.end(), path.extension().string());
    return iter != suffix.end();
}
