#include "caiwei/media.hpp"

extern "C" {

#include "libavformat/avformat.h"
#include "libavdevice/avdevice.h"
    
}

void caiwei::media::init() {
    avformat_network_init();
    avdevice_register_all();
}

void caiwei::media::stop() {
    avformat_network_deinit();
}

caiwei::media::Frame::Frame(uint32_t size) {
    this->data.resize(size);
}
