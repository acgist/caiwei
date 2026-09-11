#include "caiwei/media.hpp"
#include "caiwei/caiwei.hpp"
#include "caiwei/context.hpp"
#include "caiwei/runtime.hpp"
#include "caiwei/manager.hpp"

void caiwei::init() {
    caiwei::media::init();
    caiwei::context::init();
    caiwei::runtime::init();
    caiwei::manager::init();
}

void caiwei::stop() {
    caiwei::media::stop();
    caiwei::context::stop();
    caiwei::runtime::stop();
    caiwei::manager::stop();
}
