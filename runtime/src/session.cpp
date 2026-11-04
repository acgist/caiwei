#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/session.hpp"

caiwei::session::Session::Session() : id(caiwei::env::id()) {
    CW_LOG_I("打开会话: %s", id.c_str());
}

caiwei::session::Session::~Session() {
    CW_LOG_I("关闭会话: %s", id.c_str());
}

caiwei::session::StatefulSession::StatefulSession(Callback callback) : callback(std::move(callback)) {
}
