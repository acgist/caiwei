#include "caiwei/env.hpp"
#include "caiwei/session.hpp"

caiwei::session::Session::Session() {
    this->id = caiwei::env::id();
}
caiwei::session::StatefulSession::StatefulSession(Callback callback) : callback(callback) {
}