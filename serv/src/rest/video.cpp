#include "caiwei/rest.hpp"
#include "caiwei/log.hpp"
#include "caiwei/session.hpp"

#include "httplib.h"

#include <mutex>

static void video_play();

void caiwei::rest::rest_video_api() {
    video_play();
}

static void video_play() {
    caiwei::rest::server->Get("/v1/video/play", [](const httplib::Request& request, httplib::Response& response) {
        response.set_chunked_content_provider(caiwei::rest::content::type::EVENT, [&](size_t /*offset*/, httplib::DataSink& sink) {
            auto url  = request.get_param_value("url");
            auto type = request.get_param_value("type");
            caiwei::session::VideoPlaySession session(type, url, [&sink](const char* type, const char* data, size_t length) {
                if (sink.is_writable()) {
                    sink.write("event: ", 7);
                    sink.write(type, std::strlen(type));
                    sink.write("\n", 1);
                    sink.write("data: ", 6);
                    if(length > 0) {
                        sink.write(data, length);
                    }
                    sink.write("\n\n", 2);
                }
                return sink.is_writable();
            });
            session.get().get();
            sink.done();
            return true;
        });
    });
}