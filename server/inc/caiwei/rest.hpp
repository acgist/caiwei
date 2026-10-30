#ifndef CAIWEI_SERVER_REST_HPP
#define CAIWEI_SERVER_REST_HPP

namespace httplib {

class Server;

} // httplib

namespace caiwei {
namespace rest   {

namespace content {
namespace type    {

    inline constexpr const char* HTML  = "text/html";
    inline constexpr const char* ICON  = "image/x-icon";
    inline constexpr const char* JSON  = "application/json";
    inline constexpr const char* EVENT = "text/event-stream";

}; // type
}; // content

extern httplib::Server* server;

bool open();
void stop();
void register_api();

} // rest
} // caiwei

#endif // CAIWEI_SERVER_REST_HPP
