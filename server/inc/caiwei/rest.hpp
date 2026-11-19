#ifndef CAIWEI_SERVER_REST_HPP
#define CAIWEI_SERVER_REST_HPP

#ifndef CAIWEI_TIMEOUT_READ
#define CAIWEI_TIMEOUT_READ 60
#endif
#ifndef CAIWEI_TIMEOUT_WRITE
#define CAIWEI_TIMEOUT_WRITE 60
#endif
#ifndef CAIWEI_TIMEOUT_KEEP_ALIVE
#define CAIWEI_TIMEOUT_KEEP_ALIVE 60
#endif
#ifndef CAIWEI_PAYLOAD_MAX_LENGTH
#define CAIWEI_PAYLOAD_MAX_LENGTH (64 * 1024 * 1024)
#endif

namespace httplib {

class Server;

}  // namespace httplib

namespace caiwei {
namespace rest {

namespace content {
namespace type {

// clang-format off
inline constexpr const char* HTML  = "text/html; charset=utf-8";
inline constexpr const char* ICON  = "image/x-icon";
inline constexpr const char* JSON  = "application/json; charset=utf-8";
inline constexpr const char* EVENT = "text/event-stream; charset=utf-8";
// clang-format on

}  // namespace type
}  // namespace content

extern httplib::Server* server;

bool open();
void stop();
void register_api();

}  // namespace rest
}  // namespace caiwei

#endif  // CAIWEI_SERVER_REST_HPP
