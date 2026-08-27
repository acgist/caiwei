/**
 * REST API
 * 
 * 使用标准OpenAI接口
 * 
 * 0000: 成功
 * 0xxx: 系统错误
 * 1xxx: 网关错误（REST）
 * 2xxx: 媒体错误（MEDIA）
 * 3xxx: 推理错误（RUNTIME）
 * 9999: 未知错误
 */
#ifndef CAIWEI_REST_HPP
#define CAIWEI_REST_HPP

namespace httplib {

class Server;

} // httplib

namespace caiwei {
namespace rest   {

namespace content {
namespace type    {

    const char* const HTML  = "text/html";
    const char* const ICON  = "image/x-icon";
    const char* const JSON  = "application/json";
    const char* const EVENT = "text/event-stream";

}; // type
}; // content

extern httplib::Server* server;

extern void open();
extern void stop();
extern void rest_api();
extern void rest_text_api();
extern void rest_audio_api();
extern void rest_image_api();
extern void rest_video_api();

} // rest
} // caiwei

#endif // CAIWEI_REST_HPP
