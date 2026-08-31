/**
 * 会话
 * 
 * session -> context -> runtime
 */
#ifndef CAIWEI_SESSION_HPP
#define CAIWEI_SESSION_HPP

#include <future>
#include <string>
#include <vector>
#include <functional>

namespace caiwei  {
namespace session {
    
class Session {
public:
    std::string id;
public:
    Session();
    virtual ~Session();
};

class StatelessSession : public Session {
public:
    virtual std::vector<char> get() = 0;
};

// 返回连接是否断开标识是否需要继续处理
using Callback = std::function<bool(const char* type, const char* data, size_t length)>;

class StatefulSession : public Session {
protected:
    bool running = false;
    Callback callback;
public:
    StatefulSession(Callback callback);
public:
    virtual std::future<bool> get() = 0;
};

class AudioAsrSession : public StatefulSession {

};

class AudioTtsSession : public StatefulSession {

};

class ImageYoloSession : public StatefulSession {

};

class VideoYoloSession : public StatefulSession {

};

class VideoPlaySession : public StatefulSession {
private:
    std::string type;
    std::string url;
public:
    VideoPlaySession(const std::string type, std::string url, Callback callback);
public:
    std::future<bool> get() override;
};

class RerankingsSession : public StatelessSession {

};

class EmbeddingsSession : public StatelessSession {

};

class ChatCompletionsSession : public StatefulSession {

};

} // session
} // caiwei

#endif // CAIWEI_SESSION_HPP
