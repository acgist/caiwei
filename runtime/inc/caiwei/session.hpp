/**
 * 会话
 * 
 * session -> context -> runtime
 */
#ifndef CAIWEI_RUNTIME_SESSION_HPP
#define CAIWEI_RUNTIME_SESSION_HPP

#include <future>
#include <string>
#include <vector>
#include <functional>

#include "caiwei/text_data.hpp"

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
    virtual std::string get() = 0;
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

class VideoPlaySession : public StatefulSession {
private:
    std::string type;
    std::string url;
public:
    VideoPlaySession(const std::string type, std::string url, Callback callback);
public:
    std::future<bool> get() override;
};

class EmbeddingsSession : public StatelessSession {
private:
    const caiwei::text::EmbeddingsRequest& request;
public:
    EmbeddingsSession(const caiwei::text::EmbeddingsRequest& request);
public:
    virtual std::string get() override;
};

class RerankingsSession : public StatelessSession {
private:
    const caiwei::text::RerankingsRequest& request;
public:
    RerankingsSession(const caiwei::text::RerankingsRequest& request);
public:
    virtual std::string get() override;
};

class CompletionsSession : public StatefulSession {
private:
    caiwei::text::CompletionsRequest& request;
public:
    CompletionsSession(caiwei::text::CompletionsRequest& request, Callback callback = nullptr);
public:
    std::string get_sync();
    std::future<bool> get() override;
};

} // session
} // caiwei

#endif // CAIWEI_RUNTIME_SESSION_HPP
