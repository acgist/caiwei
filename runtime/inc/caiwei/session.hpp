/**
 * 会话请求
 * 负责处理用户任务: 每个会话请求会申请一个运行上下文
 * 
 * session -> context -> runtime
 */
#ifndef CAIWEI_RUNTIME_SESSION_HPP
#define CAIWEI_RUNTIME_SESSION_HPP

#include <string>
#include <vector>
#include <functional>

#include "caiwei/text_data.hpp"

namespace caiwei  {
namespace session {
    
class Session {
public:
    const std::string id;
public:
    Session();
    virtual ~Session();
};

class StatelessSession : public Session {
public:
    virtual std::string get() = 0;
};

using Callback = std::function<bool(const char* data, int length)>;

class StatefulSession : public Session {
protected:
    Callback callback;
public:
    StatefulSession(Callback callback);
public:
    virtual bool get() = 0;
};

class VideoPlaySession : public StatefulSession {
private:
    const std::string url;
    const std::string type;
    const bool codec;
public:
    VideoPlaySession(std::string url, std::string type, bool codec, Callback callback);
public:
    bool get() override;
protected:
    bool media_codec();
    bool media_forward();
};

class EmbeddingsSession : public StatelessSession {
private:
    caiwei::text::EmbeddingsRequest& request;
public:
    EmbeddingsSession(caiwei::text::EmbeddingsRequest& request);
public:
    std::string get() override;
};

class RerankingsSession : public StatelessSession {
private:
    caiwei::text::RerankingsRequest& request;
public:
    RerankingsSession(caiwei::text::RerankingsRequest& request);
public:
    std::string get() override;
};

class CompletionsSession : public StatefulSession {
private:
    caiwei::text::CompletionsRequest& request;
public:
    CompletionsSession(caiwei::text::CompletionsRequest& request, Callback callback = nullptr);
public:
    std::string get_sync();
    bool get() override;
};

} // session
} // caiwei

#endif // CAIWEI_RUNTIME_SESSION_HPP
