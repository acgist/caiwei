/**
 * 会话请求
 * 负责处理用户任务: 每个会话请求会申请一个运行上下文
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

using Callback = std::function<bool(const char* data, int length)>;

class StatefulSession : public Session {
protected:
    Callback callback;
public:
    StatefulSession(Callback callback);
public:
    virtual std::future<bool> get() = 0;
};

class VideoPlaySession : public StatefulSession {
private:
    const bool copy;
    const std::string type;
    const std::string url;
public:
    VideoPlaySession(bool copy, std::string type, std::string url, Callback callback);
public:
    std::future<bool> get() override;
protected:
    bool media_copy();
    bool media_codec();
};

class EmbeddingsSession : public StatelessSession {
private:
    caiwei::text::EmbeddingsRequest& request;
public:
    EmbeddingsSession(caiwei::text::EmbeddingsRequest& request);
public:
    virtual std::string get() override;
};

class RerankingsSession : public StatelessSession {
private:
    caiwei::text::RerankingsRequest& request;
public:
    RerankingsSession(caiwei::text::RerankingsRequest& request);
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
