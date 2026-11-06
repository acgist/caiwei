#include "caiwei/log.hpp"
#include "caiwei/json.hpp"
#include "caiwei/rest.hpp"
#include "caiwei/context.hpp"
#include "caiwei/session.hpp"
#include "caiwei/text_data.hpp"

#include <cstring>

#include "httplib.h"

static void get_health();
static void get_models();
static void get_video_play();
static void post_embeddings();
static void post_rerankings();
static void post_chat_completions();

void caiwei::rest::register_api() {
    get_health();
    get_models();
    get_video_play();
    post_embeddings();
    post_rerankings();
    post_chat_completions();
}

static void get_health() {
    caiwei::rest::server->Get("/v1/health", [](const httplib::Request& /* request */, httplib::Response& response) {
        response.set_content(caiwei::json::buildResponse("running"), caiwei::rest::content::type::JSON);
    });
}

static void get_models() {
    caiwei::rest::server->Get("/v1/models", [](const httplib::Request& /* request */, httplib::Response& response) {
        nlohmann::json body;
        auto list = caiwei::context::get_all_context_info();
        for (const auto& value : list) {
            body.push_back({
                {"type", value->type_name},
                {"name", value->name     },
            });
        }
        response.set_content(caiwei::json::buildResponse(body), caiwei::rest::content::type::JSON);
    });
}

static void get_video_play() {
    caiwei::rest::server->Get("/v1/video/play", [](const httplib::Request& request, httplib::Response& response) {
        auto url   = request.get_param_value("url");
        auto type  = request.get_param_value("type");
        auto codec = request.get_param_value("codec");
        if (!caiwei::text::check_security(url, type)) {
            response.status = httplib::StatusCode::BadRequest_400;
            response.set_content(caiwei::json::buildResponse("1400", "数据错误"), caiwei::rest::content::type::JSON);
            return;
        }
        response.set_chunked_content_provider(caiwei::rest::content::type::EVENT, [
            url   = std::move(url),
            type  = std::move(type),
            codec = std::move(codec)
        ](size_t /* offset */, httplib::DataSink& sink) mutable -> bool {
            caiwei::session::VideoPlaySession session(std::move(url), std::move(type), codec == "true", [&sink](const char* data, int length) {
                const bool writable = sink.is_writable();
                if (writable) {
                    if (data == nullptr) {
                        return writable;
                    } else {
                        sink.write("data: ", 6);
                        if (length <= 0) {
                            length = std::strlen(data);
                        }
                        sink.write(data, length);
                        sink.write("\n\n", 2);
                    }
                }
                return writable;
            });
            if (!session.get()) {
                CW_LOG_W("媒体播放失败");
            }
            sink.done();
            return false;
        });
    });
}

static void post_embeddings() {
    caiwei::rest::server->Post("/v1/embeddings", [](const httplib::Request& request, httplib::Response& response) {
        auto embedding = caiwei::text::json_to_embeddings(request.body);
        caiwei::session::EmbeddingsSession session(embedding);
        response.set_content(session.get(), caiwei::rest::content::type::JSON);
    });
}

static void post_rerankings() {
    caiwei::rest::server->Post("/v1/rerankings", [](const httplib::Request& request, httplib::Response& response) {
        auto reranking = caiwei::text::json_to_rerankings(request.body);
        caiwei::session::RerankingsSession session(reranking);
        response.set_content(session.get(), caiwei::rest::content::type::JSON);
    });
}

static void post_chat_completions() {
    caiwei::rest::server->Post("/v1/chat/completions", [](const httplib::Request& request, httplib::Response& response) {
        auto completions = caiwei::text::json_to_completions(request.body);
        if (!caiwei::text::check_security(completions)) {
            response.status = httplib::StatusCode::BadRequest_400;
            response.set_content(caiwei::json::buildResponse("1400", "数据错误"), caiwei::rest::content::type::JSON);
            return;
        }
        if (completions.stream) {
            response.set_chunked_content_provider(caiwei::rest::content::type::EVENT, [
                completions = std::move(completions)
            ](size_t /* offset */, httplib::DataSink& sink) mutable -> bool {
                caiwei::session::CompletionsSession session(completions, [&sink](const char* data, int length) {
                    const bool writable = sink.is_writable();
                    if (writable) {
                        if (data == nullptr) {
                            return writable;
                        } else {
                            sink.write("data: ", 6);
                            if (length < 0) {
                                length = std::strlen(data);
                            }
                            sink.write(data, length);
                            sink.write("\n\n", 2);
                        }
                    }
                    return writable;
                });
                if (!session.get()) {
                    CW_LOG_W("会话生成失败");
                }
                if (sink.is_writable()) {
                    sink.write("data: [DONE]\n\n", 14);
                }
                sink.done();
                return false;
            });
        } else {
            caiwei::session::CompletionsSession session(completions);
            response.set_content(session.get_sync(), caiwei::rest::content::type::JSON);
        }
    });
}
