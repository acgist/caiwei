#include "caiwei/env.hpp"
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

static bool sse(httplib::DataSink& sink, const char* data, int length, bool success) {
    const bool writable = sink.is_writable();
    if (writable && data != nullptr) {
        if (length <= 0) {
            length = std::strlen(data);
        }
        if (!success) {
            if (!sink.write("event: error\n", 13)) {
                return false;
            }
        }
        // clang-format off
        if (
            !sink.write("data: ", 6)  ||
            !sink.write(data, length) ||
            !sink.write("\n\n", 2)
        ) {
            // clang-format on
            return false;
        }
    }
    return writable;
}

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
        const auto list = caiwei::context::get_all_context_info();
        for (const auto& value : list) {
            // clang-format off
            body.push_back({
                {"type", get_type_name(value->type)},
                {"name", value->name               },
            });
            // clang-format on
        }
        response.set_content(caiwei::json::buildResponse(std::move(body)), caiwei::rest::content::type::JSON);
    });
}

static void get_video_play() {
    caiwei::rest::server->Get("/v1/video/play", [](const httplib::Request& request, httplib::Response& response) {
        // clang-format off
        auto url   = request.get_param_value("url");
        auto type  = request.get_param_value("type");
        auto codec = request.get_param_value("codec");
        // clang-format on
        if (!caiwei::text::check_security(url, type)) {
            response.status = httplib::StatusCode::BadRequest_400;
            return;
        }
        // clang-format off
        // 如果线程紧张可以修改阻塞模式: 使用队列异步回调
        response.set_chunked_content_provider(caiwei::rest::content::type::EVENT, [
            url   = std::move(url),
            type  = std::move(type),
            codec = std::move(codec)
        ](size_t /* offset */, httplib::DataSink& sink) mutable -> bool {
                // clang-format on
                try {
                    caiwei::session::Callback callback = std::bind(
                        sse, std::ref(sink), std::placeholders::_1, std::placeholders::_2, true
                    );
                    caiwei::session::VideoPlaySession session(std::move(url), std::move(type), codec == "true", callback);
                    if (session.get()) {
                        // -
                    } else {
                        sse(sink, caiwei::json::buildResponse("2400", "媒体播放失败").c_str(), -1, false);
                    }
                } catch (const caiwei::env::MessageCodeException& e) {
                    sse(sink, caiwei::json::buildResponse(e.code, e.message).c_str(), -1, false);
                } catch (...) {
                    sse(sink, caiwei::json::buildResponse("9999", "系统异常").c_str(), -1, false);
                }
                sink.done();
                return false;
            }
        );
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
            return;
        }
        if (completions.stream) {
            // 如果线程紧张可以修改阻塞模式: 使用队列异步回调
            response.set_chunked_content_provider(
                caiwei::rest::content::type::EVENT,
                [completions = std::move(completions)](size_t /* offset */, httplib::DataSink& sink) mutable -> bool {
                    try {
                        caiwei::session::Callback callback = std::bind(
                            sse, std::ref(sink), std::placeholders::_1, std::placeholders::_2, true
                        );
                        caiwei::session::CompletionsSession session(completions, callback);
                        if (session.get()) {
                            sse(sink, "[DONE]", -1, true);
                        } else {
                            sse(sink, caiwei::json::buildResponse("3400", "会话生成失败").c_str(), -1, false);
                        }
                    } catch (const caiwei::env::MessageCodeException& e) {
                        sse(sink, caiwei::json::buildResponse(e.code, e.message).c_str(), -1, false);
                    } catch (...) {
                        sse(sink, caiwei::json::buildResponse("9999", "系统异常").c_str(), -1, false);
                    }
                    sink.done();
                    return false;
                }
            );
        } else {
            caiwei::session::CompletionsSession session(completions);
            response.set_content(session.get_sync(), caiwei::rest::content::type::JSON);
        }
    });
}
