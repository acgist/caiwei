#include "caiwei/log.hpp"
#include "caiwei/json.hpp"
#include "caiwei/rest.hpp"
#include "caiwei/context.hpp"
#include "caiwei/session.hpp"
#include "caiwei/text_data.hpp"

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
    caiwei::rest::server->Get("/v1/health", [](const httplib::Request&, httplib::Response& response) {
        response.set_content(caiwei::json::buildResponse("running"), caiwei::rest::content::type::JSON);
    });
}

static void get_models() {
    caiwei::rest::server->Get("/v1/models", [](const httplib::Request&, httplib::Response& response) {
        const auto& list = caiwei::context::context_info_list;
        nlohmann::json ret;
        for (const auto& value : list) {
            ret.push_back({
                {"type", value.type_},
                {"name", value.name },
                {"path", value.path }
            });
        }
        response.set_content(caiwei::json::buildResponse(ret), caiwei::rest::content::type::JSON);
    });
}

static void get_video_play() {
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
        if (completions.stream) {
            response.set_chunked_content_provider(caiwei::rest::content::type::EVENT, [&](size_t /*offset*/, httplib::DataSink& sink) {
                caiwei::session::CompletionsSession session(completions, [&sink](const char* type, const char* data, size_t length) {
                    if (sink.is_writable()) {
                        sink.write("data: ", 6);
                        if(length > 0) {
                            sink.write(data, length);
                        }
                        sink.write("\n\n", 2);
                    }
                    return sink.is_writable();
                });
                session.get().get();
                if (sink.is_writable()) {
                    sink.write("data: [DONE]", 12);
                }
                sink.done();
                return true;
            });
        } else {
            caiwei::session::CompletionsSession session(completions);
            response.set_content(session.get_sync(), caiwei::rest::content::type::JSON);
        }
    });
}
