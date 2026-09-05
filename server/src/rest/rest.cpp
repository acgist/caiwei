#include "caiwei/log.hpp"
#include "caiwei/env.hpp"
#include "caiwei/json.hpp"
#include "caiwei/rest.hpp"
#include "caiwei/context.hpp"

#include "httplib.h"
#include "base64/base64.h"

#include <fstream>
#include <filesystem>

static void restHandler();
static void restGetIndex();
static void restGetHealth();
static void restGetModels();
static void restGetFavicon();
static void restGetShutdown();

httplib::Server* caiwei::rest::server = new httplib::Server;

void caiwei::rest::open() {
    restHandler();
    rest_api();
    rest_text_api();
    rest_audio_api();
    rest_image_api();
    rest_video_api();
    auto host = caiwei::env::get("CAIWEI_SERVER_HOST");
    auto port = caiwei::env::get_int("CAIWEI_SERVER_PORT");
    if (port <= 0) {
        CW_LOG_W("端口配置错误: %d", port);
        return;
    }
    CW_LOG_I("启动REST服务: %s - %d", host.c_str(), port);
    caiwei::rest::server->new_task_queue = [] {
        return new httplib::ThreadPool(4);
    };
    // TODO 测试
    caiwei::rest::server->set_read_timeout(60);
    caiwei::rest::server->set_write_timeout(60);
    caiwei::rest::server->set_keep_alive_timeout(60);
    caiwei::rest::server->set_logger([](const httplib::Request& request, const httplib::Response&) {
        CW_LOG_I("请求日志: %s - %s - %s", request.method.c_str(), request.path.c_str(), request.remote_addr.c_str());
    });
    caiwei::rest::server->listen(host, port);
    delete caiwei::rest::server;
    caiwei::rest::server = nullptr;
    CW_LOG_I("结束REST服务: %s - %d", host.c_str(), port);
}

void caiwei::rest::stop() {
    CW_LOG_I("关闭REST服务");
    caiwei::rest::server->stop();
}

void caiwei::rest::rest_api() {
    restGetIndex();
    restGetHealth();
    restGetModels();
    restGetFavicon();
    restGetShutdown();
}

static void restHandler() {
    caiwei::rest::server->set_pre_routing_handler([](const httplib::Request& request, httplib::Response& response) {
        if(caiwei::env::get_bool("CAIWEI_SECURITY")) {
            auto authorization = request.get_header_value("Authorization");
            static std::string username = caiwei::env::get("CAIWEI_USERNAME");
            static std::string password = caiwei::env::get("CAIWEI_PASSWORD");
            if(
                authorization.length()  <= 6 ||
                authorization.substr(6) != base64_encode(username + ":" + password)
            ) {
                response.status = httplib::StatusCode::Unauthorized_401;
                response.set_content(caiwei::json::buildResponse("1401", "没有授权"), caiwei::rest::content::type::JSON);
                return httplib::Server::HandlerResponse::Handled;
            }
        } else {
            response.set_header("Access-Control-Allow-Origin",  "*");
            response.set_header("Access-Control-Allow-Methods", "*");
            response.set_header("Access-Control-Allow-Headers", "*");
//          response.set_header("Access-Control-Allow-Credentials", "true");
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    caiwei::rest::server->set_error_handler([](const httplib::Request& request, httplib::Response& response) {
        CW_LOG_W("系统错误: %s - %s - %d - %s", request.path.c_str(), request.body.c_str(), response.status, response.body.c_str());
        if(response.status == httplib::StatusCode::OK_200) {
            response.status = httplib::StatusCode::InternalServerError_500;
        }
        if(response.body.empty()) {
            response.set_content(caiwei::json::buildResponse(std::to_string(1000 + response.status).c_str(), "网关错误"), caiwei::rest::content::type::JSON);
        }
    });
    caiwei::rest::server->set_exception_handler([](const httplib::Request& request, httplib::Response& response, std::exception_ptr e) {
        CW_LOG_W("系统异常: %s - %s - %d - %s", request.path.c_str(), request.body.c_str(), response.status, response.body.c_str());
        std::string message;
        try {
            std::rethrow_exception(std::move(e));
        } catch (std::exception& e) {
            message = caiwei::json::buildResponse("9999", e.what());
        } catch (...) {
            message = caiwei::json::buildResponse("9999", "未知错误");
        }
        response.status = httplib::StatusCode::InternalServerError_500;
        response.set_content(message, caiwei::rest::content::type::JSON);
    });
}

static void restGetIndex() {
    caiwei::rest::server->Get("/", [](const httplib::Request&, httplib::Response& response) {
        response.set_content(R"(<!DOCTYPE html>
<html lang="zh-cn">

<head>
  <meta charset="UTF-8">
  <title>采薇</title>
  <style type="text/css">
    p  {text-align:center;}
    pre{text-align:center;}
  </style>
</head>

<body>
  <p>
    <pre>
    小雅·采薇

    采薇采薇，薇亦作止。曰归曰归，岁亦莫止。靡室靡家，猃狁之故。不遑启居，猃狁之故。
    采薇采薇，薇亦柔止。曰归曰归，心亦忧止。忧心烈烈，载饥载渴。我戍未定，靡使归聘。
    采薇采薇，薇亦刚止。曰归曰归，岁亦阳止。王事靡盬，不遑启处。忧心孔疚，我行不来！
    彼尔维何？维常之华。彼路斯何？君子之车。戎车既驾，四牡业业。岂敢定居？一月三捷。
    驾彼四牡，四牡骙骙。君子所依，小人所腓。四牡翼翼，象弭鱼服。岂不日戒？猃狁孔棘！
    昔我往矣，杨柳依依。今我来思，雨雪霏霏。行道迟迟，载渴载饥。我心伤悲，莫知我哀！
    </pre>
  </p>
  <p><a href="https://gitee.com/acgist/caiwei">https://gitee.com/acgist/caiwei</a></p>
  <p><a href="https://github.com/acgist/caiwei">https://github.com/acgist/caiwei</a></p>
  <p>运行环境: CANN/RKNN/llama.cpp/ONNXRuntime</p>
  <p>支持模型: YOLO/Qwen3/Qwen3-ASR/Qwen3-TTS/Qwen3-VL/Qwen3-Omni/Qwen3-Reranker/Qwen3-VL-Reranker/Qwen3-Embedding/Qwen3-VL-Embedding</p>
</body>

</html>)", caiwei::rest::content::type::HTML);
    });
}

static void restGetHealth() {
    caiwei::rest::server->Get("/v1/health", [](const httplib::Request&, httplib::Response& response) {
        response.set_content(caiwei::json::buildResponse("running"), caiwei::rest::content::type::JSON);
    });
}

static void restGetModels() {
    caiwei::rest::server->Get("/v1/models", [](const httplib::Request&, httplib::Response& response) {
        const auto& list = caiwei::context::context_info_list;
        nlohmann::json ret;
        for (const auto& value : list) {
            ret.push_back({
                {"name", value.name},
                {"path", value.path}
            });
        }
        response.set_content(caiwei::json::buildResponse(ret), caiwei::rest::content::type::JSON);
    });
}

static void restGetFavicon() {
    caiwei::rest::server->Get("/favicon.ico", [](const httplib::Request&, httplib::Response& response) {
        std::filesystem::path path("./favicon.ico");
        auto file_size = std::filesystem::file_size(path);
        if(file_size <= 0) {
            response.status = httplib::StatusCode::NotFound_404;
            response.set_content(caiwei::json::buildResponse("1404", "文件无效"), caiwei::rest::content::type::JSON);
            return;
        }
        std::ifstream stream;
        stream.open(path, std::ios_base::binary);
        if (stream.is_open()) {
            std::vector<char> data(file_size);
            stream.read(data.data(), file_size);
            response.set_content(data.data(), file_size, caiwei::rest::content::type::ICON);
        }
        stream.close();
    });
}

static void restGetShutdown() {
    caiwei::rest::server->Get("/shutdown", [](const httplib::Request&, httplib::Response& response) {
        response.set_content(caiwei::json::buildResponse("正在关机..."), caiwei::rest::content::type::JSON);
        caiwei::rest::stop();
    });
}
