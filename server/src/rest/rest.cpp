#include "caiwei/log.hpp"
#include "caiwei/env.hpp"
#include "caiwei/json.hpp"
#include "caiwei/rest.hpp"

#include <mutex>
#include <fstream>
#include <filesystem>

#include "httplib.h"

static void get_index();
static void get_favicon();
static void get_shutdown();

static void release_server();
static void register_handler();
static bool constant_time_password_check(std::string_view a, std::string_view z);

static std::mutex server_mutex;

httplib::Server* caiwei::rest::server = nullptr;

bool caiwei::rest::open() {
    {
        std::lock_guard<std::mutex> lock(server_mutex);
        if (caiwei::rest::server == nullptr) {
            caiwei::rest::server = new httplib::Server;
        } else {
            caiwei::rest::server->wait_until_ready();
            return caiwei::rest::server->is_running();
        }
    }
    const auto host     = caiwei::env::get("CAIWEI_SERVER_HOST");
    const auto port     = caiwei::env::get_int("CAIWEI_SERVER_PORT");
    const auto min_pool = caiwei::env::get_int("CAIWEI_SERVER_MIN_POOL");
    const auto max_pool = caiwei::env::get_int("CAIWEI_SERVER_MAX_POOL");
    if (port <= 0) {
        CW_LOG_W("REST服务端口配置错误: %d", port);
        release_server();
        return false;
    }
    if (min_pool <= 0 || max_pool <= 0 || min_pool > max_pool) {
        CW_LOG_W("REST服务线程配置错误: %d - %d", min_pool, max_pool);
        release_server();
        return false;
    }
    caiwei::rest::server->set_read_timeout(60);
    caiwei::rest::server->set_write_timeout(60);
    caiwei::rest::server->set_keep_alive_timeout(60);
    caiwei::rest::server->set_payload_max_length(64LL * 1024 * 1024);
    caiwei::rest::server->new_task_queue = [min_pool, max_pool] {
        return new httplib::ThreadPool(min_pool, max_pool);
    };
    caiwei::rest::server->set_logger([](const httplib::Request& request, const httplib::Response& /* response */) {
        CW_LOG_I("请求日志: %s => %s : %s", request.remote_addr.c_str(), request.method.c_str(), request.path.c_str());
    });
    register_handler();
    get_index();
    get_favicon();
    get_shutdown();
    register_api();
    CW_LOG_I("启动REST服务: %s:%d => %d ~ %d", host.c_str(), port, min_pool, max_pool);
    const bool ret = caiwei::rest::server->listen(host, port);
    if (!ret) {
        CW_LOG_W("监听REST服务失败");
    }
    release_server();
    CW_LOG_I("结束REST服务: %s:%d", host.c_str(), port);
    return ret;
}

void caiwei::rest::stop() {
    std::lock_guard<std::mutex> lock(server_mutex);
    if (caiwei::rest::server == nullptr) {
        CW_LOG_W("无效REST服务");
    } else {
        CW_LOG_I("关闭REST服务");
        caiwei::rest::server->stop();
    }
}

static void get_index() {
    caiwei::rest::server->Get("/", [](const httplib::Request& /* request */, httplib::Response& response) {
        response.set_content(R"(<!DOCTYPE html>
<html lang="zh-cn">

<head>
  <meta charset="UTF-8">
  <title>采薇</title>
  <style type="text/css">
    p, pre{text-align:center;}
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
  <p>运行环境: RKNN2/RKNN3/llama.cpp/ONNXRuntime</p>
  <p>支持YOLO模型: YOLO-CLS/YOLO-DET</p>
  <p>支持QWEN模型: Qwen3/Qwen3-ASR/Qwen3-VL/Qwen3-Embedding/Qwen3-Reranker</p>
</body>

</html>)", caiwei::rest::content::type::HTML);
    });
}

static void get_favicon() {
    caiwei::rest::server->Get("/favicon.ico", [](const httplib::Request& /* request */, httplib::Response& response) {
        const std::filesystem::path path("./favicon.ico");
        std::error_code error_code;
        const auto file_size = std::filesystem::file_size(path, error_code);
        if(error_code || file_size <= 0) {
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

static void get_shutdown() {
    caiwei::rest::server->Get("/shutdown", [](const httplib::Request& /* request */, httplib::Response& response) {
        #if CAIWEI_DEBUG
        response.set_content(caiwei::json::buildResponse("正在关机..."), caiwei::rest::content::type::JSON);
        caiwei::rest::stop();
        #else
        response.status = httplib::StatusCode::NotFound_404;
        response.set_content(caiwei::json::buildResponse("1404", "没有接口"), caiwei::rest::content::type::JSON);
        #endif
    });
}

static void release_server() {
    std::lock_guard<std::mutex> lock(server_mutex);
    if (caiwei::rest::server != nullptr) {
        delete caiwei::rest::server;
        caiwei::rest::server = nullptr;
    }
}

static void register_handler() {
    caiwei::rest::server->set_pre_routing_handler([](const httplib::Request& request, httplib::Response& response) {
        caiwei::env::set_thread_name("rest-thread");
        response.set_header("Access-Control-Allow-Origin",  "*");
        response.set_header("Access-Control-Allow-Methods", "GET, PUT, POST, PATCH, DELETE, OPTIONS");
        response.set_header("Access-Control-Allow-Headers", "Accept, Authorization, Content-Type, X-Requested-With");
        response.set_header("Access-Control-Max-Age",       "3600");
        if (request.method == "OPTIONS") {
            response.status = httplib::StatusCode::NoContent_204;
            return httplib::Server::HandlerResponse::Handled;
        }
        if(caiwei::env::get_bool("CAIWEI_SECURITY")) {
            const auto username = caiwei::env::get("CAIWEI_USERNAME");
            const auto password = caiwei::env::get("CAIWEI_PASSWORD");
            const auto authorization = request.get_header_value("Authorization");
            std::string_view authorization_view(authorization);
            if (authorization_view.starts_with("Basic ")) {
                authorization_view.remove_prefix(6);
            }
            if(!constant_time_password_check(authorization_view, httplib::detail::base64_encode(username + ":" + password))) {
                response.status = httplib::StatusCode::Unauthorized_401;
                response.set_header("WWW-Authenticate", "Basic realm=\"CAIWEI\"");
                response.set_content(caiwei::json::buildResponse("1401", "没有授权"), caiwei::rest::content::type::JSON);
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    caiwei::rest::server->set_error_handler([](const httplib::Request& request, httplib::Response& response) {
        CW_LOG_W("系统错误: %s - [%s] - %d - [%s]", request.path.c_str(), request.body.c_str(), response.status, response.body.c_str());
        if(response.status == httplib::StatusCode::OK_200) {
            response.status = httplib::StatusCode::InternalServerError_500;
        }
        if(response.body.empty()) {
            response.set_content(caiwei::json::buildResponse(std::to_string(1000 + response.status).c_str(), "网关错误"), caiwei::rest::content::type::JSON);
        }
    });
    caiwei::rest::server->set_exception_handler([](const httplib::Request& request, httplib::Response& response, std::exception_ptr e) {
        CW_LOG_W("系统异常: %s - [%s] - %d - [%s]", request.path.c_str(), request.body.c_str(), response.status, response.body.c_str());
        std::string message;
        try {
            std::rethrow_exception(std::move(e));
        } catch (const caiwei::env::MessageCodeException& e) {
            message = caiwei::json::buildResponse(e.code, e.message);
        } catch (const std::exception& e) {
            message = caiwei::json::buildResponse("9999", e.what());
        } catch (...) {
            message = caiwei::json::buildResponse("9999", "未知错误");
        }
        response.status = httplib::StatusCode::InternalServerError_500;
        response.set_content(message, caiwei::rest::content::type::JSON);
    });
}

static bool constant_time_password_check(std::string_view a, std::string_view z) {
    std::size_t ret = a.size() ^ z.size();
    const std::size_t max_size = std::max(a.size(), z.size());
    for (std::size_t i = 0; i < max_size; ++i) {
        const auto av = i < a.size() ? static_cast<unsigned char>(a[i]) : 0U;
        const auto zv = i < z.size() ? static_cast<unsigned char>(z[i]) : 0U;
        ret |= static_cast<std::size_t>(av ^ zv);
    }
    return ret == 0;
}
