#include "caiwei/env.hpp"

#include <map>

std::atomic_int caiwei::env::id_index = caiwei::env::min_id_index;

// 默认配置
static std::map<std::string, std::string> default_config = {
    {"CAIWEI_DEVICE_ID",   "0"    }, // 设备ID
    {"CAIWEI_SERVER_PORT", "8888" }, // 监听端口
    {"CAIWEI_USERNAME",    "admin"}, // 接口账号
    {"CAIWEI_PASSWORD",    "admin"}, // 接口密码
    {"CAIWEI_SECURITY",    "OFF"  }, // 安全验证
    {"CAIWEI_TIMEOUT",     "30000"}, // 超时时间
    {"CAIWEI_YOLO_DET_IOU_THRESHOLD",        "0.6"}, // 物体检测IOU阈值
    {"CAIWEI_YOLO_DET_CONFIDENCE_THRESHOLD", "0.4"}, // 物体检测置信度阈值
};

std::string caiwei::env::get(const std::string& name) {
    const char* value = std::getenv(name.c_str());
    if(!value || std::strlen(value) == 0) {
        auto iterator = default_config.find(name);
        if (iterator == default_config.end()) {
            SPDLOG_WARN("不支持的配置：{}", name);
            value = "";
        } else {
            value = iterator->second.c_str();
        }
    }
    return value;
}

void caiwei::env::set(const std::string& name, const std::string& value) {
    SPDLOG_INFO("设置配置：{} - {}", name, value);
    #if _WIN32
    _putenv_s(name.c_str(), value.c_str());
    #elif defined(__linux) || defined(__linux__)
    setenv(name.c_str(), value.c_str(), true);
    #else
    SPDLOG_WARN("设置失败：{} - {}", name, value);
    #endif
}
