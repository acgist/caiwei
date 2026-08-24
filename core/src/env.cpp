#include "caiwei/env.hpp"
#include "caiwei/log.hpp"

#include <map>

std::atomic_int caiwei::env::id_index = caiwei::env::min_id_index;

// 默认配置
static std::map<std::string, std::string> default_config = {
    {"CAIWEI_VERSION",     "1.0.0"}, // 版本号
    {"CAIWEI_CUDA_ID",     "0"    }, // CUDA ID
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
    if (!value || std::strlen(value) == 0) {
        auto iterator = default_config.find(name);
        if (iterator == default_config.end()) {
            LOG_WARN("不支持的环境配置: %s", name.c_str());
            value = "";
        } else {
            value = iterator->second.c_str();
        }
    }
    return value;
}

int caiwei::env::get_int(const std::string& name) {
    return std::stoi(caiwei::env::get(name));
}

bool caiwei::env::get_bool(const std::string& name) {
    return caiwei::env::get(name) == "ON";
}

float caiwei::env::get_float(const std::string& name) {
    return std::stof(caiwei::env::get(name));
}

std::string caiwei::env::get_string(const std::string& name) {
    return caiwei::env::get(name);
}

void caiwei::env::set(const std::string& name, const std::string& value) {
    LOG_INFO("设置环境配置: %s = %s", name.c_str(), value.c_str());
#if OS_WIN
    _putenv_s(name.c_str(), value.c_str());
#elif OS_LINUX
    setenv(name.c_str(), value.c_str(), true);
#else
    LOG_WARN("设置环境失败: %s = %s", name.c_str(), value.c_str());
#endif
}
