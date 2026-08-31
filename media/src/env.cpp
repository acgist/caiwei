#include "caiwei/env.hpp"
#include "caiwei/log.hpp"

#include <map>
#include <cstring>

std::atomic_uint32_t caiwei::env::id_index = 0;

// 默认配置
static std::map<std::string, std::string> default_config = {
    {"CAIWEI_VERSION",     "1.0.0"   }, // 版本号
    {"CAIWEI_CUDA_ID",     "0"       }, // CUDA ID
    {"CAIWEI_SERVER_HOST", "0.0.0.0" }, // 监听主机
    {"CAIWEI_SERVER_PORT", "8888"    }, // 监听端口
    {"CAIWEI_USERNAME",    "admin"   }, // 接口账号
    {"CAIWEI_PASSWORD",    "admin"   }, // 接口密码
    {"CAIWEI_SECURITY",    "OFF"     }, // 安全验证
    {"CAIWEI_TIMEOUT",     "30000"   }, // 超时时间
    // DET
    {"CAIWEI_DET_W",                    "640"          }, // 超时时间
    {"CAIWEI_DET_H",                    "640"          }, // 超时时间
    {"CAIWEI_DET_TYPE",                 "YOLO_DET"     }, // 超时时间
    {"CAIWEI_DET_NAME",                 "yolo26n"      }, // 超时时间
    {"CAIWEI_DET_PATH",                 "yolo26n.onnx" }, // 超时时间
    {"CAIWEI_DET_CLASS_SIZE",           "80"           }, // 超时时间
    {"CAIWEI_DET_IOU_THRESHOLD",        "0.6"          }, // 物体检测IOU阈值
    {"CAIWEI_DET_CONFIDENCE_THRESHOLD", "0.4"          }, // 物体检测置信度阈值
    
};

std::string caiwei::env::get(const std::string& name) {
    const char* value = std::getenv(name.c_str());
    if (!value || std::strlen(value) == 0) {
        auto iterator = default_config.find(name);
        if (iterator == default_config.end()) {
            CW_LOG_W("不支持的环境配置: %s", name.c_str());
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
    CW_LOG_I("设置环境配置: %s = %s", name.c_str(), value.c_str());
#if OS_WIN
    _putenv_s(name.c_str(), value.c_str());
#elif OS_LINUX
    setenv(name.c_str(), value.c_str(), true);
#else
    CW_LOG_W("设置环境失败: %s = %s", name.c_str(), value.c_str());
#endif
}

void caiwei::env::print_all_env() {
    for (auto& pair : default_config) {
        CW_LOG_I("%-48s = %s", pair.first.c_str(), caiwei::env::get(pair.first).c_str());
    }
}
