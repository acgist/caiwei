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
    // CLS
    {"CAIWEI_CLS_W",                    "224"              }, // 宽度
    {"CAIWEI_CLS_H",                    "224"              }, // 高度
    {"CAIWEI_CLS_TOP_K",                "5"                }, // TOP-K
    {"CAIWEI_CLS_CLASS_SIZE",           "1000"             }, // 类型
    {"CAIWEI_CLS_CONFIDENCE_THRESHOLD", "0.4"              }, // 置信度阈值
    // DET
    {"CAIWEI_DET_W",                    "640"              }, // 宽度
    {"CAIWEI_DET_H",                    "640"              }, // 高度
    {"CAIWEI_DET_CLASS_SIZE",           "80"               }, // 类型
    {"CAIWEI_DET_IOU_THRESHOLD",        "0.6"              }, // IOU阈值
    {"CAIWEI_DET_CONFIDENCE_THRESHOLD", "0.4"              }, // 置信度阈值
    // OBB
    // SEG
    // SEM
    // POSE
    // DEPTH
    // ASR
    // TTS
    // LLM
    {"CAIWEI_LLM_MAX_TOKEN_LENGTH",     "8092"             }, // 最大上下文长度
    {"CAIWEI_LLM_TOKEN_BOS",            "<|im_start|>"     },
    {"CAIWEI_LLM_TOKEN_EOS",            "<|im_end|>"       },
    {"CAIWEI_LLM_TOKEN_PAD",            "<|endoftext|>"    },
    {"CAIWEI_LLM_TOKEN_BTHINK",         "<think>"          },
    {"CAIWEI_LLM_TOKEN_ETHINK",         "</think>"         },
    {"CAIWEI_LLM_TOKEN_BTOOLCALL",      "<tool_call>"      },
    {"CAIWEI_LLM_TOKEN_ETOOLCALL",      "</tool_call>"     },
    {"CAIWEI_LLM_ENABLE_THINKING",      "enable_thinking"  },
    // VLM
    {"CAIWEI_VLM_MAX_TOKEN_LENGTH",     "8092"             }, // 最大上下文长度
    {"CAIWEI_VLM_TOKEN_BOS",            "<|im_start|>"     },
    {"CAIWEI_VLM_TOKEN_EOS",            "<|im_end|>"       },
    {"CAIWEI_VLM_TOKEN_PAD",            "<|endoftext|>"    },
    {"CAIWEI_VLM_TOKEN_BIMAGE",         "<|vision_start|>" },
    {"CAIWEI_VLM_TOKEN_CIMAGE",         "<|image_pad|>"    },
    {"CAIWEI_VLM_TOKEN_EIMAGE",         "<|vision_end|>"   },
    {"CAIWEI_VLM_TOKEN_BVIDEO",         "<|vision_start|>" },
    {"CAIWEI_VLM_TOKEN_CVIDEO",         "<|video_pad|>"    },
    {"CAIWEI_VLM_TOKEN_EVIDEO",         "<|vision_end|>"   },
    {"CAIWEI_VLM_TOKEN_BTHINK",         "<think>"          },
    {"CAIWEI_VLM_TOKEN_ETHINK",         "</think>"         },
    {"CAIWEI_VLM_TOKEN_BTOOLCALL",      "<tool_call>"      },
    {"CAIWEI_VLM_TOKEN_ETOOLCALL",      "</tool_call>"     },
    {"CAIWEI_VLM_ENABLE_THINKING",      "enable_thinking"  },
    // MLLM
    {"CAIWEI_MLLM_MAX_TOKEN_LENGTH",     "8092"             }, // 最大上下文长度
    {"CAIWEI_MLLM_TOKEN_BOS",            "<|im_start|>"     },
    {"CAIWEI_MLLM_TOKEN_EOS",            "<|im_end|>"       },
    {"CAIWEI_MLLM_TOKEN_PAD",            "<|endoftext|>"    },
    {"CAIWEI_MLLM_TOKEN_BAUDIO",         "<|audio_start|>"  },
    {"CAIWEI_MLLM_TOKEN_CAUDIO",         "<|audio_pad|>"    },
    {"CAIWEI_MLLM_TOKEN_EAUDIO",         "<|audio_end|>"    },
    {"CAIWEI_MLLM_TOKEN_BIMAGE",         "<|vision_start|>" },
    {"CAIWEI_MLLM_TOKEN_CIMAGE",         "<|image_pad|>"    },
    {"CAIWEI_MLLM_TOKEN_EIMAGE",         "<|vision_end|>"   },
    {"CAIWEI_MLLM_TOKEN_BVIDEO",         "<|vision_start|>" },
    {"CAIWEI_MLLM_TOKEN_CVIDEO",         "<|video_pad|>"    },
    {"CAIWEI_MLLM_TOKEN_EVIDEO",         "<|vision_end|>"   },
    {"CAIWEI_MLLM_TOKEN_BTHINK",         "<think>"          },
    {"CAIWEI_MLLM_TOKEN_ETHINK",         "</think>"         },
    {"CAIWEI_MLLM_TOKEN_BTOOLCALL",      "<tool_call>"      },
    {"CAIWEI_MLLM_TOKEN_ETOOLCALL",      "</tool_call>"     },
    {"CAIWEI_MLLM_ENABLE_THINKING",      "enable_thinking"  },
    // EMBEDDING
    // RERANKING
    // VL_EMBEDDING
    // VL_RERANKING
    // 模型配置: 类型,厂商,名称,路径
    {"CAIWEI_CONTEXT_INFO", R"(
CLS,YOLO,yolo26n-cls,yolo26n-cls.onnx
DET,YOLO,yolo26n-det,yolo26n-det.onnx
LLM,QWEN,qwen3-llm,Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf
    )"},
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
