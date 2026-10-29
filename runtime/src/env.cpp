#include <map>
#include <mutex>
#include <cstring>
#include <unordered_map>

#include "caiwei/env.hpp"
#include "caiwei/log.hpp"

static std::mutex mutex;
static std::unordered_map<std::string, std::string> env_map;

std::atomic_uint32_t caiwei::env::id_index = 0;

static std::map<std::string, std::string> default_config = {
    {"CAIWEI_VERSION",     "1.0.0"  }, // 版本号
    {"CAIWEI_CUDA_ID",     "0"      }, // CUDA ID
    {"CAIWEI_RKNN_ID",     "0"      }, // RKNN ID
    {"CAIWEI_SERVER_HOST", "0.0.0.0"}, // 监听主机
    {"CAIWEI_SERVER_PORT", "8888"   }, // 监听端口
    {"CAIWEI_SERVER_POOL", "4"      }, // 线程数量
    {"CAIWEI_USERNAME",    "admin"  }, // 接口账号
    {"CAIWEI_PASSWORD",    "admin"  }, // 接口密码
    {"CAIWEI_SECURITY",    "OFF"    }, // 安全验证
    {"CAIWEI_TIMEOUT",     "30000"  }, // 超时时间
    // 运行配置
    {"CAIWEI_RUNTIME_TIMEOUT",      "30" },
    {"CAIWEI_RUNTIME_KEEPALIVE",    "600"},
    // 共享配置
    {"CAIWEI_RKNN2_SHARE",          "false"},
    {"CAIWEI_RKNN2_MIN_POOL",       "2"    },
    {"CAIWEI_RKNN2_MAX_POOL",       "8"    },
    {"CAIWEI_RKNN3_SHARE",          "true" },
    {"CAIWEI_RKNN3_MIN_POOL",       "0"    },
    {"CAIWEI_RKNN3_MAX_POOL",       "1"    },
    {"CAIWEI_LLAMACPP_SHARE",       "true" },
    {"CAIWEI_LLAMACPP_MIN_POOL",    "0"    },
    {"CAIWEI_LLAMACPP_MAX_POOL",    "1"    },
    {"CAIWEI_ONNXRUNTIME_SHARE",    "false"},
    {"CAIWEI_ONNXRUNTIME_MIN_POOL", "2"    },
    {"CAIWEI_ONNXRUNTIME_MAX_POOL", "8"    },
    // CLS
    {"CAIWEI_CLS_C",                    "3"   },
    {"CAIWEI_CLS_H",                    "224" },
    {"CAIWEI_CLS_W",                    "224" },
    {"CAIWEI_CLS_CLASS_SIZE",           "1000"},
    {"CAIWEI_CLS_TOP_K",                "5"   },
    {"CAIWEI_CLS_CONFIDENCE_THRESHOLD", "0.4" },
    // DET
    {"CAIWEI_DET_C",                    "3"  },
    {"CAIWEI_DET_H",                    "640"},
    {"CAIWEI_DET_W",                    "640"},
    {"CAIWEI_DET_CLASS_SIZE",           "80" },
    {"CAIWEI_DET_IOU_THRESHOLD",        "0.6"},
    {"CAIWEI_DET_CONFIDENCE_THRESHOLD", "0.4"},
    // ASR
    {"CAIWEI_ASR_MAX_TOKEN_LENGTH", "24576"          },
    {"CAIWEI_ASR_TOKEN_BOS",        "<|im_start|>"   },
    {"CAIWEI_ASR_TOKEN_EOS",        "<|im_end|>"     },
    {"CAIWEI_ASR_TOKEN_PAD",        "<|endoftext|>"  },
    {"CAIWEI_ASR_TOKEN_BAUDIO",     "<|audio_start|>"},
    {"CAIWEI_ASR_TOKEN_CAUDIO",     "<|audio_pad|>"  },
    {"CAIWEI_ASR_TOKEN_EAUDIO",     "<|audio_end|>"  },
    {"CAIWEI_ASR_TOKEN_BTHINKING",  "<think>"        },
    {"CAIWEI_ASR_TOKEN_ETHINKING",  "</think>"       },
    {"CAIWEI_ASR_TOKEN_BTOOLCALL",  "<tool_call>"    },
    {"CAIWEI_ASR_TOKEN_ETOOLCALL",  "</tool_call>"   },
    {"CAIWEI_ASR_MEDIA_MARKER",     "<__media__>"    },
    {"CAIWEI_ASR_AUDIO_MARKER",     "<audio>"        },
    {"CAIWEI_ASR_ENABLE_THINKING",  "enable_thinking"},
    // LLM
    {"CAIWEI_LLM_MAX_TOKEN_LENGTH", "8092"           },
    {"CAIWEI_LLM_TOKEN_BOS",        "<|im_start|>"   },
    {"CAIWEI_LLM_TOKEN_EOS",        "<|im_end|>"     },
    {"CAIWEI_LLM_TOKEN_PAD",        "<|endoftext|>"  },
    {"CAIWEI_LLM_TOKEN_BTHINKING",  "<think>"        },
    {"CAIWEI_LLM_TOKEN_ETHINKING",  "</think>"       },
    {"CAIWEI_LLM_TOKEN_BTOOLCALL",  "<tool_call>"    },
    {"CAIWEI_LLM_TOKEN_ETOOLCALL",  "</tool_call>"   },
    {"CAIWEI_LLM_ENABLE_THINKING",  "enable_thinking"},
    // VLM
    {"CAIWEI_VLM_MAX_TOKEN_LENGTH", "8092"            },
    {"CAIWEI_VLM_TOKEN_BOS",        "<|im_start|>"    },
    {"CAIWEI_VLM_TOKEN_EOS",        "<|im_end|>"      },
    {"CAIWEI_VLM_TOKEN_PAD",        "<|endoftext|>"   },
    {"CAIWEI_VLM_TOKEN_BIMAGE",     "<|vision_start|>"},
    {"CAIWEI_VLM_TOKEN_CIMAGE",     "<|image_pad|>"   },
    {"CAIWEI_VLM_TOKEN_EIMAGE",     "<|vision_end|>"  },
    {"CAIWEI_VLM_TOKEN_BVIDEO",     "<|vision_start|>"},
    {"CAIWEI_VLM_TOKEN_CVIDEO",     "<|video_pad|>"   },
    {"CAIWEI_VLM_TOKEN_EVIDEO",     "<|vision_end|>"  },
    {"CAIWEI_VLM_TOKEN_BTHINKING",  "<think>"         },
    {"CAIWEI_VLM_TOKEN_ETHINKING",  "</think>"        },
    {"CAIWEI_VLM_TOKEN_BTOOLCALL",  "<tool_call>"     },
    {"CAIWEI_VLM_TOKEN_ETOOLCALL",  "</tool_call>"    },
    {"CAIWEI_VLM_MEDIA_MARKER",     "<__media__>"     },
    {"CAIWEI_VLM_IMAGE_MARKER",     "<image>"         },
    {"CAIWEI_VLM_VIDEO_MARKER",     "<video>"         },
    {"CAIWEI_VLM_ENABLE_THINKING",  "enable_thinking" },
    {"CAIWEI_VLM_WIDTH",            "384"             },
    {"CAIWEI_VLM_HEIGHT",           "384"             },
    // EMBEDDING
    {"CAIWEI_EMBEDDING_MAX_TOKEN_LENGTH", "8092"},
    // RERANKING
    {"CAIWEI_RERANKING_MAX_TOKEN_LENGTH", "8092"},
    {"CAIWEI_RERANKING_TOKEN_BOS",          "<|im_start|>"},
    {"CAIWEI_RERANKING_TOKEN_EOS",          "<|im_end|>"  },
    {"CAIWEI_RERANKING_TOKEN_BTHINKING",    "<think>"     },
    {"CAIWEI_RERANKING_TOKEN_ETHINKING",    "</think>"    },
    {"CAIWEI_RERANKING_INSTRUCTION_KEY",    "<Instruct>: "},
    {"CAIWEI_RERANKING_QUERY_KEY",          "<Query>: "   },
    {"CAIWEI_RERANKING_DOCUMENT_KEY",       "<Document>: "},
    {"CAIWEI_RERANKING_SYSTEM_PROMPT",      R"(Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".)"},
    {"CAIWEI_RERANKING_INSTRUCTION_PROMPT", R"(Given a web search query, retrieve relevant passages that answer the query)"},
    // 模型配置: 类型,厂商,名称,路径
    // RKNN3   : LLM模型文件|LLM权重文件|embedding文件|tokenizer文件
    // RKNN3   : LLM模型文件|LLM权重文件|embedding文件|tokenizer文件|ASR模型文件|ASR权重文件
    // RKNN3   : LLM模型文件|LLM权重文件|embedding文件|tokenizer文件|VLM模型文件|VLM权重文件
    // LLAMACPP: LLM模型文件
    // LLAMACPP: LLM模型文件|ASR模型文件
    // LLAMACPP: LLM模型文件|VLM模型文件
    {"CAIWEI_CONTEXT_INFO", R"(
CLS,YOLO,yolo26n-cls,yolo26n-cls.onnx
DET,YOLO,yolo26n-det,yolo26n-det.onnx
ASR,QWEN,qwen3-asr,Qwen3-ASR/Qwen3-ASR-0.6B-Q4_K_M.gguf|Qwen3-ASR/mmproj-Qwen3-ASR-0.6b-Q4_K_M.gguf
LLM,QWEN,qwen3-llm,Qwen3/Qwen3-1.7B-Q4_K_M.gguf
VLM,QWEN,qwen3-vlm,Qwen3-VL/Qwen3-VL-2B-Instruct-Q4_K_M.gguf|Qwen3-VL/mmproj-Qwen3-VL-2b-Instruct-Q4_K_M.gguf
EMBEDDING,QWEN,qwen3-embedding,Qwen3-Embedding/Qwen3-Embedding-0.6B-Q4_K_M.gguf
RERANKING,QWEN,qwen3-reranking,Qwen3-Reranker/Qwen3-Reranker-0.6B-Q4_K_M.gguf
    )"},
};

std::string caiwei::env::get(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex);
    if (auto iter = env_map.find(name); iter != env_map.end()) {
        return iter->second;
    }
    const char* value = std::getenv(name.c_str());
    if (value == nullptr || std::strlen(value) == 0) {
        auto iterator = default_config.find(name);
        if (iterator == default_config.end()) {
            CW_LOG_W("不支持的环境配置: %s", name.c_str());
            value = "";
        } else {
            value = iterator->second.c_str();
        }
    }
    env_map.emplace(name, value);
    return value;
}

int caiwei::env::get_int(const std::string& name) {
    return std::stoi(caiwei::env::get(name));
}

int64_t caiwei::env::get_long(const std::string& name) {
    return std::stoll(caiwei::env::get(name));
}

bool caiwei::env::get_bool(const std::string& name) {
    return caiwei::env::get(name) == "ON" || caiwei::env::get(name) == "true";
}

float caiwei::env::get_float(const std::string& name) {
    return std::stof(caiwei::env::get(name));
}

std::string caiwei::env::get_string(const std::string& name) {
    return caiwei::env::get(name);
}

void caiwei::env::set(const std::string& name, const std::string& value) {
    CW_LOG_I("设置环境配置: %s = %s", name.c_str(), value.c_str());
    std::lock_guard<std::mutex> lock(mutex);
    env_map.insert_or_assign(name, value);
}

void caiwei::env::print_all_env() {
    for (auto& pair : default_config) {
        CW_LOG_I("%-48s = %s", pair.first.c_str(), caiwei::env::get(pair.first).c_str());
    }
}
