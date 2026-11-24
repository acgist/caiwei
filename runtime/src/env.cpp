#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/tool.hpp"

#include <atomic>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#if CAIWEI_OS_WIN
#include <Windows.h>
#else
#include <pthread.h>
#define THREAD_NAME_MAX_LENGTH 15
#endif

// clang-format off
static std::once_flag    env_map_once_flag;
static std::shared_mutex env_mutex;
static std::unordered_map<std::string, std::string> env_map;
// clang-format on

static std::vector<std::string> allow_path;
static std::vector<std::string> allow_host;
static std::vector<std::string> allow_media_type;

std::atomic<uint32_t> caiwei::env::id_index = 0;

static std::unordered_set<std::string> security_config = {"CAIWEI_APIKEY", "CAIWEI_PASSWORD"};

static std::map<std::string, std::string> default_config = {
    // clang-format off
    // 公共配置
    {"CAIWEI_VERSION",         CAIWEI_VERSION       },
    {"CAIWEI_CUDA_ID",         "0"                  },
    {"CAIWEI_CUDA_MEM",        "1024"               },
    {"CAIWEI_RKNN_ID",         "0"                  },
    {"CAIWEI_SERVER_HOST",     "0.0.0.0"            },
    {"CAIWEI_SERVER_PORT",     "8888"               },
    {"CAIWEI_SERVER_MIN_POOL", "4"                  },
    {"CAIWEI_SERVER_MAX_POOL", "64"                 },
    {"CAIWEI_TIMEOUT",         "30000"              },
    {"CAIWEI_MEL_FILTERS",     "mel_128_filters.txt"},
    // 安全配置
    #if CAIWEI_DEBUG
    {"CAIWEI_SECURITY",         "OFF"   },
    {"CAIWEI_APIKEY",           "caiwei"},
    {"CAIWEI_USERNAME",         "admin" },
    {"CAIWEI_PASSWORD",         "admin" },
    {"CAIWEI_ALLOW_PATH",       ""      },
    {"CAIWEI_ALLOW_HOST",       ""      },
    {"CAIWEI_ALLOW_MEDIA_TYPE", ""      },
    #else
    {"CAIWEI_SECURITY",         "ON"       },
    {"CAIWEI_APIKEY",           "caiwei"   },
    {"CAIWEI_USERNAME",         "admin"    },
    {"CAIWEI_PASSWORD",         "admin"    },
    {"CAIWEI_ALLOW_PATH",       "/data/"   },
    {"CAIWEI_ALLOW_HOST",       ""         },
    {"CAIWEI_ALLOW_MEDIA_TYPE", "data,file"},
    #endif
    // 运行配置
    {"CAIWEI_RUNTIME_TIMEOUT",   "30" },
    {"CAIWEI_RUNTIME_INTERVAL",  "30" },
    {"CAIWEI_RUNTIME_KEEPALIVE", "600"},
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
    {"CAIWEI_VLM_IMAGE_WIDTH",      "384"             },
    {"CAIWEI_VLM_IMAGE_HEIGHT",     "384"             },
    {"CAIWEI_VLM_IMAGE_MIN_TOKEN",  "0"               },
    {"CAIWEI_VLM_IMAGE_MAX_TOKEN",  "256"             },
    // EMBEDDING
    {"CAIWEI_EMBEDDING_MAX_TOKEN_LENGTH", "8092"},
    // RERANKING
    {"CAIWEI_RERANKING_MAX_TOKEN_LENGTH",   "8092"        },
    {"CAIWEI_RERANKING_TOKEN_BOS",          "<|im_start|>"},
    {"CAIWEI_RERANKING_TOKEN_EOS",          "<|im_end|>"  },
    {"CAIWEI_RERANKING_TOKEN_BTHINKING",    "<think>"     },
    {"CAIWEI_RERANKING_TOKEN_ETHINKING",    "</think>"    },
    {"CAIWEI_RERANKING_INSTRUCTION_KEY",    "<Instruct>: "},
    {"CAIWEI_RERANKING_QUERY_KEY",          "<Query>: "   },
    {"CAIWEI_RERANKING_DOCUMENT_KEY",       "<Document>: "},
    {"CAIWEI_RERANKING_SYSTEM_PROMPT",      R"(Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".)"},
    {"CAIWEI_RERANKING_INSTRUCTION_PROMPT", R"(Given a web search query, retrieve relevant passages that answer the query)"},
    // 模型配置: 类型,厂家,名称,路径
    // RKNN3   : LLM模型文件|LLM权重文件|embedding文件|tokenizer文件
    // RKNN3   : LLM模型文件|LLM权重文件|embedding文件|tokenizer文件|ASR模型文件|ASR权重文件
    // RKNN3   : LLM模型文件|LLM权重文件|embedding文件|tokenizer文件|VLM模型文件|VLM权重文件
    // LLAMACPP: LLM模型文件
    // LLAMACPP: LLM模型文件|ASR模型文件
    // LLAMACPP: LLM模型文件|VLM模型文件
#if ENABLE_CAIWEI_BACKEND_RKNN
    {"CAIWEI_CONTEXT_INFO", R"(
CLS,YOLO,yolo26n-cls,yolo26n-cls.rknn
DET,YOLO,yolo26n-det,yolo26n-det.rknn
ASR,QWEN,qwen3-asr,Qwen3-ASR/llm/Qwen3-ASR-0.6B.rknn|Qwen3-ASR/llm/Qwen3-ASR-0.6B.weight|Qwen3-ASR/llm/Qwen3-ASR-0.6B.embed.bin|Qwen3-ASR/llm/Qwen3-ASR-0.6B.tokenizer.gguf|Qwen3-ASR/asr/Qwen3-ASR-0.6B.rknn|Qwen3-ASR/asr/Qwen3-ASR-0.6B.weight
LLM,QWEN,qwen3-llm,Qwen3/Qwen3-1.7B.rknn|Qwen3/Qwen3-1.7B.weight|Qwen3/Qwen3-1.7B.embed.bin|Qwen3/Qwen3-1.7B.tokenizer.gguf
VLM,QWEN,qwen3-vlm,Qwen3-VL/llm/Qwen3-VL-2B-Instruct.rknn|Qwen3-VL/llm/Qwen3-VL-2B-Instruct.weight|Qwen3-VL/llm/Qwen3-VL-2B-Instruct.embed.bin|Qwen3-VL/llm/Qwen3-VL-2B-Instruct.tokenizer.gguf|Qwen3-VL/vlm/Qwen3-VL-2B-Instruct.rknn|Qwen3-VL/vlm/Qwen3-VL-2B-Instruct.weight
EMBEDDING,QWEN,qwen3-embedding,Qwen3-Embedding/Qwen3-Embedding-0.6B.rknn|Qwen3-Embedding/Qwen3-Embedding-0.6B.weight|Qwen3-Embedding/Qwen3-Embedding-0.6B.embed.bin|Qwen3-Embedding/Qwen3-Embedding-0.6B.tokenizer.gguf
RERANKING,QWEN,qwen3-reranking,Qwen3-Reranker/Qwen3-Reranker-0.6B.rknn|Qwen3-Reranker/Qwen3-Reranker-0.6B.weight|Qwen3-Reranker/Qwen3-Reranker-0.6B.embed.bin|Qwen3-Reranker/Qwen3-Reranker-0.6B.tokenizer.gguf
    )"},
#else
    {"CAIWEI_CONTEXT_INFO", R"(
CLS,YOLO,yolo26n-cls,yolo26n-cls.onnx
DET,YOLO,yolo26n-det,yolo26n-det.onnx
ASR,QWEN,qwen3-asr,Qwen3-ASR/Qwen3-ASR-0.6B-Q4_K_M.gguf|Qwen3-ASR/mmproj-Qwen3-ASR-0.6b-Q4_K_M.gguf
LLM,QWEN,qwen3-llm,Qwen3/Qwen3-1.7B-Q4_K_M.gguf
VLM,QWEN,qwen3-vlm,Qwen3-VL/Qwen3-VL-2B-Instruct-Q4_K_M.gguf|Qwen3-VL/mmproj-Qwen3-VL-2b-Instruct-Q4_K_M.gguf
EMBEDDING,QWEN,qwen3-embedding,Qwen3-Embedding/Qwen3-Embedding-0.6B-Q4_K_M.gguf
RERANKING,QWEN,qwen3-reranking,Qwen3-Reranker/Qwen3-Reranker-0.6B-Q4_K_M.gguf
    )"},
#endif
    // clang-format on
};

std::string caiwei::env::get(const std::string& name) {
    {
        std::shared_lock<std::shared_mutex> lock(env_mutex);
        if (auto iter = env_map.find(name); iter != env_map.end()) {
            return iter->second;
        }
    }
    {
        std::unique_lock<std::shared_mutex> lock(env_mutex);
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
}

bool caiwei::env::get_bool(const std::string& name) {
    std::string value = caiwei::env::get(name);
    return value == "ON" || value == "true" || value == "on" || value == "TRUE";
}

int caiwei::env::get_int(const std::string& name) {
    std::string value = caiwei::env::get(name);
    try {
        return std::stoi(value);
    } catch (...) {
        CW_LOG_W("配置错误: %s => %s", name.c_str(), value.c_str());
    }
    return 0;
}

float caiwei::env::get_float(const std::string& name) {
    std::string value = caiwei::env::get(name);
    try {
        return std::stof(value);
    } catch (...) {
        CW_LOG_W("配置错误: %s => %s", name.c_str(), value.c_str());
    }
    return 0.0F;
}

std::vector<std::string> caiwei::env::get_vector(const std::string& name, const char delim, const bool skip_empty) {
    std::string value = caiwei::env::get(name);
    if (value.empty()) {
        return {};
    }
    std::string line;
    std::vector<std::string> vector;
    std::stringstream stream(value);
    while (std::getline(stream, line, delim)) {
        line = caiwei::tool::strip(line);
        if (skip_empty && line.empty()) {
            continue;
        }
        vector.push_back(std::move(line));
    }
    return vector;
}

void caiwei::env::set(const std::string& name, const std::string& value, bool force) {
    // clang-format off
    if (
        !force &&
        (
            name == "CAIWEI_ALLOW_PATH"       ||
            name == "CAIWEI_ALLOW_HOST"       ||
            name == "CAIWEI_ALLOW_MEDIA_TYPE" ||
            name == "CAIWEI_CONTEXT_INFO"
        )
    ) {
        CW_LOG_W("配置禁止动态修改: %s", name.c_str());
        return;
    }
    // clang-format on
    {
        std::lock_guard<std::shared_mutex> lock(env_mutex);
        if (!default_config.contains(name)) {
            CW_LOG_W("不支持的环境配置: %s", name.c_str());
            return;
        }
        if (security_config.contains(name)) {
            CW_LOG_I("设置环境配置: %s = ****", name.c_str());
        } else {
            CW_LOG_I("设置环境配置: %s = %s", name.c_str(), value.c_str());
        }
        env_map.insert_or_assign(name, value);
    }
}

void caiwei::env::init() {
    std::call_once(env_map_once_flag, []() {
        allow_path = caiwei::env::get_vector("CAIWEI_ALLOW_PATH");
        allow_host = caiwei::env::get_vector("CAIWEI_ALLOW_HOST");
        allow_media_type = caiwei::env::get_vector("CAIWEI_ALLOW_MEDIA_TYPE");
        for (auto& path : allow_path) {
            std::error_code error_code;
            auto canon = std::filesystem::weakly_canonical(path, error_code);
            if (error_code) {
                path = std::filesystem::path(path).lexically_normal().string();
            } else {
                path = canon.string();
            }
            while (path.size() > 1 && (path.back() == '/' || path.back() == '\\')) {
                path.pop_back();
            }
        }
    });
    if (get_bool("CAIWEI_SECURITY")) {
        if (get("CAIWEI_APIKEY").empty() || get("CAIWEI_APIKEY") == "caiwei") {
            CW_LOG_W("当前使用默认密钥");
        }
        if (get("CAIWEI_USERNAME").empty() || get("CAIWEI_USERNAME") == "admin") {
            CW_LOG_W("当前使用默认账号");
        }
        if (get("CAIWEI_PASSWORD").empty() || get("CAIWEI_PASSWORD") == "admin") {
            CW_LOG_W("当前使用默认密码");
        }
    } else {
        CW_LOG_W("没有开启安全模式");
    }
}

void caiwei::env::stop() {
}

void caiwei::env::print_all_env() {
    for (const auto& [k, _] : default_config) {
        if (security_config.contains(k)) {
            CW_LOG_I("%-36s = ****", k.c_str());
        } else {
            CW_LOG_I("%-36s = %s", k.c_str(), get(k).c_str());
        }
    }
}

void caiwei::env::set_thread_name(const std::string& name) {
#if CAIWEI_OS_WIN
    std::wstring thread_name(name.begin(), name.end());
    SetThreadDescription(GetCurrentThread(), thread_name.c_str());
#else
    if (name.size() > THREAD_NAME_MAX_LENGTH) {
        std::string thread_name = name.substr(0, THREAD_NAME_MAX_LENGTH);
        pthread_setname_np(pthread_self(), thread_name.c_str());
    } else {
        pthread_setname_np(pthread_self(), name.c_str());
    }
#endif
}

const std::vector<std::string>& caiwei::env::get_allow_path() {
    return allow_path;
}

const std::vector<std::string>& caiwei::env::get_allow_host() {
    return allow_host;
}

const std::vector<std::string>& caiwei::env::get_allow_media_type() {
    return allow_media_type;
}
