# API

## GET /v1/health
## GET /v1/models
## GET /v1/video/play

* `url` : 视频地址
* `type`: 媒体类型(`rtp|sdp|file|http|rtmp|rtsp|device`)

## POST /v1/rerankings

```
{
    "model": "模型名称",
    "input": "文档内容" | [ "文档内容", ... ]
}

{
    "object": "list",
    "model" : "模型名称",
    "data"  : [
        {
            "index"    : "文档内容索引",
            "object"   : "embedding",
            "embedding": [嵌入向量...]
        },
        ...
    ],
    "usage": {
        "prompt_tokens": 提示词消耗的token数量
        "total_tokens" : 请求消耗总的token数量
    }
}
```

## POST /v1/embeddings

```
{
    "model"    : "模型名称",
    "query"    : "查询内容",
    "documents": [ "文档内容", ... ],
    "instruct" : "提示指令"
}

{
    "object": "list",
    "model" : "模型名称",
    "data"  : [
        {
            "index" : "文档内容索引",
            "object": "reranking",
            "score" : 文档评分
        },
        ...
    ],
    "usage": {
        "prompt_tokens": 提示词消耗的token数量
        "total_tokens" : 请求消耗总的token数量
    }
}
```

## POST /v1/chat/completions

* 单个模型参数`model`就是标准`OpenAI`接口
* 指定多个模型`model_list`输出结果根据`model`字段判断类型
* 持续识别参数`media_url`/`media_type`支持`ASR`/`VLM`/`YOLO`模型

```
{
    "stream"  : 是否流式返回内容
    "model"   : "模型名称",
    "messages": [
        {
            "role"        : "角色",
            "name"        : "名称",
            "tool_call_id": "方法ID",
            "content"     : 消息内容 | [
                {
                    "type"     : "消息类型",
                    "text"     : "文本消息",
                    "audio"    : "音频消息",
                    "image"    : "图片消息",
                    "video"    : "视频消息",
                    "audio_url": "音频消息",
                    "image_url": "图片消息",
                    "video_url": "视频消息"
                },
                ...
            ],
            "tool_calls": [
                {
                    "id"      : "方法ID",
                    "type"    : "方法类型",
                    "function": {
                        "name"     : "方法名称",
                        "arguments": "方法参数",
                    },
                },
                ...
            ]
        },
        ...
    ],
    "seed" : 随机种子,
    "top_k": top_k,
    "top_p": top_p,
    "temperature": 温度,
    "repeat_penalty": 重复惩罚,
    "presence_penalty": 存在惩罚,
    "frequency_penalty": 频率惩罚,
    "max_completion_tokens": 最大推理token数量,
    "tools": [
        {
            "type"    : "方法类型",
            "function": {
                "name"       : "方法名称",
                "description": "方法描述",
                "parameters" : {
                    参数描述
                }
            }
        },
        ...
    ],
    "extra_body": {
        "video_fps"      : 视频识别间隔帧数,
        "asr_samples"    : ASR 识别帧数大小,
        "asr_queue_size" : ASR 识别队列大小,
        "vlm_frames"     : VLM 识别帧数大小,
        "yolo_queue_size": YOLO识别队列大小,
        "media_url"      : "持续识别媒体地址",
        "media_type"     : "持续识别媒体类型",
        "enable_thinking": 是否开启思考模式,
        "model_list"     : ["模型列表", ...],
    },
}

{
    "created": 创建时间
    "id"     : "消息ID",
    "model"  : "模型名称",
    "object" : "chat.completion",
    "choices": [
        {
            "index"        : 消息索引,
            "finish_reason": "完成原因",
            "message"      : {
                "role"             : "角色",
                "content"          : "生成内容",
                "refusal"          : "拒绝原因",
                "reasoning_content": "思考内容",
                "tool_calls"       : [
                    {
                        "id"   : "方法ID",
                        "type" : "方法类型",
                        "index": 方法索引,
                        "function": {
                            "name"     : "方法名称",
                            "arguments": "方法参数",
                        }
                    },
                    ...
                ],
            }
        },
        ...
    ],
    "usage": {
        "prompt_tokens"    : 提示词消耗的token数量
        "completion_tokens": 生成时消耗的token数量
        "total_tokens"     : 请求消耗总的token数量
    }
}

{
    "created": 创建时间
    "id"     : "消息ID",
    "model"  : "模型名称",
    "object" : "chat.completion.chunk",
    "choices": [
        {
            "index"        : 消息索引,
            "finish_reason": "完成原因",
            "delta"        : {
                "role"             : "角色",
                "content"          : "生成内容",
                "refusal"          : "拒绝原因",
                "reasoning_content": "思考内容",
                "tool_calls"       : [
                    {
                        "id"   : "方法ID",
                        "type" : "方法类型",
                        "index": 方法索引,
                        "function": {
                            "name"     : "方法名称",
                            "arguments": "方法参数",
                        }
                    },
                    ...
                ],
            }
        },
        ...
    ],
    "usage": {
        "prompt_tokens"    : 提示词消耗的token数量
        "completion_tokens": 生成时消耗的token数量
        "total_tokens"     : 请求消耗总的token数量
    }
}
```

## 参考文档

* https://github.com/openai/openai-openapi
