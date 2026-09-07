# API

## /v1/health

健康装

## /v1/models

模型列表

## /v1/video/play

视频播放

## /v1/rerankings

文档重排

## /v1/embeddings

文档嵌入

## /v1/chat/completions

会话补全

* `LLM` : 文本聊天(`model`)
* `ASR` : 语音识别(`model/model_asr`)
* `VLM` : 视频理解(`model/model_vlm`)
* `YOLO`: 视觉模型(`model/model_yolo`)
* 正常单个模型参数`model`就是标准`OpenAI`接口
* 如果指定多个模型输出结果根据`model`字段判断类型
