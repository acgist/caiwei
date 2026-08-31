# Qwen3

|模型|说明|
|:--|:--|
|Qwen3-4B|支持思考开关|
|Qwen3-4B-Base|基础模型|
|Qwen3-4B-SafeRL|支持思考开关|
|Qwen3-4B-Instruct-2507|没有思考|
|Qwen3-4B-Thinking-2507|必须思考|

## 模型下载

```
modelscope download --model Qwen/Qwen3-4B --local_dir ./Qwen3-4B
```

## 模型训练
## 模型微调
## 模型导出

### RKNN

```
python export_llm.py                   \
--model_path      /data/rkllm/Qwen3-4B \
--export_llm_path /data/rkllm/onnx/Qwen3-4B.onnx

python export_rknn.py                            \
--platform  rk1828                               \
--onnx_path /data/rkllm/onnx/Qwen3-4B.onnx       \
--config    /data/rkllm/onnx/Qwen3-4B.config.pkl \
--rknn_path /data/rkllm/rknn/Qwen3-4B.rknn

scp onnx/Qwen3-4B.* rknn/Qwen3-4B.* 192.168.1.100:/data/model

rkllm3-server                           \
  -a          "Qwen3-4B"                \
  --port      8000                      \
  --host      0.0.0.0                   \
  --model     ./Qwen3-4B.rknn           \
  --vocab     ./Qwen3-4B.tokenizer.gguf \
  --embed     ./Qwen3-4B.embed.bin      \
  --log_level 0

rkllm3-server                                       \
  -a          "Qwen3-Embedding-0.6B"                \
  --port      8000                                  \
  --host      0.0.0.0                               \
  --model     ./Qwen3-Embedding-0.6B.rknn           \
  --vocab     ./Qwen3-Embedding-0.6B.tokenizer.gguf \
  --embed     ./Qwen3-Embedding-0.6B.embed.bin      \
  --log_level 0                                     \
  --embedding
```

## 模型推理

```
python run.py
```

## 性能评估

|设备|CANN|RKNN|llama.cpp|ONNXRuntime|
|:--|:--|:--|:--|:--|
|CANN(Atlas350)|||||
|RKNN(RK1828)|||||
|CUDA(RTX4090)|||||
|OpenVINO(i5-1135G7)|||||

## 文档资料

* https://github.com/QwenLM
* https://qwen.readthedocs.io/zh-cn/latest/
* https://modelscope.cn/models/Qwen/Qwen3-4B
* https://modelscope.cn/models/Qwen/Qwen3-4B-Base
* https://modelscope.cn/models/Qwen/Qwen3-4B-SafeRL
* https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507
* https://modelscope.cn/models/Qwen/Qwen3-4B-Thinking-2507
