# RKNN

* 版本: 

* `RKNN3`跑`Qwen`模型

* RKNN3: RK1820/RK1828

## 安装教程

```
```

## 模型转换

### Qwen3

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

### Qwen3-VL

```
python export_llm.py                               \
--model_path      /data/rkllm/Qwen3-VL-4B-Instruct \
--export_llm_path /data/rkllm/onnx/Qwen3-VL-4B-Instruct.onnx

python export_rknn.py                                        \
--platform  rk1828                                           \
--onnx_path /data/rkllm/onnx/Qwen3-VL-4B-Instruct.onnx       \
--config    /data/rkllm/onnx/Qwen3-VL-4B-Instruct.config.pkl \
--rknn_path /data/rkllm/rknn/Qwen3-VL-4B-Instruct.rknn

python export_vision.py                               \
--model_path         /data/rkllm/Qwen3-VL-4B-Instruct \
--export_vision_path /data/rkllm/onnx/Qwen3-VL-4B-Instruct.onnx

python export_rknn.py                                  \
--platform  rk1828                                     \
--onnx_path /data/rkllm/onnx/Qwen3-VL-4B-Instruct.onnx \
--rknn_path /data/rkllm/rknn/Qwen3-VL-4B-Instruct.rknn \
--no_prune_mode

rkllm3-server                                             \
  -a            "Qwen3-VL-4B-Instruct"                    \
  --port        8000                                      \
  --host        0.0.0.0                                   \
  --model       ./llm/Qwen3-VL-4B-Instruct.rknn           \
  --vocab       ./llm/Qwen3-VL-4B-Instruct.tokenizer.gguf \
  --embed       ./llm/Qwen3-VL-4B-Instruct.embed.bin      \
  --model2      ./vlm/Qwen3-VL-4B-Instruct.rknn           \
  --img-start   "<|vision_start|>"                        \
  --img-end     "<|vision_end|>"                          \
  --img-content "<|image_pad|>"                           \
  --log_level   0
```

## 相关链接

* https://github.com/airockchip/rknn3-toolkit
* https://github.com/airockchip/rknn3-model-zoo
