# RKNN3

* 版本: `RK1820/RK1828`

* `RKNN2`跑`YOLO`模型
* `RKNN3`跑`Qwen`模型

## 安装教程

```
pip install -r requirements_cp312-1.1.0.txt
pip install rknn3_toolkit-1.1.0-cp312-cp312-manylinux2014_x86_64.whl
```

* 不要随便修改依赖🤡🤡🤡(遇到依赖问题重新安装环境)
* 尽量保证`rknn3-toolkit`/`rknn3-model-zoo`和`rknn3_api`版本一致

## 模型转换

如果开启量化使用参考官方文档。

### YOLO-CLS
### YOLO-DET
### YOLO-SEG
### YOLO-POSE

### Qwen3

```
# https://github.com/airockchip/rknn3-model-zoo/tree/main/examples/Qwen3

cd rknn3-model-zoo/examples/Qwen3/python

mkdir onnx rknn

python export_llm.py                \
    --modelscope                    \
    --model_path      Qwen/Qwen3-4B \
    --export_llm_path onnx/Qwen3-4B.onnx

python export_rknn.py                    \
    --platform  rk1828                   \
    --config    onnx/Qwen3-4B.config.pkl \
    --onnx_path onnx/Qwen3-4B.onnx       \
    --rknn_path rknn/Qwen3-4B.rknn

scp                              \
    onnx/Qwen3-4B.config.pkl     \
    onnx/Qwen3-4B.embed.bin      \
    onnx/Qwen3-4B.tokenizer.gguf \
    rknn/Qwen3-4B.*              \
    192.168.1.100:/data/model/Qwen3-4B

rkllm3-server                             \
    -a          "Qwen3-4B"                \
    -c          4096                      \
    --port      8000                      \
    --host      0.0.0.0                   \
    --model     ./Qwen3-4B.rknn           \
    --embed     ./Qwen3-4B.embed.bin      \
    --vocab     ./Qwen3-4B.tokenizer.gguf \
    --log_level 0
```

### Qwen3-VL

```
# https://github.com/airockchip/rknn3-model-zoo/tree/main/examples/Qwen3_VL

cd rknn3-model-zoo/examples/Qwen3_VL/python/vision

mkdir onnx-vlm rknn-vlm

python export_vision.py                            \
    --modelscope                                   \
    --model_path         Qwen/Qwen3-VL-2B-Instruct \
    --export_vision_path onnx-vlm/Qwen3-VL-2B-Instruct.onnx

python export_rknn.py                              \
    --platform  rk1828                             \
    --onnx_path onnx-vlm/Qwen3-VL-2B-Instruct.onnx \
    --rknn_path rknn-vlm/Qwen3-VL-2B-Instruct.rknn

scp                                 \
    rknn-vlm/Qwen3-VL-2B-Instruct.* \
    192.168.1.100:/data/model/Qwen3-VL-2B-Instruct/vlm

cd rknn3-model-zoo/examples/Qwen3_VL/python/llm

mkdir onnx-llm rknn-llm

python export_llm.py                            \
    --modelscope                                \
    --model_path      Qwen/Qwen3-VL-2B-Instruct \
    --export_llm_path onnx-llm/Qwen3-VL-2B-Instruct.onnx

python export_rknn.py                                    \
    --platform  rk1828                                   \
    --config    onnx-llm/Qwen3-VL-2B-Instruct.config.pkl \
    --onnx_path onnx-llm/Qwen3-VL-2B-Instruct.onnx       \
    --rknn_path rknn-llm/Qwen3-VL-2B-Instruct.rknn

scp                                              \
    onnx-llm/Qwen3-VL-2B-Instruct.config.pkl     \
    onnx-llm/Qwen3-VL-2B-Instruct.embed.bin      \
    onnx-llm/Qwen3-VL-2B-Instruct.tokenizer.gguf \
    rknn-llm/Qwen3-VL-2B-Instruct.*              \
    192.168.1.100:/data/model/Qwen3-VL-2B-Instruct/llm

rkllm3-server                                               \
    -a            "Qwen3-VL-2B-Instruct"                    \
    --port        8000                                      \
    --host        0.0.0.0                                   \
    --model       ./llm/Qwen3-VL-2B-Instruct.rknn           \
    --embed       ./llm/Qwen3-VL-2B-Instruct.embed.bin      \
    --vocab       ./llm/Qwen3-VL-2B-Instruct.tokenizer.gguf \
    --model2      ./vlm/Qwen3-VL-2B-Instruct.rknn           \
    --img-start   "<|vision_start|>"                        \
    --img-end     "<|vision_end|>"                          \
    --img-content "<|image_pad|>"                           \
    --img-width   384                                       \
    --img-height  384                                       \
    --log_level   0
```

### Qwen3-ASR

```
# https://github.com/airockchip/rknn3-model-zoo/tree/main/examples/Qwen3_ASR

cd rknn3-model-zoo/examples/Qwen3_ASR/python/audio/online

mkdir onnx-asr rknn-asr

python export_audio_onnx.py                   \
    --modelscope                              \
    --model_path          Qwen/Qwen3-ASR-0.6B \
    --export_encoder_path onnx-asr/Qwen3-ASR-0.6B.onnx

python export_audio_rknn.py                  \
    --platform  rk1828                       \
    --onnx_path onnx-asr/Qwen3-ASR-0.6B.onnx \
    --rknn_path rknn-asr/Qwen3-ASR-0.6B.rknn

scp                           \
    rknn-asr/Qwen3-ASR-0.6B.* \
    192.168.1.100:/data/model/Qwen3-ASR-0.6B/asr

cd rknn3-model-zoo/examples/Qwen3_ASR/python/llm

mkdir onnx-llm rknn-llm

python export_llm.py                      \
    --modelscope                          \
    --model_path      Qwen/Qwen3-ASR-0.6B \
    --export_llm_path onnx-llm/Qwen3-ASR-0.6B.onnx

python export_rknn.py                              \
    --platform  rk1828                             \
    --config    onnx-llm/Qwen3-ASR-0.6B.config.pkl \
    --onnx_path onnx-llm/Qwen3-ASR-0.6B.onnx       \
    --rknn_path rknn-llm/Qwen3-ASR-0.6B.rknn

scp                                        \
    onnx-llm/Qwen3-ASR-0.6B.config.pkl     \
    onnx-llm/Qwen3-ASR-0.6B.embed.bin      \
    onnx-llm/Qwen3-ASR-0.6B.tokenizer.gguf \
    rknn-llm/Qwen3-ASR-0.6B.*              \
    192.168.1.100:/data/model/Qwen3-ASR-0.6B/llm

rkllm3-server                                           \
    -a              "Qwen3-ASR-0.6B"                    \
    --port          8000                                \
    --host          0.0.0.0                             \
    --model         ./llm/Qwen3-ASR-0.6B.rknn           \
    --embed         ./llm/Qwen3-ASR-0.6B.embed.bin      \
    --vocab         ./llm/Qwen3-ASR-0.6B.tokenizer.gguf \
    --model3        ./asr/Qwen3-ASR-0.6B.rknn           \
    --audio-start   "<|audio_start|>"                   \
    --audio-end     "<|audio_end|>"                     \
    --audio-content "<|audio_pad|>"                     \
    --log_level     0
```

### Qwen3-Reranker

```
# https://github.com/airockchip/rknn3-model-zoo/tree/main/examples/Qwen3_Reranker

cd rknn3-model-zoo/examples/Qwen3_Reranker/python

mkdir onnx rknn

python export_llm.py                           \
    --modelscope                               \
    --model_path      Qwen/Qwen3-Reranker-0.6B \
    --export_llm_path onnx/Qwen3-Reranker-0.6B.onnx

python export_rknn.py                               \
    --platform  rk1820                              \
    --config    onnx/Qwen3-Reranker-0.6B.config.pkl \
    --onnx_path onnx/Qwen3-Reranker-0.6B.onnx       \
    --rknn_path rknn/Qwen3-Reranker-0.6B.rknn

scp                                         \
    onnx/Qwen3-Reranker-0.6B.config.pkl     \
    onnx/Qwen3-Reranker-0.6B.embed.bin      \
    onnx/Qwen3-Reranker-0.6B.tokenizer.gguf \
    rknn/Qwen3-Reranker-0.6B.*              \
    192.168.1.100:/data/model/Qwen3-Reranker-0.6B

rkllm3-server                                        \
    -a          "Qwen3-Reranker-0.6B"                \
    --port      8000                                 \
    --host      0.0.0.0                              \
    --model     ./Qwen3-Reranker-0.6B.rknn           \
    --embed     ./Qwen3-Reranker-0.6B.embed.bin      \
    --vocab     ./Qwen3-Reranker-0.6B.tokenizer.gguf \
    --log_level 0
```

### Qwen3-Embedding

```
# https://github.com/airockchip/rknn3-model-zoo/tree/main/examples/Qwen3_Embedding

cd rknn3-model-zoo/examples/Qwen3_Embedding/python

mkdir onnx rknn

python export_llm.py                            \
    --modelscope                                \
    --model_path      Qwen/Qwen3-Embedding-0.6B \
    --export_llm_path onnx/Qwen3-Embedding-0.6B.onnx

python export_rknn.py                                \
    --platform  rk1828                               \
    --config    onnx/Qwen3-Embedding-0.6B.config.pkl \
    --onnx_path onnx/Qwen3-Embedding-0.6B.onnx       \
    --rknn_path rknn/Qwen3-Embedding-0.6B.rknn

scp                                          \
    onnx/Qwen3-Embedding-0.6B.config.pkl     \
    onnx/Qwen3-Embedding-0.6B.embed.bin      \
    onnx/Qwen3-Embedding-0.6B.tokenizer.gguf \
    rknn/Qwen3-Embedding-0.6B.*              \
    192.168.1.100:/data/model/Qwen3-Embedding-0.6B

rkllm3-server                                         \
    -a          "Qwen3-Embedding-0.6B"                \
    --port      8000                                  \
    --host      0.0.0.0                               \
    --model     ./Qwen3-Embedding-0.6B.rknn           \
    --embed     ./Qwen3-Embedding-0.6B.embed.bin      \
    --vocab     ./Qwen3-Embedding-0.6B.tokenizer.gguf \
    --log_level 0                                     \
    --embedding
```

## 相关链接

* https://github.com/airockchip/rknn3-toolkit
* https://github.com/airockchip/rknn3-model-zoo
* https://www.rock-chips.com/a/cn/news/rockchip/2026/0309/2163.html
* https://github.com/airockchip/rknn3-toolkit/blob/main/rknn3-runtime/rkllm3-server/README_CN.md
