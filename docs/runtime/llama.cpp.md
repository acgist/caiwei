# llama.cpp

* 版本: master

## 安装教程

```
git clone https://github.com/ggml-org/llama.cpp.git --depth=1
```

## 模型转换

数据类型`f32/f16/bf16/q8_0`

### Qwen3

```
# 模型转换
python convert_hf_to_gguf.py Qwen/Qwen3-4B --outfile Qwen3-4B --outtype f16
# 模型量化
llama-quantize Qwen3-4B/Qwen3-4B-F16.gguf Qwen3-4B/Qwen3-4B-Q4_K_M.gguf q4_k_m
# 模型测试
llama-cli -m Qwen3-4B/Qwen3-4B-Q4_K_M.gguf --jinja
```

### Qwen3-VL

```
# 模型转换
python convert_hf_to_gguf.py Qwen/Qwen3-VL-2B-Instruct --outfile Qwen3-VL-2B-Instruct --outtype f16
python convert_hf_to_gguf.py Qwen/Qwen3-VL-2B-Instruct --outfile Qwen3-VL-2B-Instruct --outtype f16 --mmproj
# 模型量化
llama-quantize Qwen3-VL-2B-Instruct/Qwen3-VL-2B-Instruct-F16.gguf        Qwen3-VL-2B-Instruct/Qwen3-VL-2B-Instruct-Q4_K_M.gguf        q4_k_m
llama-quantize Qwen3-VL-2B-Instruct/mmproj-Qwen3-VL-2b-Instruct-F16.gguf Qwen3-VL-2B-Instruct/mmproj-Qwen3-VL-2b-Instruct-Q4_K_M.gguf q4_k_m
# 模型测试
llama-cli -m Qwen3-VL-2B-Instruct/Qwen3-VL-2B-Instruct-Q4_K_M.gguf -mm Qwen3-VL-2B-Instruct/mmproj-Qwen3-VL-2b-Instruct-Q4_K_M.gguf --jinja
```

### Qwen3-ASR

```
# 模型转换
python convert_hf_to_gguf.py Qwen/Qwen3-ASR-0.6B --outfile Qwen3-ASR-0.6B --outtype f16
python convert_hf_to_gguf.py Qwen/Qwen3-ASR-0.6B --outfile Qwen3-ASR-0.6B --outtype f16 --mmproj
# 模型量化
llama-quantize Qwen3-ASR-0.6B/Qwen3-ASR-0.6B-F16.gguf        Qwen3-ASR-0.6B/Qwen3-ASR-0.6B-Q4_K_M.gguf        q4_k_m
llama-quantize Qwen3-ASR-0.6B/mmproj-Qwen3-ASR-0.6b-F16.gguf Qwen3-ASR-0.6B/mmproj-Qwen3-ASR-0.6b-Q4_K_M.gguf q4_k_m
# 模型测试
llama-cli -m Qwen3-ASR-0.6B/Qwen3-ASR-0.6B-Q4_K_M.gguf -mm Qwen3-ASR-0.6B/mmproj-Qwen3-ASR-0.6b-Q4_K_M.gguf --jinja
```

### Qwen3-Reranker

```
# 模型转换
python convert_hf_to_gguf.py Qwen/Qwen3-Reranker-0.6B --outfile Qwen3-Reranker-0.6B --outtype f16
# 模型量化
llama-quantize Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B-F16.gguf Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B-Q4_K_M.gguf q4_k_m
# 模型测试
llama-server -m Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B-Q4_K_M.gguf --host 0.0.0.0 --reranking
```

### Qwen3-Embedding

```
# 模型转换
python convert_hf_to_gguf.py Qwen/Qwen3-Embedding-0.6B --outfile Qwen3-Embedding-0.6B --outtype f16
# 模型量化
llama-quantize Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B-F16.gguf Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B-Q4_K_M.gguf q4_k_m
# 模型测试
llama-server -m Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B-Q4_K_M.gguf --host 0.0.0.0 --embeddings
```

## 相关链接

* https://github.com/ggml-org/llama.cpp
* https://github.com/ggml-org/llama.cpp/blob/master/tools/cli
* https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd
* https://github.com/ggml-org/llama.cpp/tree/master/tools/server
* https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize