# llama.cpp

* 版本: master

## 安装教程

```
git clone https://github.com/ggml-org/llama.cpp.git --depth=1
```

## 模型转换

### Qwen3

```
# TODO

python convert_hf_to_gguf.py \
  /data/caiwei/merged_hf_model \
  --outfile model.f16.gguf \
  --outtype f16

./llama‑quantize ./model.f16.gguf model.q4_k_m.gguf q4_k_m
```

### Qwen3-VL

## 相关链接

* https://github.com/ggml-org/llama.cpp
