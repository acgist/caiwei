# Qwen3-Embedding

## 模型下载

```
modelscope download --model Qwen/Qwen3-Embedding-0.6B --local_dir ./Qwen/Qwen3-Embedding-0.6B
```

## 模型训练
## 模型微调
## 模型导出
## 模型推理

```
python run.py
```

## 性能评估

|设备|CANN|RKNN|llama.cpp|ONNXRuntime|原版|
|:--|:--|:--|:--|:--|:--|
|CANN(Ascend 310P)  ||||||
|CANN(Ascend 910C)  ||||||
|RKNN(RK1828)       ||||||
|RKNN(RK3588)       ||||||
|CUDA(Tesla L40S)   ||||||
|CUDA(Tesla V100)   ||||||
|CUDA(RTX 5060 Ti)  ||||||
|OpenVINO(i5-1135G7)||||||

## 文档资料

* https://github.com/QwenLM/Qwen3-Embedding
* https://modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B
* https://modelscope.cn/models/Qwen/Qwen3-Embedding-4B
* https://modelscope.cn/models/Qwen/Qwen3-Embedding-8B
