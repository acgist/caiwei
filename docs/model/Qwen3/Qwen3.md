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
## 模型推理

```
python run.py
```

## 性能评估

|设备|CANN|RKNN|llama.cpp|ONNXRuntime|Python|
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

* https://github.com/QwenLM
* https://qwen.readthedocs.io/zh-cn/latest/
* https://modelscope.cn/models/Qwen/Qwen3-4B
* https://modelscope.cn/models/Qwen/Qwen3-4B-Base
* https://modelscope.cn/models/Qwen/Qwen3-4B-SafeRL
* https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507
* https://modelscope.cn/models/Qwen/Qwen3-4B-Thinking-2507
