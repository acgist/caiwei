# Qwen3-ASR

执行流程: ANS -> VAD -> AGC -> ASR

完整语音处理流程

- AEC
- ANS
- VAD
- AGC
- ASR
- Diar
- SID

|英文缩写|英文全称|中文术语|
|:--|:--|:--|
|AEC |Acoustic Echo Cancellation  |回声消除|
|ANS |Acoustic Noise Suppression  |噪声抑制、降噪|
|VAD |Voice Activity Detection    |语音活动检测、语音端点检测、静音检测|
|AGC |Automatic Gain Control      |自动增益控制、音量自动调节|
|ASR |Automatic Speech Recognition|自动语音识别、语音转写|
|Diar|Speaker Diarization         |说话人分簇、说话人日志|
|SID |Speaker Identification      |说话人识别、声纹识别|

## 模型下载

```
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ./Qwen3-ASR-0.6B
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
