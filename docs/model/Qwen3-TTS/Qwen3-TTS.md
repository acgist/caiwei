# Qwen3-TTS

|模型|说明|
|:--|:--|
|Qwen3-TTS-12Hz-1.7B-Base|语音克隆|
|Qwen3-TTS-12Hz-0.6B-CustomVoice|定制语音|
|Qwen3-TTS-12Hz-1.7B-VoiceDesign|语音设计|

## 模型下载

```
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir ./Qwen3-TTS-12Hz-1.7B-Base
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local_dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local_dir ./Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

## 模型训练
## 模型微调
## 模型导出
## 模型推理

```
python run_base.py
python run_voice_design.py
python run_custom_voice.py
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
