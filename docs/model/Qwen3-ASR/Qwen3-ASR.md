# Qwen3-ASR

|英文缩写|英文全称|中文术语|
|:--|:--|:--|
|AEC |Acoustic Echo Cancellation  |回声消除|
|ANS |Acoustic Noise Suppression  |噪声抑制、降噪|
|VAD |Voice Activity Detection    |语音活动检测、静音检测|
|AGC |Automatic Gain Control      |自动增益控制|
|ASR |Automatic Speech Recognition|自动语音识别、语音转写|
|Diar|Speaker Diarization         |说话人分簇|
|SID |Speaker Identification      |说话人识别、声纹识别|

### 降噪算法

* https://github.com/xiph/rnnoise
* https://modelscope.cn/models/iic/speech_frcrn_ans_cirm_16k
* https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base

> 模型对小音量和噪声音频识别支持较好所以不用额外操作

## 模型下载

```
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ./Qwen/Qwen3-ASR-0.6B
```

## 模型训练
## 模型微调
## 模型导出
## 模型推理

```
python run.py
```

## 性能评估

* 模型`Qwen3-ASR-0.6B`
* `llama.cpp`量化格式`Q4_K_M`
* `Prefill/Generate`每秒`token`数量

|设备|RKNN|llama.cpp|ONNXRuntime|原版|
|:--|:--|:--|:--|:--|
|CPU(i5-1135G7)  |-       |70/60  |-|-|
|RKNN(RK1828)    |1100/120|-      |-|-|
|RKNN(RK3588)    |-       |-      |-|-|
|CUDA(RTX 4060)  |-       |-      |-|-|
|CUDA(Tesla L40S)|-       |-      |-|-|
|CUDA(Tesla V100)|-       |600/220|-|-|

## 文档资料

* https://github.com/modelscope/ms-swift/
* https://swift.readthedocs.io/zh-cn/latest/index.html
* https://www.modelscope.cn/docs/llm-training-and-inference/intro/swift-installation
* https://www.modelscope.cn/docs/llm-training-and-inference/user-guide/command-line-parameters
* https://github.com/QwenLM/Qwen3-ASR
* https://modelscope.cn/models/Qwen/Qwen3-ASR-0.6B
* https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B
