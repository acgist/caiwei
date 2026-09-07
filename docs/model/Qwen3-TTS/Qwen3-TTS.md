# Qwen3-TTS

## 模型下载

```
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir ./Qwen/Qwen3-TTS-12Hz-1.7B-Base
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local_dir ./Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local_dir ./Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

## 模型训练
## 模型微调
## 模型导出
## 模型推理

```
python run_voice_clone.py
python run_voice_design.py
python run_voice_custom.py
```

## 性能评估
## 文档资料

* https://github.com/QwenLM/Qwen3-TTS
* https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base
* https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base
* https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
* https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
* https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
* https://modelscope.cn/models/Qwen/Qwen3-TTS-Tokenizer-12Hz
* https://modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B
