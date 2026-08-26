# 采薇

```
小雅·采薇

采薇采薇，薇亦作止。曰归曰归，岁亦莫止。靡室靡家，猃狁之故。不遑启居，猃狁之故。
采薇采薇，薇亦柔止。曰归曰归，心亦忧止。忧心烈烈，载饥载渴。我戍未定，靡使归聘。
采薇采薇，薇亦刚止。曰归曰归，岁亦阳止。王事靡盬，不遑启处。忧心孔疚，我行不来！
彼尔维何？维常之华。彼路斯何？君子之车。戎车既驾，四牡业业。岂敢定居？一月三捷。
驾彼四牡，四牡骙骙。君子所依，小人所腓。四牡翼翼，象弭鱼服。岂不日戒？猃狁孔棘！
昔我往矣，杨柳依依。今我来思，雨雪霏霏。行道迟迟，载渴载饥。我心伤悲，莫知我哀！
```

## 视觉模型

|功能|类型|YOLOv11|YOLOv26|支持计划|
|:--|:--|:--|:--|:--|
|图像分类|cls  |√|√|○|
|目标检测|det  |√|√|○|
|有向检测|obb  |√|√|○|
|实例分割|seg  |√|√|○|
|语义分割|sem  |×|√|○|
|姿态估计|pose |√|√|○|
|深度估计|depth|×|√|○|

## 语言模型

|功能|模型|下载|支持计划|
|:--|:--|:--|:--|
|ASR      |Qwen3-ASR            |https://modelscope.cn/models/Qwen/Qwen3-ASR-0.6B                 |○|
|         |                     |https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B                 |○|
|TTS      |Qwen3-TTS-Base       |https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base       |○|
|         |                     |https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base       |○|
|         |Qwen3-TTS-CustomVoice|https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice|○|
|         |                     |https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice|○|
|         |Qwen3-TTS-VoiceDesign|https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign|○|
|LLM      |Qwen3                |https://modelscope.cn/models/Qwen/Qwen3-0.6B                     |○|
|         |                     |https://modelscope.cn/models/Qwen/Qwen3-1.7B                     |○|
|         |                     |https://modelscope.cn/models/Qwen/Qwen3-4B                       |○|
|         |Qwen3-Base           |https://modelscope.cn/models/Qwen/Qwen3-0.6B-Base                |○|
|         |                     |https://modelscope.cn/models/Qwen/Qwen3-1.7B-Base                |○|
|         |                     |https://modelscope.cn/models/Qwen/Qwen3-4B-Base                  |○|
|         |Qwen3-SafeRL         |https://modelscope.cn/models/Qwen/Qwen3-4B-SafeRL                |○|
|         |Qwen3-Instruct-2507  |https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507         |○|
|         |Qwen3-Thinking-2507  |https://modelscope.cn/models/Qwen/Qwen3-4B-Thinking-2507         |○|
|VLM      |Qwen3-VL-Instruct    |https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct           |○|
|         |Qwen3-VL-Thinking    |https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Thinking           |○|
|MLLM     |Qwen3-Omni-Instruct  |https://modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Instruct    |○|
|         |Qwen3-Omni-Thinking  |https://modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Thinking    |○|
|         |Qwen3-Omni-Captioner |https://modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Captioner   |○|
|Reranking|Qwen3-Reranker       |https://modelscope.cn/models/Qwen/Qwen3-Reranker-0.6B            |○|
|         |Qwen3-VL-Reranker    |https://modelscope.cn/models/Qwen/Qwen3-VL-Reranker-2B           |○|
|Embedding|Qwen3-Embedding      |https://modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B           |○|
|         |Qwen3-VL-Embedding   |https://modelscope.cn/models/Qwen/Qwen3-VL-Embedding-2B          |○|

## 通用推理SDK

|SDK|官方地址|支持计划|
|:--|:--|:--|
|MNN        |https://github.com/alibaba/MNN          |○|
|llama.cpp  |https://github.com/ggml-org/llama.cpp   |○|
|ONNXRuntime|https://github.com/microsoft/onnxruntime|○|
|NCNN       |https://github.com/Tencent/ncnn         |×|
|LiteRT     |https://github.com/google-ai-edge/LiteRT|×|
|ExecuTorch |https://github.com/pytorch/executorch   |×|

## 厂商推理SDK

|SDK|官方地址|支持计划|
|:--|:--|:--|
|CANN         |https://www.hiascend.com/cann              |○|
|RKNN         |https://github.com/airockchip/rknn-toolkit2|○|
|             |https://github.com/airockchip/rknn3-toolkit|○|
|TensorRT     |https://github.com/NVIDIA/TensorRT         |×|
|OpenVINO     |https://github.com/openvinotoolkit/openvino|×|

> √=已支持 ○=计划中 ×=未支持
