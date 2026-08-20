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

## 支持模型

### 视觉模型

|功能|类型|YOLOv11|YOLOv26|
|:--|:--|:--|:--|
|图像分类|cls|√|√|
|目标检测|det|√|√|
|有向检测|obb|√|√|
|实例分割|seg|√|√|
|语义分割|sem|×|√|
|姿态估计|pose|√|√|
|深度估计|depth|×|√|

### 语言模型

|功能|模型|
|:--|:--|
|ASR|Qwen3-ASR / Whisper|
|TTS|Qwen3-TTS|
|LLM|Qwen3 / DeepSeek-R1|
|VLM|Qwen3-VL|
|MLLM|Qwen3-Omni|
|Reranking|Qwen3-Reranker|
|Embedding|Qwen3-Embedding|

### 其他模型

|功能|模型|
|:--|:--|
|人脸检测|SCRFD|
|特征提取|ArcFace|
|噪声抑制|RNNoise|
|活动检测|Silero-VAD|

## 运行环境

|运行环境|官方地址|
|:--|:--|
|MNN|https://github.com/alibaba/MNN|
|RKNN|https://github.com/airockchip|
|Ascend|https://github.com/Ascend|
|OpenVINO|https://github.com/openvinotoolkit/openvino|
|TensorRT|https://github.com/NVIDIA/TensorRT|
|llama.cpp|https://github.com/ggml-org/llama.cpp|
|ONNXRuntime|https://github.com/microsoft/onnxruntime|

> 支持顺序：ONNXRuntime -> RKNN -> OpenVINO -> llama.cpp -> TensorRT -> MNN -> Ascend

## 依赖项目

|依赖项目|官方地址|
|:--|:--|
|HTTP||
|JSON||
|FFMPEG||
