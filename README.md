# 采薇

采薇一个`AI`模型部署平台，主要支持`YOLO`和`QWEN`相关模型，支持`RKNN`、`CUDA`等等后端。

```
小雅·采薇

采薇采薇，薇亦作止。曰归曰归，岁亦莫止。靡室靡家，猃狁之故。不遑启居，猃狁之故。
采薇采薇，薇亦柔止。曰归曰归，心亦忧止。忧心烈烈，载饥载渴。我戍未定，靡使归聘。
采薇采薇，薇亦刚止。曰归曰归，岁亦阳止。王事靡盬，不遑启处。忧心孔疚，我行不来！
彼尔维何？维常之华。彼路斯何？君子之车。戎车既驾，四牡业业。岂敢定居？一月三捷。
驾彼四牡，四牡骙骙。君子所依，小人所腓。四牡翼翼，象弭鱼服。岂不日戒？猃狁孔棘！
昔我往矣，杨柳依依。今我来思，雨雪霏霏。行道迟迟，载渴载饥。我心伤悲，莫知我哀！
```

<p align="center">
    <a target="_blank" href="https://starchart.cc/acgist/caiwei">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/acgist/caiwei?style=flat-square&label=Github%20stars&color=crimson" />
    </a>
    <img alt="Gitee stars" src="https://img.shields.io/badge/dynamic/json?style=flat-square&label=Gitee%20stars&color=crimson&url=https://gitee.com/api/v5/repos/acgist/caiwei&query=$.stargazers_count&cacheSeconds=3600" />
    <br />
    <img alt="GitHub Workflow Status" src="https://img.shields.io/github/actions/workflow/status/acgist/caiwei/build.yml?style=flat-square&branch=master" />
    <img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/acgist/caiwei?style=flat-square&color=orange" />
    <img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/acgist/caiwei?style=flat-square&color=blue" />
    <img alt="GitHub" src="https://img.shields.io/github/license/acgist/caiwei?style=flat-square&color=blue" />
</p>

----

## 通用推理SDK

|SDK|官方地址|支持计划|
|:--|:--|:--|
|llama.cpp  |https://github.com/ggml-org/llama.cpp   |○|
|ONNXRuntime|https://github.com/microsoft/onnxruntime|○|

## 厂商推理SDK

|SDK|官方地址|支持计划|
|:--|:--|:--|
|CANN|https://www.hiascend.com/cann              |○|
|RKNN|https://github.com/airockchip/rknn-toolkit2|○|
|    |https://github.com/airockchip/rknn3-toolkit|○|

## 后端支持

|设备|llama.cpp|ONNXRuntime|
|:--|:--|:--|
|CANN    |?|?|
|RKNN    |?|?|
|CUDA    |○|○|
|OpenVINO|○|○|

## 模型支持

|模型|CANN|RKNN|llama.cpp|ONNXRuntime|
|:--|:--|:--|:--|:--|
|YOLO-CLS          |○|○|×|√|
|YOLO-DET          |○|○|×|√|
|YOLO-OBB          |?|?|×|?|
|YOLO-SEG          |○|○|×|√|
|YOLO-SEM          |?|?|×|?|
|YOLO-POSE         |○|○|×|√|
|YOLO-DEPTH        |?|?|×|?|
|Qwen3-ASR         |○|○|○|×|
|Qwen3-TTS         |?|?|?|×|
|Qwen3             |○|○|√|×|
|Qwen3-VL          |○|○|○|×|
|Qwen3-Omni        |?|?|?|×|
|Qwen3-Reranker    |○|○|○|×|
|Qwen3-Embedding   |○|○|○|×|
|Qwen3-VL-Reranker |?|?|?|×|
|Qwen3-VL-Embedding|?|?|?|×|

> √=已支持 ○=计划中 ?=未计划 ×=不支持

## 日志前缀

```
* = 修改代码
+ = 增加功能
- = 减少功能
~ = 优化代码
% = 重要更新
@ = 修复问题
! = 版本发布
$ = 配置更新
# = 文档更新
& = 依赖升级
? = 其他修改
```
