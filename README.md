# 采薇

基于`C++23`开发的`AI`模型部署平台

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
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/acgist/caiwei?style=flat-square&label=Github%20stars&color=crimson" />
    <img alt="Gitee stars" src="https://img.shields.io/badge/dynamic/json?style=flat-square&label=Gitee%20stars&color=crimson&url=https://gitee.com/api/v5/repos/acgist/caiwei&query=$.stargazers_count&cacheSeconds=3600" />
    <br />
    <img alt="GitHub Workflow Status" src="https://img.shields.io/github/actions/workflow/status/acgist/caiwei/build.yml?style=flat-square&branch=master" />
    <img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/acgist/caiwei?style=flat-square&color=orange" />
    <img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/acgist/caiwei?style=flat-square&color=blue" />
    <img alt="GitHub" src="https://img.shields.io/github/license/acgist/caiwei?style=flat-square&color=blue" />
</p>

----

## 项目文档

* [部署文档](docs/deploy.md)
* [接口文档](docs/API.md)
* [模型文档](docs/model)
* [运行时库](docs/runtime)

## 运行时库

|运行时库|官方地址|支持计划|
|:--|:--|:--|
|RKNN2      |https://github.com/airockchip/rknn-toolkit2|√|
|RKNN3      |https://github.com/airockchip/rknn3-toolkit|√|
|llama.cpp  |https://github.com/ggml-org/llama.cpp      |√|
|ONNXRuntime|https://github.com/microsoft/onnxruntime   |√|

## 模型支持

|模型|类型|RKNN2|RKNN3|llama.cpp|ONNXRuntime|
|:--|:--|:--|:--|:--|:--|
|YOLO-CLS       |cls      |√|×|×|√|
|YOLO-DET       |det      |√|×|×|√|
|Qwen3          |llm      |×|√|√|×|
|Qwen3-VL       |vlm      |×|√|√|×|
|Qwen3-ASR      |asr      |×|√|√|×|
|Qwen3-Reranker |reranking|×|√|√|×|
|Qwen3-Embedding|embedding|×|√|√|×|
