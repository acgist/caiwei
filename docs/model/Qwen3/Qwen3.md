# Qwen3

## 模型下载

```
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./Qwen/Qwen3-0.6B
modelscope download --model Qwen/Qwen3-1.7B --local_dir ./Qwen/Qwen3-1.7B
modelscope download --model Qwen/Qwen3-4B --local_dir ./Qwen/Qwen3-4B
```

## 模型训练
## 模型微调

```
CUDA_VISIBLE_DEVICES=0               \
swift sft                            \
    --model Qwen/Qwen3-1.7B          \
    --model_type qwen3               \
    --tuner_type lora                \
    --dataset ./sft.jsonl            \
    --max_length 8192                \
    --torch_dtype float16            \
    --learning_rate 1e-4             \
    --num_train_epochs 1             \
    --save_steps 100                 \
    --save_total_limit 4             \
    --target_modules all-linear      \
    --gradient_checkpointing true    \
    --gradient_accumulation_steps 4  \
    --bnb_4bit_quant_type nf4        \
    --bnb_4bit_compute_dtype float16 \
    --bnb_4bit_use_double_quant true \
    --output_dir ./Qwen3-1.7B-lora
    # 忽略空的思考内容损失
    --loss_scale ignore_empty_think
    # 指定微调线性层的名称
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
```

> 没有思考内容自动填充思考内容`<think>\n\n</think>\n\n`
> 如果带有思考内容数据格式如下`<think>\n思考内容\n</think>\n\n`

## 模型导出

```
swift export              \
    --merge_lora true     \
    --torch_dtype float16 \
    --adapters Qwen3-1.7B-lora/v0-20260907-205115/checkpoint-1
```

## 模型推理

```
python run_lora.py
python run.py

CUDA_VISIBLE_DEVICES=0    \
swift infer               \
    --stream true         \
    --temperature 0       \
    --infer_backend pt    \
    --max_new_tokens 1024 \
    --adapters Qwen3-1.7B-lora/v0-20260907-205115/checkpoint-1

CUDA_VISIBLE_DEVICES=0    \
swift infer               \
    --stream true         \
    --temperature 0       \
    --infer_backend pt    \
    --max_new_tokens 1024 \
    --model Qwen3-1.7B-lora/v0-20260907-205115/checkpoint-1-merged
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

* https://github.com/QwenLM/Qwen3
* https://github.com/modelscope/ms-swift/
* https://modelscope.cn/models/Qwen/Qwen3-0.6B
* https://modelscope.cn/models/Qwen/Qwen3-1.7B
* https://modelscope.cn/models/Qwen/Qwen3-0.6B-Base
* https://modelscope.cn/models/Qwen/Qwen3-1.7B-Base
* https://modelscope.cn/models/Qwen/Qwen3-4B
* https://modelscope.cn/models/Qwen/Qwen3-4B-Base
* https://modelscope.cn/models/Qwen/Qwen3-4B-SafeRL
* https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507
* https://modelscope.cn/models/Qwen/Qwen3-4B-Thinking-2507
* https://modelscope.cn/models/Qwen/Qwen3-8B
* https://modelscope.cn/models/Qwen/Qwen3-8B-Base
* https://swift.readthedocs.io/zh-cn/latest/index.html
* https://www.modelscope.cn/docs/llm-training-and-inference/intro/swift-installation
* https://www.modelscope.cn/docs/llm-training-and-inference/user-guide/command-line-parameters
