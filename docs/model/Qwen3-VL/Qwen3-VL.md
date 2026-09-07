# Qwen3-VL

## 模型下载

```
modelscope download --model Qwen/Qwen3-VL-2B-Instruct --local_dir ./Qwen/Qwen3-VL-2B-Instruct
modelscope download --model Qwen/Qwen3-VL-2B-Thinking --local_dir ./Qwen/Qwen3-VL-2B-Thinking
```

## 模型训练
## 模型微调

```
FPS_MAX_FRAMES=8                      \
CUDA_VISIBLE_DEVICES=0                \
swift sft                             \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --model_type qwen3_vl             \
    --tuner_type lora                 \
    --dataset ./sft.jsonl             \
    --max_length 8192                 \
    --torch_dtype float16             \
    --learning_rate 1e-4              \
    --num_train_epochs 1              \
    --freeze_vit true                 \
    --freeze_aligner true             \
    --save_steps 100                  \
    --save_total_limit 4              \
    --target_modules all-linear       \
    --gradient_checkpointing true     \
    --gradient_accumulation_steps 4   \
    --bnb_4bit_quant_type nf4         \
    --bnb_4bit_compute_dtype float16  \
    --bnb_4bit_use_double_quant true  \
    --output_dir ./Qwen3-VL-2B-Instruct-lora
    # 指定微调的线性层
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
```

## 模型导出

```
swift export              \
    --merge_lora true     \
    --torch_dtype float16 \
    --adapters Qwen3-VL-2B-Instruct-lora/v0-20260907-170906/checkpoint-1
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
    --adapters Qwen3-VL-2B-Instruct-lora/v0-20260907-170906/checkpoint-1

CUDA_VISIBLE_DEVICES=0    \
swift infer               \
    --stream true         \
    --temperature 0       \
    --infer_backend pt    \
    --max_new_tokens 1024 \
    --model Qwen3-VL-2B-Instruct-lora/v0-20260907-170906/checkpoint-1-merged

<image>
/data/caiwei/test/caiwei.jpg

<video>
/data/caiwei/test/caiwei.mp4
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

* https://github.com/QwenLM/Qwen3-VL
* https://github.com/modelscope/ms-swift/
* https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct
* https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Thinking
* https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct
* https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Thinking
* https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct
* https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Thinking
* https://swift.readthedocs.io/zh-cn/latest/index.html
* https://www.modelscope.cn/docs/llm-training-and-inference/intro/swift-installation
* https://www.modelscope.cn/docs/llm-training-and-inference/user-guide/command-line-parameters
