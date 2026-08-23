# Qwen3-VL

## 模型下载
## 模型训练
## 模型微调

```
CUDA_VISIBLE_DEVICES=0                \
swift sft                             \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --model_type qwen3_vl             \
    --tuner_type lora                 \
    --dataset ./sft.jsonl             \
    --optim paged_adamw_8bit          \
    --target_modules all-linear       \
    --lora_rank 8                     \
    --lora_alpha 16                   \
    --max_length 8192                 \
    --torch_dtype float16             \
    --warmup_ratio 0.05               \
    --learning_rate 1e-4              \
    --num_train_epochs 3              \
    --freeze_vit true                 \
    --freeze_aligner true             \
    --save_steps 100                  \
    --save_total_limit 5              \
    --gradient_checkpointing true     \
    --gradient_accumulation_steps 4   \
    --per_device_train_batch_size 1   \
    --bnb_4bit_quant_type nf4         \
    --bnb_4bit_compute_dtype float16  \
    --bnb_4bit_use_double_quant true  \
    --output_dir ./Qwen3-VL-4B-Instruct-lora
    # 微调线性层
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
```

## 模型导出

### LoRA

```
CUDA_VISIBLE_DEVICES=0 \
swift export           \
    --merge_lora true  \
    --adapters Qwen3-VL-4B-Instruct-lora/v0-20260715-094947/checkpoint-3/

CUDA_VISIBLE_DEVICES=0                             \
swift export                                       \
    --merge_lora true                              \
    --quant_bits 4                                 \
    --quant_method bnb                             \
    --bnb_4bit_quant_type nf4                      \
    --bnb_4bit_use_double_quant true               \
    --output_dir Qwen3-VL-4B-Instruct-lora-BNB-NF4 \
    --adapters Qwen3-VL-4B-Instruct-lora/v0-20260715-094947/checkpoint-3/
```

### RKNN

```
python export_llm.py                               \
--model_path      /data/rkllm/Qwen3-VL-4B-Instruct \
--export_llm_path /data/rkllm/onnx/Qwen3-VL-4B-Instruct.onnx

python export_rknn.py                                        \
--platform  rk1828                                           \
--onnx_path /data/rkllm/onnx/Qwen3-VL-4B-Instruct.onnx       \
--config    /data/rkllm/onnx/Qwen3-VL-4B-Instruct.config.pkl \
--rknn_path /data/rkllm/rknn/Qwen3-VL-4B-Instruct.rknn

python export_vision.py                               \
--model_path         /data/rkllm/Qwen3-VL-4B-Instruct \
--export_vision_path /data/rkllm/onnx/Qwen3-VL-4B-Instruct.onnx

python export_rknn.py                                  \
--platform  rk1828                                     \
--onnx_path /data/rkllm/onnx/Qwen3-VL-4B-Instruct.onnx \
--rknn_path /data/rkllm/rknn/Qwen3-VL-4B-Instruct.rknn \
--no_prune_mode

rkllm3-server                                             \
  -a            "Qwen3-VL-4B-Instruct"                    \
  --port        8000                                      \
  --host        0.0.0.0                                   \
  --model       ./llm/Qwen3-VL-4B-Instruct.rknn           \
  --vocab       ./llm/Qwen3-VL-4B-Instruct.tokenizer.gguf \
  --embed       ./llm/Qwen3-VL-4B-Instruct.embed.bin      \
  --model2      ./vlm/Qwen3-VL-4B-Instruct.rknn           \
  --img-start   "<|vision_start|>"                        \
  --img-end     "<|vision_end|>"                          \
  --img-content "<|image_pad|>"                           \
  --log_level   0
```

## 模型推理

```
CUDA_VISIBLE_DEVICES=0    \
swift infer               \
    --stream true         \
    --temperature 0       \
    --infer_backend pt    \
    --max_new_tokens 2048 \
    --adapters Qwen3-VL-4B-Instruct-lora/v0-20260715-094947/checkpoint-3/

CUDA_VISIBLE_DEVICES=0        \
swift infer                   \
    --stream true             \
    --temperature 0           \
    --infer_backend vllm      \
    --max_new_tokens 2048     \
    --merge_lora true         \
    --vllm_max_model_len 8192 \
    --adapters Qwen3-VL-4B-Instruct-lora/v0-20260715-094947/checkpoint-3/

CUDA_VISIBLE_DEVICES=0    \
swift infer               \
    --stream true         \
    --temperature 0       \
    --infer_backend pt    \
    --max_new_tokens 2048 \
    --model Qwen3-VL-4B-Instruct-lora/v0-20260715-094947/checkpoint-3-merged/

<video>

/data/vlm/sft/video.mp4
```

## 性能评估

|设备|MNN|RKNN|llama.cpp|ONNXRuntime|
|:--|:--|:--|:--|:--|
|RKNN(RK1828)|||||
|CUDA(RTX4090)|||||
|OpenVINO(i5-1135G7)|||||

## 文档资料
