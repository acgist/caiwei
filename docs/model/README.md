# 模型

- 原版运行
- 模型微调

## 注意事项

* `jsonl`文件最后不能空行
* `dtype`和`torch_dtype`设置为`torch.float16`是因为`Tesla V100`显卡不支持`torch.bfloat16`
