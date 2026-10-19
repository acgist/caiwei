# YOLO图像分类

## 模型下载
## 模型训练

```
# 数据集
dataset-|
  |-train/类型1/训练图片文件列表
  |-train/类型2/训练图片文件列表
  |-val/类型1/验证图片文件列表
  |-val/类型2/验证图片文件列表
# 开始训练
python train.py
```

## 模型微调
## 模型导出

```
# ONNX
python export_onnx.py
# RKNN
python export_rknn.py
```

## 模型推理

```
python run_image.py
python run_video.py
```

## 性能评估

* 模型`yolo26n`
* 图片大小: `512x512`
* 括号内容表示模型消耗时间
* `RK3588`数据表示`i8/f16`精度下的性能
* `RK3588`设置大核`taskset -c 4-7 ./test_yolo_cls`(性能更好)
* `RK3588`设置大核`taskset -c 4-7 ./test_yolo_runtime`(性能更好)

|设备|RKNN|llama.cpp|ONNXRuntime|原版|
|:--|:--|:--|:--|:--|
|CPU(i5-1135G7)  |-           |-|3ms(2ms)|10ms|
|RKNN(RK1828)    |-           |-|-       |-   |
|RKNN(RK3588)    |4/5ms(3/5ms)|-|-       |-   |
|CUDA(RTX 4060)  |-           |-|-       |-   |
|CUDA(Tesla L40S)|-           |-|-       |-   |
|CUDA(Tesla V100)|-           |-|1ms(1ms)|-   |

## 文档资料

* https://docs.ultralytics.com/zh
* https://docs.ultralytics.com/zh/modes/train
* https://docs.ultralytics.com/zh/modes/export
* https://docs.ultralytics.com/zh/modes/predict
* https://docs.ultralytics.com/zh/models/yolo26
* https://docs.ultralytics.com/zh/tasks/classify
* https://docs.ultralytics.com/zh/guides/yolo-data-augmentation
