# YOLO目标检测

## 模型下载
## 模型训练

```
# 数据集
dataset.yaml
dataset-|
  |-train/训练图片文件列表
  |-train/训练标记文件列表
  |-train/classes.txt
  |-val/验证图片文件列表
  |-val/验证标记文件列表
  |-val/classes.txt
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

* 图片大小: `1280*720`
* 括号内容表示模型消耗时间
* `RK3588`数据表示`i8/f16/i8-toolkit`精度下的性能

|设备|RKNN|llama.cpp|ONNXRuntime|原版|
|:--|:--|:--|:--|:--|
|CPU(i5-1135G7)   |-                     |-|40ms(30ms)|52ms|
|RKNN(RK1828)     |-                     |-|-         |-   |
|RKNN(RK3588)     |45/90/45ms(35/65/35ms)|-|-         |-   |
|CUDA(RTX 4060)   |-                     |-|-         |-   |
|CUDA(Tesla L40S) |-                     |-|-         |-   |
|CUDA(Tesla V100) |-                     |-|50ms(8ms) |-   |

## 文档资料

* https://docs.ultralytics.com/zh
* https://docs.ultralytics.com/zh/modes/train
* https://docs.ultralytics.com/zh/modes/export
* https://docs.ultralytics.com/zh/modes/predict
* https://docs.ultralytics.com/zh/models/yolo26
* https://docs.ultralytics.com/zh/tasks/detect
* https://docs.ultralytics.com/zh/guides/yolo-data-augmentation
