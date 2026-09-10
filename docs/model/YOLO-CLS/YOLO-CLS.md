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
python export_onnx.py
```

## 模型推理

```
python run_image.py
python run.py
```

## 性能评估

* 图片大小: 1280*720

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

* https://docs.ultralytics.com/zh
* https://docs.ultralytics.com/zh/modes/train
* https://docs.ultralytics.com/zh/modes/export
* https://docs.ultralytics.com/zh/modes/predict
* https://docs.ultralytics.com/zh/models/yolo26
* https://docs.ultralytics.com/zh/tasks/classify
* https://docs.ultralytics.com/zh/guides/yolo-data-augmentation
