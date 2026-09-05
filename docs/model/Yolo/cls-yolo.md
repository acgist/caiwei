# YOLO图像分类

## 模型下载

## 模型训练

```
# 数据集
dataset-|
  |-train/训练图片文件
  |-val/验证图片文件
# 开始训练
python cls_train.py
```

## 模型微调

## 模型导出

```
python cls_export.py
```

## 模型推理

```
python cls_run.py
```

## 性能评估

* 图片大小: 1280*720

|设备|CANN|RKNN|llama.cpp|ONNXRuntime|Python|
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
