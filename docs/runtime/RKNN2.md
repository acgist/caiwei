# RKNN2

* 版本: `v2.3.2`
* 设备: `RK3566/RK3568/RK3576/RK3588`

* `RKNN2`跑`YOLO`模型
* `RKNN3`跑`Qwen`模型
* ~~`RKLLM`跑`Qwen`模型~~

## 安装教程

```
cat /sys/kernel/debug/rknpu/load
```

## 模型转换

**默认直接使用`YOLO`官方转换工具，追求极致速度可以使用瑞芯微的模型转换工具。**

```
pip install rknn-toolkit2
```

### YOLO-CLS

```
python export_rknn.py
```

### YOLO-DET

```
python export_rknn.py
```

### YOLO-SEG

```
python export_rknn.py
```

### YOLO-POSE

```
python export_rknn.py
```

## 相关链接

* https://github.com/airockchip/rknn-llm
* https://github.com/airockchip/rknn-toolkit2
* https://github.com/airockchip/rknn_model_zoo
* https://docs.ultralytics.com/datasets
* https://docs.ultralytics.com/zh/modes/export
* https://docs.ultralytics.com/zh/integrations/rockchip-rknn
