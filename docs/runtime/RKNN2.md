# RKNN2

* 版本: `v2.3.2`
* 设备: `RK3566/RK3568/RK3576/RK3588`

* `RKNN2`跑`YOLO`模型
* `RKNN3`跑`Qwen`模型
* ~~`RKLLM`跑`Qwen`模型~~

## 安装教程

```
# 性能查看
cat /sys/kernel/debug/rknpu/load
```

## 模型转换

```
pip install rknn-toolkit2
```

* `YOLO`官方工具需要`rknn-toolkit2`版本`2.3.2`

### 量化数据

```
find ./dataset/ -type f -iregex '.*\.\(jpg\|jpeg\|png\)$' > dataset.txt
```

### YOLO-CLS

```
# YOLO官方导出
python export_rknn.py
# rknn-toolkit2导出
python export_rknn_i8.py
```

### YOLO-DET

```
# YOLO官方导出
python export_rknn.py
# 位置输出没有归一化不能直接使用rknn-toolkit2量化导出，建议使用瑞芯微定制`YOLO`模型工具导出模型然后量化。
```

## 相关链接

* https://github.com/airockchip/rknn-toolkit2
* https://github.com/airockchip/rknn_model_zoo
* https://docs.ultralytics.com/datasets
* https://docs.ultralytics.com/zh/modes/export
* https://docs.ultralytics.com/zh/integrations/rockchip-rknn
