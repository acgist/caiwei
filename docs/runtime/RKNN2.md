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

```
pip install rknn-toolkit2
```

* `YOLO`官方工具需要`rknn-toolkit2`版本: `2.3.2`
* 追求速度建议使用瑞芯微官方定制的`YOLO`导出工具然后量化

### 生成数据

```
find ./dataset/ -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) > dataset.txt
```

### YOLO-CLS

```
# YOLO官方导出
python export_rknn.py
# 瑞芯微官方导出
python export_rknn_i8.py
```

### YOLO-DET

```
# YOLO官方导出
python export_rknn.py
# 官网参考
/ultralytics/nn/backends/rknn.py
/ultralytics/utils/export/rknn.py
/ultralytics/utils/export/engine.py
```

## 相关链接

* https://github.com/airockchip/rknn-llm
* https://github.com/airockchip/rknn-toolkit2
* https://github.com/airockchip/rknn_model_zoo
* https://docs.ultralytics.com/datasets
* https://docs.ultralytics.com/zh/modes/export
* https://docs.ultralytics.com/zh/integrations/rockchip-rknn
