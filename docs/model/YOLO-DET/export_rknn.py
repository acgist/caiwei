from ultralytics import YOLO

model = YOLO("yolo26n.pt")
# F16
# model.export(format="rknn", name="rk3588")
# 量化图片文件放到`val`目录
# I8
# https://docs.ultralytics.com/datasets/detect/coco8
model.export(format="rknn", name="rk3588", quantize=8, data="coco8.yaml")
