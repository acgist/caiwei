from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")
# F16
model.export(format="rknn", name="rk3588")
# 官方不支持量化
# I8
# https://docs.ultralytics.com/datasets/segment/coco8-seg
# model.export(format="rknn", name="rk3588", quantize=8, data="coco8-seg.yaml")
