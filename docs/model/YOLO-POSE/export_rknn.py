from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")
# F16
model.export(format="rknn", name="rk3588")
# 官方不支持量化
# I8
# https://docs.ultralytics.com/datasets/pose/coco8-pose
# model.export(format="rknn", name="rk3588", quantize=8, data="coco8-pose.yaml")
