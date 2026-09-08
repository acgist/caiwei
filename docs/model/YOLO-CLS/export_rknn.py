from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
# F16
model.export(format="rknn", name="rk3588")
# 官方不支持量化
# I8
# https://docs.ultralytics.com/datasets/classify/cifar10
# model.export(format="rknn", name="rk3588", quantize=8, data="cifar10")
