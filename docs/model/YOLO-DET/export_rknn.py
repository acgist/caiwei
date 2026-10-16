from ultralytics import YOLO

model = YOLO("yolo26n.pt")
# I8
model.export(format="rknn", name="rk3588", quantize=8, data="dataset.yaml")
# F16
# model.export(format="rknn", name="rk3588")

# 量化图片: https://docs.ultralytics.com/datasets/detect/coco8
