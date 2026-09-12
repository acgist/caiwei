from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
model.export(format="rknn", name="rk3588")
