from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")
model.export(format="rknn", name="rk3588")
