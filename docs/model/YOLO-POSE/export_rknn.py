from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")
model.export(format="rknn", name="rk3588")
