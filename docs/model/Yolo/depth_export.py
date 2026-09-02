from ultralytics import YOLO

model = YOLO("yolo26n-depth.pt")
model.export(format="onnx")
