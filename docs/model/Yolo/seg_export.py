from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")
model.export(format="onnx")
