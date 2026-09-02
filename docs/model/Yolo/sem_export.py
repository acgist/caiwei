from ultralytics import YOLO

model = YOLO("yolo26n-sem.pt")
model.export(format="onnx")
