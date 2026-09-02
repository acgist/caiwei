from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")
model.export(format="onnx")
