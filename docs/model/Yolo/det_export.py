from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.export(format="onnx", end2end=False)  # export