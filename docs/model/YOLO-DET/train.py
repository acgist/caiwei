from ultralytics import YOLO

model = YOLO("yolo26n.pt", task="detect")
model.train(
    # 参数列表: https://docs.ultralytics.com/zh/modes/train
    # 数据增强: https://docs.ultralytics.com/zh/guides/yolo-data-augmentation
    # fmt: off
    single_cls = False,
    data       = "./dataset.yaml",
    imgsz      = 640,
    epochs     = 128,
    batch      = 64,
    shear      = 5.0,
    erasing    = 0.2,
    degrees    = 15.0,
    # fmt: on
)
model.export(format="onnx", end2end=False)
