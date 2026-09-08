import os

from rknn.api import RKNN

os.makedirs("./rknn", exist_ok=True)
rknn = RKNN(verbose=True)
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform="rk3588",
)
rknn.load_onnx("./yolo26n-pose.onnx")
# find ./coco8-pose/ -type f > dataset.txt
rknn.build(do_quantization=True, dataset="./dataset.txt")
rknn.export_rknn("./rknn/yolo26n-pose-rk3588-i8.rknn")
rknn.release()
