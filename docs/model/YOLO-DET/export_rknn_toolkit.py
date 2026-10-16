import os

from rknn.api import RKNN

os.makedirs("./rknn", exist_ok=True)
rknn = RKNN(verbose=True)
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform="rk3588",
    disable_rules=["fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa"],
)
rknn.load_onnx("./yolo26n-det.onnx")
rknn.build(do_quantization=True, dataset="./dataset.txt")
rknn.export_rknn("./rknn/yolo26n-det-rk3588-i8-toolkit.rknn")
rknn.release()
