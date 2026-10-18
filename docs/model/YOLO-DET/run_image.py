import cv2

from ultralytics import YOLO

model = YOLO("yolo26n.pt")
image = cv2.imread("../../../test/acgist.jpg")
results = model(image)
results[0].show()

import time

model = YOLO("yolo26n.onnx", task="detect")
a = time.time()
for i in range(1000):
    results = model(image, verbose=False)
z = time.time()
print((z - a) * 1000 / 1000)
