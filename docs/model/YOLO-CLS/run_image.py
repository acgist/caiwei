import cv2

from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
image = cv2.imread("../../../test/samoyed.jpg")
results = model(image)
results[0].show()

import time

model = YOLO("yolo26n-cls.onnx", task="classify")
a = time.time()
for i in range(1000):
    results = model(image, verbose=False)
z = time.time()
print((z - a) * 1000 / 1000)
