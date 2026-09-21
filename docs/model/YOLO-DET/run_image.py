import cv2

from ultralytics import YOLO

model = YOLO("yolo26n.pt")
image = cv2.imread("../../../test/acgist.jpg")
results = model(image)
results[0].show()

# import time
# a = time.time()
# for i in range(100):
#     results = model(image, verbose=False)
# z = time.time()
# print((z - a) * 1000 / 100)
