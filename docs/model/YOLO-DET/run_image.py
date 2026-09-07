import cv2

from ultralytics import YOLO

model = YOLO("yolo26n.pt")
image = cv2.imread("../../../test/acgist.jpg")
results = model(image)
results[0].show()
