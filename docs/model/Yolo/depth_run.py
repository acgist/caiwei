import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import colorize_depth

model = YOLO("yolo26n-depth.pt")
video = cv2.VideoCapture(0)
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    for result in model.predict(frame, stream=True, verbose=False):
        # fmt: off
        depth = result.depth.data
        frame = cv2.addWeighted(frame, 0.5, colorize_depth(depth, cmap="spectral"), 0.5, 0)
        # fmt: on
    cv2.imshow("yolo26n-depth", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video.release()
cv2.destroyAllWindows()
