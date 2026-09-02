import cv2

from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
video = cv2.VideoCapture(0)
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    for result in model.predict(frame, stream=True, verbose=False):
        # fmt: off
        cls  = result.names[int(result.probs.top1)]
        conf = float(result.probs.top1conf)
        if conf < 0.40:
            continue
        cv2.putText(frame, f"{cls} {conf:.2f}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)
        # fmt: on
    cv2.imshow("yolo26n-cls", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video.release()
cv2.destroyAllWindows()
