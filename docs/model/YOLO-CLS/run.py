import cv2

from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
video = cv2.VideoCapture("../../../test/caiwei.mp4")
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    for result in model.predict(frame, stream=True, verbose=False):
        cls = result.names[int(result.probs.top1)]
        conf = float(result.probs.top1conf)
        if conf < 0.50:
            continue
        cv2.putText(
            frame,
            f"{cls} {conf:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            1,
        )
    cv2.imshow("yolo26n", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video.release()
cv2.destroyAllWindows()
