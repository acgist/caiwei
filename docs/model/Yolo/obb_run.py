import cv2

from ultralytics import YOLO

model = YOLO("yolo26n-obb.pt")
video = cv2.VideoCapture(0)
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    for result in model.predict(frame, conf=0.60, stream=True, verbose=False):
        # fmt: off
        print(result.obb)
        box  = result.obb.xyxyxyxy
        # conf = float(result.obb.conf.numpy())
        cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)
        # cv2.putText(frame, f"{conf:.2f}", (box[0][0], box[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)
        # fmt: on
    cv2.imshow("yolo26n", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video.release()
cv2.destroyAllWindows()
