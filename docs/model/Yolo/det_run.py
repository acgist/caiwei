import cv2

from ultralytics import YOLO

model = YOLO("yolo26n.pt")
video = cv2.VideoCapture(0)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    for result in model.predict(frame, conf=0.60, stream=True, verbose=False):
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls  = result.names[int(box.cls[0])]
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(
                frame,
                f"{cls} {conf:.2f}",
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                1,
            )
    cv2.imshow("yolo-det", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video.release()
cv2.destroyAllWindows()
