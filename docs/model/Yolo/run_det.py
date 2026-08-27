import cv2
from ultralytics import YOLO

# model = YOLO("yolo11s.pt")
model = YOLO("yolo26s.pt")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    res = model.predict(frame, conf=0.60, verbose=False)[0]
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        c = int(box.cls[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{c} {conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            1,
        )

    cv2.imshow("yolo26", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
