import cv2

from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")
video = cv2.VideoCapture("../../../test/caiwei.mp4")
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    for result in model.predict(frame, conf=0.50, stream=True, verbose=False):
        if not result.boxes:
            continue
        for box, keypoint in zip(result.boxes, result.keypoints):
            # 画框
            cls = result.names[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
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
            # 骨架
            for i in range(keypoint.data.shape[0]):
                points = keypoint.data[i]
                for point in points:
                    x, y, c = map(float, point)
                    if c < 0.5:
                        continue
                    x = int(round(x))
                    y = int(round(y))
                    cv2.circle(frame, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
                #  0-鼻子
                #  1-左眼  2-右眼  3-左耳  4-右耳
                #  5-左肩  6-右肩  7-左肘  8-右肘  9-左腕   10-右腕
                # 11-左髋 12-右髋 13-左膝 14-右膝 15-左脚踝 16-右脚踝
                # fmt: off
                skeleton = [
                    [ 0,  1], [ 0,  2], [0, 3], [0, 4],
                    [ 5,  7], [ 7,  9],
                    [ 6,  8], [ 8, 10],
                    [11, 13], [13, 15],
                    [12, 14], [14, 16],
                ]
                # fmt: on
                for a, z in skeleton:
                    a_point = points[a]
                    z_point = points[z]
                    if a_point[2] < 0.5 or z_point[2] < 0.5:
                        continue
                    x1, y1 = int(a_point[0]), int(a_point[1])
                    x2, y2 = int(z_point[0]), int(z_point[1])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imshow("yolo26n", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video.release()
cv2.destroyAllWindows()
