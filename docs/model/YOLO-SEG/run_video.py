import cv2
import numpy as np

from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")
video = cv2.VideoCapture("../../../test/caiwei.mp4")
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    for result in model.predict(
        frame, conf=0.50, stream=True, verbose=False, retina_masks=True
    ):
        if not result.boxes:
            continue
        for box, mask in zip(result.boxes, result.masks):
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
            # 掩码
            frame_h, frame_w = frame.shape[:2]
            mask = mask.data[0].numpy()
            # retina_masks=True
            mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
            # retina_masks=False
            # mask_h, mask_w = mask.shape
            # frame_mask_w = frame_w
            # frame_mask_h = int(frame_mask_w * mask_h / mask_w)
            # mask = cv2.resize(
            #     mask, (frame_mask_w, frame_mask_h), interpolation=cv2.INTER_LINEAR
            # )
            # mask = (mask > 0.5).astype(np.uint8)
            # mask = mask[
            #     int((frame_mask_h - frame_h) / 2) : int(
            #         (frame_mask_h - frame_h) / 2 + frame_h
            #     ),
            #     int((frame_mask_w - frame_w) / 2) : int(
            #         (frame_mask_w - frame_w) / 2 + frame_w
            #     ),
            # ]
            overlay = frame.copy()
            overlay[mask > 0] = (0, 255, 0)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.imshow("yolo26n", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video.release()
cv2.destroyAllWindows()
