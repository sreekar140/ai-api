import cv2
import torch
import time
from ultralytics import YOLO

# -------- DEVICE --------
device = '0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# -------- MODEL --------
model = YOLO("yolov8m.pt")  # use n for smooth display

# -------- VIDEO --------
cap = cv2.VideoCapture("video4.mp4")if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# FPS handling (safe)
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps == 0:
    fps = 25

frame_delay = int(1000 / fps)
print(f"Video FPS: {fps}")

# -------- CONTROL --------
DETECTION_INTERVAL = 3  # seconds

last_detection_time = 0
last_boxes = []
last_ids = set()

# -------- WINDOW --------
cv2.namedWindow("Crowd Detection", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video ended, restarting...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    current_time = time.time()

    # -------- DETECTION --------
    if current_time - last_detection_time >= DETECTION_INTERVAL:

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=0.4,
            device=device,
            verbose=False
        )

        current_ids = set()
        boxes_data = []

        for r in results:
            if r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, ids):
                    boxes_data.append((box, track_id))
                    current_ids.add(track_id)

        last_boxes = boxes_data
        last_ids = current_ids
        last_detection_time = current_time

    # -------- DRAW --------
    for box, track_id in last_boxes:
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    count = len(last_ids)

    # -------- DISPLAY --------
    cv2.putText(frame, f"People Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Crowd Detection", frame)

    # -------- EXIT --------
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()