from ultralytics import YOLO
import cv2
import numpy as np
import threading
from collections import deque

# ========================= CONFIG =========================
model = YOLO("yolov8l.pt")
cap = cv2.VideoCapture("video1.mp4")

CONF_THRESHOLD    = 0.30
IOU_THRESHOLD     = 0.40
NMS_THRESHOLD     = 0.40
DETECT_EVERY_N    = 3        # Every 3 frames
FRAME_W, FRAME_H  = 640, 480
MIN_BOX_RATIO     = 0.005    # Min 0.5% of frame
MAX_BOX_RATIO     = 0.40     # Max 40% of frame (filters walls)
MIN_ASPECT        = 0.20     # w/h ratio min
MAX_ASPECT        = 3.5      # w/h ratio max
SMOOTH_LEN        = 6        # Smoothing window
# =========================================================

FRAME_AREA  = FRAME_W * FRAME_H
latest_boxes = []
latest_count = 0
lock         = threading.Lock()
is_detecting = False
history      = deque(maxlen=SMOOTH_LEN)
frame_count  = 0


def valid_box(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False
    ratio  = (w * h) / FRAME_AREA
    aspect = w / h
    return (MIN_BOX_RATIO <= ratio <= MAX_BOX_RATIO and
            MIN_ASPECT <= aspect <= MAX_ASPECT)


def detect(frame):
    global latest_boxes, latest_count, is_detecting
    h, w = frame.shape[:2]
    all_boxes = []

    # Run on 3 orientations to handle angled/tilted cameras
    orientations = [
        (frame,                              "orig"),
        (cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE),    "cw"),
        (cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE), "ccw"),
    ]

    for img, mode in orientations:
        results = model(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                        verbose=False, imgsz=640)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != 0:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Remap rotated coordinates back to original frame
                if mode == "cw":
                    x1, y1, x2, y2 = (h - y2, x1, h - y1, x2)
                elif mode == "ccw":
                    x1, y1, x2, y2 = (y1, w - x2, y2, w - x1)

                # Clamp and validate
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                if valid_box(x1, y1, x2, y2):
                    all_boxes.append((x1, y1, x2, y2, conf))

    # Merge all orientations with NMS
    if all_boxes:
        bxywh  = [[x1, y1, x2-x1, y2-y1] for x1,y1,x2,y2,_ in all_boxes]
        scores = [c for *_, c in all_boxes]
        idxs   = cv2.dnn.NMSBoxes(bxywh, scores, CONF_THRESHOLD, NMS_THRESHOLD)
        final  = [all_boxes[i] for i in idxs.flatten()] if len(idxs) else []
    else:
        final = []

    with lock:
        latest_boxes = final
        latest_count = len(final)
    is_detecting = False


# ─────────────── Draw helpers ────────────────────────────
def draw_box(frame, x1, y1, x2, y2, conf, idx):
    color = (0, 230, 0) if conf >= 0.55 else (0, 180, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"#{idx+1} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    lx, ly = x1, max(y1 - 4, th + 4)
    cv2.rectangle(frame, (lx, ly - th - 2), (lx + tw + 4, ly + 2), color, -1)
    cv2.putText(frame, label, (lx + 2, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(frame, count):
    # Semi-transparent dark panel
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Big count
    cv2.putText(frame, str(count), (16, 58),
                cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 255, 120), 3, cv2.LINE_AA)
    cv2.putText(frame, "people", (90, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2, cv2.LINE_AA)


# ─────────────── Main loop ───────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame       = cv2.resize(frame, (FRAME_W, FRAME_H))
    display     = frame.copy()
    frame_count += 1

    # Kick off background detection every N frames
    if frame_count % DETECT_EVERY_N == 0 and not is_detecting:
        is_detecting = True
        threading.Thread(target=detect, args=(frame.copy(),), daemon=True).start()

    # Draw latest results
    with lock:
        boxes = list(latest_boxes)
        count = latest_count

    for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
        draw_box(display, x1, y1, x2, y2, conf, i)

    # Smooth the count
    if frame_count % DETECT_EVERY_N == 0:
        history.append(count)
    stable = round(sum(history) / len(history)) if history else 0

    draw_hud(display, stable)

    cv2.imshow("People Counter", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()