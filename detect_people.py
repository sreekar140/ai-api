import cv2
import torch
from ultralytics import YOLO
from fastapi import FastAPI

app = FastAPI()

# -------- DEVICE --------
device = '0' if torch.cuda.is_available() else 'cpu'

# -------- MODEL --------
model = YOLO('yolov8n.pt')  # changed only model size

# -------- VIDEO PATHS --------
video_paths = {
    "hall1": "video1.mp4",
    "hall2": "video1.mp4",
    "hall3": "video1.mp4",
    "hall4": "video1.mp4",
}

# -------- TIME TRACKING --------
current_times = {
    "hall1": 0,
    "hall2": 0,
    "hall3": 0,
    "hall4": 0,
}

INTERVAL = 2


# -------- FUNCTION --------
def get_count(hall):
    current_time = current_times[hall]

    cap = cv2.VideoCapture(video_paths[hall])
    cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        current_times[hall] = 0
        return 0

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    results = model.predict(
        frame,
        classes=[0],
        conf=0.4,
        device=device,
        verbose=False
    )

    count = sum(len(r.boxes) for r in results)

    # update time properly
    current_times[hall] = current_time + INTERVAL

    return count


# -------- STATUS --------
def get_status(count, capacity=100):
    if count < capacity * 0.5:
        return "low"
    elif count < capacity * 0.75:
        return "medium"
    else:
        return "high"


# -------- APIs --------
@app.get("/data1")
def hall1():
    count = get_count("hall1")
    return {"zone": "Dining Hall 1", "count": count, "capacity": 100, "status": get_status(count)}


@app.get("/data2")
def hall2():
    count = get_count("hall2")
    return {"zone": "Dining Hall 2", "count": count, "capacity": 100, "status": get_status(count)}


@app.get("/data3")
def hall3():
    count = get_count("hall3")
    return {"zone": "Dining Hall 3", "count": count, "capacity": 100, "status": get_status(count)}


@app.get("/data4")
def hall4():
    count = get_count("hall4")
    return {"zone": "Dining Hall 4", "count": count, "capacity": 100, "status": get_status(count)}