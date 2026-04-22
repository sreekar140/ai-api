import cv2
import torch
import time
import threading
from ultralytics import YOLO
from fastapi import FastAPI
from contextlib import asynccontextmanager

# -------- DEVICE --------
device = '0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# -------- SHARED STATE (one per hall) --------
shared_data = {
    1: {"count": 0, "status": "low", "last_updated": 0},
    2: {"count": 0, "status": "low", "last_updated": 0},
    3: {"count": 0, "status": "low", "last_updated": 0},
    4: {"count": 0, "status": "low", "last_updated": 0},
}
state_lock = threading.Lock()

# -------- STATUS --------
def get_status(count, capacity=20):
    percentage = (count / capacity) * 100
    if percentage < 35:
        return "low"
    elif percentage < 65:
        return "medium"
    else:
        return "high"

# -------- VIDEO LOOP (each hall gets its OWN model instance) --------
def run_video_loop(hall_id, video_file):
    # Each thread loads its own model — fixes the thread-safety crash
    model = YOLO("yolov8m.pt")
    print(f"Hall {hall_id} thread started! Video: {video_file}")

    cap = cv2.VideoCapture(video_file)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0:
        fps = 25
    frame_delay = 1 / fps
    print(f"Hall {hall_id} FPS: {fps}")

    DETECTION_INTERVAL = 3
    last_detection_time = 0
    last_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Hall {hall_id} video ended, restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            last_ids = set()
            continue

        current_time = time.time()

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
            for r in results:
                if r.boxes.id is not None:
                    ids = r.boxes.id.cpu().numpy().astype(int)
                    for track_id in ids:
                        current_ids.add(track_id)

            last_ids = current_ids
            last_detection_time = current_time
            count = len(last_ids)

            with state_lock:
                shared_data[hall_id]["count"] = count
                shared_data[hall_id]["status"] = get_status(count)
                shared_data[hall_id]["last_updated"] = current_time

            print(f"Hall {hall_id} count updated: {count}")

        time.sleep(frame_delay)

    cap.release()

# -------- VIDEOS CONFIG --------
# Swap in your actual filenames here
VIDEOS = {
    1: "video.mp4",
    2: "video1.mp4",
    3: "video2.mp4",
    4: "video3.mp4",
}

# -------- LIFESPAN --------
@asynccontextmanager
async def lifespan(app: FastAPI):
    for hall_id, video_file in VIDEOS.items():
        thread = threading.Thread(target=run_video_loop, args=(hall_id, video_file), daemon=True)
        thread.start()
        print(f"Hall {hall_id} video loop started!")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# -------- HELPER --------
def make_response(zone, count, status):
    return {
        "zone": zone,
        "count": count,
        "capacity": 20,
        "status": status,
    }

# -------- SINGLE ENDPOINT --------
@app.get("/all")
def get_all():
    with state_lock:
        return {
            "hall1": make_response("Dining Hall 1", shared_data[1]["count"], shared_data[1]["status"]),
            "hall2": make_response("Dining Hall 2", shared_data[2]["count"], shared_data[2]["status"]),
            "hall3": make_response("Dining Hall 3", shared_data[3]["count"], shared_data[3]["status"]),
            "hall4": make_response("Dining Hall 4", shared_data[4]["count"], shared_data[4]["status"]),
        }

# uvicorn detect_people:app --host 0.0.0.0 --port 8000
# ngrok http --domain=smog-baboon-gloomily.ngrok-free.dev 8000
