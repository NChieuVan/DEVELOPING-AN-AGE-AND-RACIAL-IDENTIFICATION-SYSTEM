import cv2
import time
import threading
import numpy as np

from load_model import load_tracking_model, load_age_race_classifier
from write_log import FPSStressLogger

# ================== CONFIG ==================
CAMERA_ID = 0
FRAME_W, FRAME_H = 640, 360

MAX_CLASSIFY = 2
CLS_REFRESH_SEC = 2.0
FPS_HIST_LEN = 30

# ================== SHARED STATE ==================
latest_frame = None
latest_draw = None
stop_flag = False

frame_lock = threading.Lock()
draw_lock = threading.Lock()

cls_cache = {}

# ================== FPS LOGGER ==================
fps_logger = FPSStressLogger(
    window_sec=5 * 60,
    log_file="stress_test_fps_log.txt"
)

# ================== IOU TRACKER ==================
class IoUTracker:
    def __init__(self, iou_thres=0.4, max_lost_sec=0.6):
        self.iou_thres = iou_thres
        self.max_lost_sec = max_lost_sec
        self.tracks = {}
        self.next_id = 1

    @staticmethod
    def iou(a, b):
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
        areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
        return inter / (areaA + areaB - inter + 1e-6)

    def update(self, detections):
        now = time.time()
        used = set()

        for tid, t in list(self.tracks.items()):
            best_iou, best_idx = 0.0, -1
            for i, d in enumerate(detections):
                if i in used:
                    continue
                iou = self.iou(t["bbox"], d)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou >= self.iou_thres and best_idx >= 0:
                self.tracks[tid]["bbox"] = detections[best_idx]
                self.tracks[tid]["last_seen"] = now
                used.add(best_idx)

        for i, d in enumerate(detections):
            if i not in used:
                self.tracks[self.next_id] = {
                    "bbox": d,
                    "last_seen": now
                }
                self.next_id += 1

        for tid in list(self.tracks.keys()):
            if now - self.tracks[tid]["last_seen"] > self.max_lost_sec:
                del self.tracks[tid]

        return [{"id": tid, "bbox": t["bbox"]} for tid, t in self.tracks.items()]

# ================== LOAD MODELS ==================
print("Loading models...")
yolo = load_tracking_model()
clf = load_age_race_classifier()
tracker = IoUTracker()
print("Models loaded!")

# ================== THREAD: CAPTURE ==================
def capture_thread():
    global latest_frame, stop_flag
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue
        with frame_lock:
            latest_frame = {"frame": frame, "ts": time.time()}
    cap.release()

# ================== THREAD: INFERENCE ==================
def inference_thread():
    global latest_draw
    last_ts = 0

    while not stop_flag:
        with frame_lock:
            if latest_frame is None or latest_frame["ts"] == last_ts:
                frame = None
            else:
                frame = latest_frame["frame"].copy()
                ts = latest_frame["ts"]
                last_ts = ts

        if frame is None:
            time.sleep(0.005)
            continue

        detections = yolo(frame)
        det_bboxes = [d["bbox"] for d in detections]
        tracks = tracker.update(det_bboxes)

        tracks = sorted(
            tracks,
            key=lambda t: (t["bbox"][2]-t["bbox"][0])*(t["bbox"][3]-t["bbox"][1]),
            reverse=True
        )

        now = time.time()
        h, w = frame.shape[:2]

        for i, t in enumerate(tracks):
            tid = t["id"]
            x1,y1,x2,y2 = map(int, t["bbox"])
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w,x2), min(h,y2)
            t["bbox"] = [x1,y1,x2,y2]

            if x2<=x1 or y2<=y1:
                continue

            if i >= MAX_CLASSIFY and tid not in cls_cache:
                continue

            if tid in cls_cache and now - cls_cache[tid]["ts"] < CLS_REFRESH_SEC:
                t.update(cls_cache[tid])
                continue

            face = frame[y1:y2, x1:x2]
            age, age_c, race, race_c = clf(face)
            cls_cache[tid] = {
                "age": age,
                "age_conf": age_c,
                "race": race,
                "race_conf": race_c,
                "ts": now
            }
            t.update(cls_cache[tid])

        for t in tracks:
            x1,y1,x2,y2 = t["bbox"]
            label = f"ID:{t['id']}"
            if "age" in t:
                label += f" A:{t['age']} R:{t['race']}"
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,label,(x1,max(0,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        with draw_lock:
            latest_draw = {"frame": frame, "ts": ts, "tracks": tracks}

# ================== THREAD: DISPLAY ==================
def display_thread():
    global stop_flag
    fps_hist = []

    while not stop_flag:
        with draw_lock:
            if latest_draw is None:
                data = None
            else:
                data = {
                    "frame": latest_draw["frame"].copy(),
                    "ts": latest_draw["ts"],
                    "tracks": latest_draw["tracks"]
                }

        if data is None:
            time.sleep(0.005)
            continue

        now = time.time()
        latency = now - data["ts"]
        face_count = len(data["tracks"])

        if face_count >= 1:
            fps_hist.append(1.0 / max(latency, 1e-6))
            if len(fps_hist) > FPS_HIST_LEN:
                fps_hist.pop(0)

        fps_avg = sum(fps_hist)/len(fps_hist) if fps_hist else 0.0

        fps_logger.update(latency, face_count)

        cv2.putText(data["frame"], f"E2E Latency: {latency*1000:.1f} ms", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(data["frame"], f"E2E FPS(avg): {fps_avg:.1f}", (10,55),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        cv2.imshow("Age & Race Realtime (Optimized)", data["frame"])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            fps_logger.save()
            stop_flag = True
            break

    cv2.destroyAllWindows()

# ================== MAIN ==================
def main():
    t1 = threading.Thread(target=capture_thread)
    t2 = threading.Thread(target=inference_thread)
    t3 = threading.Thread(target=display_thread)

    t1.start(); t2.start(); t3.start()
    t3.join()
    t1.join(); t2.join()

if __name__ == "__main__":
    main()
