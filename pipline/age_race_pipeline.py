import cv2
import time
import threading
import queue
import numpy as np

from load_model import load_tracking_model, load_age_race_classifier
from write_log import FPSStressLogger

# ================== CONFIG ==================
CAMERA_ID = 0
FRAME_W, FRAME_H = 640, 340 

DETECT_EVERY_N_FRAMES = 3  

# tracking
IOU_THRES = 0.4
MAX_LOST_SEC = 0.8

# classification async
MAX_CLASSIFY = 1
CLS_REFRESH_SEC = 3.0
CLS_QUEUE_MAX = 8
FACE_MIN_SIZE = 20

# display
FPS_HIST_LEN = 30
WINDOW_NAME = "Age & Race Realtime (Detect Skip)"
AGE_CLASSES = [
    "0-10",    # index 0
    "11-19",   # index 1
    "20-30",   # index 2
    "31-40",   # index 3
    "41-50",   # index 4
    "51-69",   # index 5
    "70+"      # index 6
]
RACE_CLASSES = [
    "Asian",   # index 0
    "White",   # index 1
    "Black",   # index 2
    "Indian",  # index 3
    "Others"   # index 4
]


# logger
fps_logger = FPSStressLogger(window_sec=5 * 60, log_file="stress_test_fps_log.txt")

# ================== SHARED ==================
latest_frame = None
latest_draw = None
stop_flag = False

frame_lock = threading.Lock()
draw_lock = threading.Lock()

cls_cache = {}
cls_lock = threading.Lock()

cls_queue = queue.Queue(maxsize=CLS_QUEUE_MAX)

# ================== TRACKER ==================
class IoUTracker:
    def __init__(self, iou_thres=0.4, max_lost_sec=0.8):
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
            best_iou, best_idx = 0, -1
            for i, d in enumerate(detections):
                if i in used:
                    continue
                v = self.iou(t["bbox"], d)
                if v > best_iou:
                    best_iou, best_idx = v, i
            if best_iou >= self.iou_thres and best_idx >= 0:
                self.tracks[tid]["bbox"] = detections[best_idx]
                self.tracks[tid]["last_seen"] = now
                used.add(best_idx)

        for i, d in enumerate(detections):
            if i not in used:
                self.tracks[self.next_id] = {"bbox": d, "last_seen": now}
                self.next_id += 1

        for tid in list(self.tracks.keys()):
            if now - self.tracks[tid]["last_seen"] > self.max_lost_sec:
                del self.tracks[tid]
                with cls_lock:
                    cls_cache.pop(tid, None)

        return [{"id": tid, "bbox": t["bbox"]} for tid, t in self.tracks.items()]

tracker = IoUTracker(IOU_THRES, MAX_LOST_SEC)

# ================== LOAD MODELS ==================
print("Loading models...")
yolo = load_tracking_model()
clf = load_age_race_classifier()
print("Models loaded!")

# ================== CAPTURE ==================
def capture_thread():
    global latest_frame
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    while not stop_flag:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = {"frame": frame, "ts": time.time()}
        else:
            time.sleep(0.005)

    cap.release()

# ================== CLASSIFIER ==================
def classifier_thread():
    while not stop_flag:
        try:
            tid, face = cls_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            age, ac, race, rc = clf(face)
            with cls_lock:
                cls_cache[tid] = {
                    "age": age,
                    "age_conf": ac,
                    "race": race,
                    "race_conf": rc,
                    "ts": time.time()
                }
        except:
            pass

# ================== DETECT + TRACK ==================
def detect_track_draw_thread():
    global latest_draw
    frame_idx = 0
    last_dets = []

    while not stop_flag:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame["frame"].copy()
            ts = latest_frame["ts"]

        h, w = frame.shape[:2]

        # 🔥 detect 1/5 frame
        if frame_idx % DETECT_EVERY_N_FRAMES == 0:
            dets = []
            for d in yolo(frame):
                x1,y1,x2,y2 = map(int, d["bbox"])
                if (x2-x1)>=FACE_MIN_SIZE and (y2-y1)>=FACE_MIN_SIZE:
                    dets.append([x1,y1,x2,y2])
            last_dets = dets
        else:
            dets = last_dets

        tracks = tracker.update(dets)

        # classify async
        now = time.time()
        tracks_sorted = sorted(tracks,
            key=lambda t:(t["bbox"][2]-t["bbox"][0])*(t["bbox"][3]-t["bbox"][1]),
            reverse=True)

        for t in tracks_sorted[:MAX_CLASSIFY]:
            tid = t["id"]
            x1,y1,x2,y2 = t["bbox"]
            with cls_lock:
                cached = cls_cache.get(tid)
            if cached and now-cached["ts"]<CLS_REFRESH_SEC:
                continue
            face = frame[y1:y2,x1:x2].copy()
            try:
                cls_queue.put_nowait((tid, face))
            except:
                pass

        # draw
        for t in tracks_sorted:
            x1,y1,x2,y2 = t["bbox"]
            tid = t["id"]
            with cls_lock:
                cached = cls_cache.get(tid)
            label = f"ID:{tid}"
            if cached: 
                label += f" Age:{RACE_CLASSES[cached['age']]} Conf:{cached['age_conf']:.2f}  | Race:{RACE_CLASSES[cached['race']]} Conf:{cached['race_conf']:.2f}"
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,label,(x1,max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)
            

        with draw_lock:
            latest_draw = {"frame": frame, "ts": ts, "tracks": tracks_sorted}

        frame_idx += 1

# ================== DISPLAY ==================
def display_thread():
    fps_hist = []
    while not stop_flag:
        with draw_lock:
            if latest_draw is None:
                continue
            frame = latest_draw["frame"].copy()
            ts = latest_draw["ts"]
            face_count = len(latest_draw["tracks"])

        latency = time.time() - ts
        if face_count>0:
            fps_hist.append(1/max(latency,1e-6))
            if len(fps_hist)>FPS_HIST_LEN:
                fps_hist.pop(0)

        fps_logger.update(latency, face_count)

        cv2.putText(frame,f"E2E Latency: {latency*1000:.1f} ms",(10,20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
        cv2.putText(frame,f"E2E FPS(avg): {sum(fps_hist)/len(fps_hist):.1f}",(10,45),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            fps_logger.save(write_raw=False)
            break

    cv2.destroyAllWindows()

# ================== MAIN ==================
def main():
    threading.Thread(target=capture_thread, daemon=True).start()
    threading.Thread(target=classifier_thread, daemon=True).start()
    threading.Thread(target=detect_track_draw_thread, daemon=True).start()
    display_thread()

if __name__=="__main__":
    main()
