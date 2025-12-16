
# # import sys
# # import os
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # import cv2
# # import threading
# # import queue
# # import time
# # import numpy as np

# # # Chọn backend: 'onnx' hoặc 'torch'
# # USE_ONNX = True  # Đổi True nếu muốn dùng ONNX cho cả YOLO và classifier
# # if USE_ONNX:
# #     print("Using ONNX backend for both YOLO and Classifier")
# # else:
# #     print("Using PyTorch backend for both YOLO and Classifier")

# # if USE_ONNX:
# #     from pipline.load_yolo_onnx import load_yolo_onnx
# #     from pipline.load_onnx_classifier import load_onnx_classifier
# # else:
# #     from pipline.load_model import load_face_detector, load_age_race_classifier


# # # Config
# # CAMERA_ID = 0  # or path to video file
# # DISPLAY = True

# # # Age and Race class names
# # AGE_CLASSES = [
# #     '0-10', '11-19', '20-30', '31-40', '41-50', '51-69', '70+'
# # ]
# # RACE_CLASSES = [
# #     'Asian', 'White', 'Black', 'Indian', 'Others'
# # ]

# # # Thread-safe queues
# # frame_queue = queue.Queue(maxsize=5)
# # det_queue = queue.Queue(maxsize=5)
# # result_queue = queue.Queue(maxsize=5)


# # # Load models (do this once at start)
# # if USE_ONNX:
# #     face_detector = load_yolo_onnx("checkpoint/yolov11n-face.onnx")
# #     age_race_classifier = load_onnx_classifier("checkpoint/age_race_multihead.onnx")
# # else:
# #     face_detector = load_face_detector()
# #     age_race_classifier = load_age_race_classifier()

# # def reader_thread():
# #     cap = cv2.VideoCapture(CAMERA_ID)
# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break
# #         if not frame_queue.full():
# #             frame_queue.put(frame)
# #         else:
# #             time.sleep(0.01)
# #     cap.release()


# # def detection_thread():
# #     while True:
# #         frame = frame_queue.get()
# #         bboxes, scores = face_detector(frame)
# #         if len(bboxes) > 0:
# #             det_queue.put((frame, bboxes, scores))
# #         else:
# #             # No face detected, still forward frame for display (optional)
# #             det_queue.put((frame, [], []))



# # def classification_thread():
# #     while True:
# #         frame, bboxes, scores = det_queue.get()
# #         results = []
# #         for bbox, score in zip(bboxes, scores):
# #             try:
# #                 x1, y1, x2, y2 = map(int, bbox)
# #                 face_img = frame[y1:y2, x1:x2]
# #                 if face_img.size == 0 or (x2-x1)<=0 or (y2-y1)<=0:
# #                     continue
# #                 age_idx, age_conf, race_idx, race_conf = age_race_classifier(face_img)
# #                 results.append({
# #                     'bbox': bbox,
# #                     'score': score,
# #                     'age': AGE_CLASSES[age_idx],
# #                     'age_conf': age_conf,
# #                     'race': RACE_CLASSES[race_idx],
# #                     'race_conf': race_conf
# #                 })
# #             except Exception as e:
# #                 print(f"Classification error: {e}")
# #         result_queue.put((frame, results))

# # def display_thread():
# #     fps_list = []
# #     last_time = time.time()
# #     while True:
# #         frame, results = result_queue.get()
# #         # Nếu không có kết quả classification, vẫn vẽ bbox YOLO (chỉ có bbox, score)
# #         if not results:
# #             # Lấy bbox từ det_queue (nếu có)
# #             pass  # Không có bbox nào, không vẽ
# #         else:
# #             for res in results:
# #                 x1, y1, x2, y2 = map(int, res['bbox'])
# #                 label = f"{res.get('age', '')} ({res.get('age_conf', 0)*100:.1f}%) - {res.get('race', '')} ({res.get('race_conf', 0)*100:.1f}%)"
# #                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
# #                 cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
# #         # FPS
# #         now = time.time()
# #         fps = 1/(now-last_time)
# #         fps_list.append(fps)
# #         last_time = now
# #         cv2.putText(frame, f"FPS: {fps:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
# #         if DISPLAY:
# #             cv2.imshow('Age & Race Identification', frame)
# #             # Nhấn 'q' để thoát
# #             if cv2.waitKey(1) & 0xFF == ord('q'):
# #                 cv2.destroyAllWindows()
# #                 # Save FPS log
# #                 with open('fps_log.txt', 'w') as f:
# #                     for v in fps_list:
# #                         f.write(f"{v}\n")
# #                 import os
# #                 os._exit(0)  # Dừng toàn bộ chương trình ngay lập tức

# # def main():
# #     threads = [
# #         threading.Thread(target=reader_thread, daemon=True),
# #         threading.Thread(target=detection_thread, daemon=True),
# #         threading.Thread(target=classification_thread, daemon=True),
# #         threading.Thread(target=display_thread, daemon=True)
# #     ]
# #     for t in threads:
# #         t.start()
# #     for t in threads:
# #         t.join()

# # if __name__ == '__main__':
# #     main()

# import sys
# import os
# import cv2
# import threading
# import time
# import numpy as np

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from classification.multi_head import MultiHeadClassifier

# # ================== CONFIG ==================
# BACKEND = "onnx"          # "pt" | "onnx"
# CAMERA_ID = 0
# FRAME_W, FRAME_H = 640, 480
# MAX_FACES = 3
# CLASSIFY_EVERY_N_FRAMES = 5
# fps_hist = []
# FPS_HIST_LEN = 30   # trung bình 30 frame

# AGE_CLASSES = ['0-10','11-19','20-30','31-40','41-50','51-69','70+']
# RACE_CLASSES = ['Asian','White','Black','Indian','Others']

# # ================== SHARED STATE ==================
# latest_frame = None        # (frame, ts)
# latest_overlays = []
# stop_flag = False

# frame_lock = threading.Lock()
# overlay_lock = threading.Lock()

# # ================== BACKEND LOADER ==================

# def load_backend(backend):
#     if backend == "onnx":
#         from pipline.load_yolo_onnx import load_yolo_onnx
#         from pipline.load_onnx_classifier import load_onnx_classifier

#         face_detector = load_yolo_onnx("checkpoint/yolov11n-face.onnx")
#         classifier = load_onnx_classifier("checkpoint/age_race_multihead.onnx")

#         return face_detector, classifier

#     elif backend == "pt":
#         import torch
#         from ultralytics import YOLO
#         import torchvision.transforms as T

#         device = "cuda" if torch.cuda.is_available() else "cpu"

#         yolo = YOLO("checkpoint/yolov11n-face.pt").to(device)
#         yolo.fuse()

#         clf = MultiHeadClassifier(num_age_classes=7, num_race_classes=5, backbone_name="mobilenet_v2")
#         checkpoint = torch.load("checkpoint/model_last.pth", map_location=device)
#         clf.load_state_dict(checkpoint['model_state_dict'])

#         clf.eval()

#         transform = T.Compose([
#             T.ToPILImage(),
#             T.Resize((224,224)),
#             T.ToTensor(),
#             T.Normalize([0.5]*3, [0.5]*3)
#         ])

#         def face_detector(frame):
#             res = yolo(frame, verbose=False)[0]
#             bboxes, scores = [], []
#             if res.boxes:
#                 for b in res.boxes:
#                     x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
#                     conf = float(b.conf[0].cpu())
#                     bboxes.append([x1,y1,x2,y2])
#                     scores.append(conf)
#             return bboxes, scores

#         def classifier(face_img):
#             x = transform(face_img).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 age_logits, race_logits = clf(x)
#             age_p = torch.softmax(age_logits,1)
#             race_p = torch.softmax(race_logits,1)
#             ai = age_p.argmax(1).item()
#             ri = race_p.argmax(1).item()
#             return ai, age_p[0,ai].item(), ri, race_p[0,ri].item()

#         return face_detector, classifier

#     else:
#         raise ValueError("BACKEND phải là 'pt' hoặc 'onnx'")

# # ================== LOAD BACKEND ==================
# face_detector, age_race_classifier = load_backend(BACKEND)
# print(f"Running with backend: {BACKEND}")

# # ================== THREADS ==================

# def reader_thread():
#     global latest_frame, stop_flag
#     cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

#     while not stop_flag:
#         ret, frame = cap.read()
#         if not ret:
#             continue
#         ts = time.time()
#         with frame_lock:
#             latest_frame = (frame, ts)

# def inference_thread():
#     global latest_frame, latest_overlays
#     frame_id = 0
#     cached_cls = []

#     while not stop_flag:
#         with frame_lock:
#             if latest_frame is None:
#                 continue
#             frame, ts = latest_frame
#             frame = frame.copy()

#         bboxes, scores = face_detector(frame)
#         overlays = []

#         if bboxes:
#             areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in bboxes]
#             order = np.argsort(areas)[::-1][:MAX_FACES]

#             if len(cached_cls) != len(order):
#                 cached_cls = [None]*len(order)

#             for i, idx in enumerate(order):
#                 x1,y1,x2,y2 = map(int, bboxes[idx])
#                 face = frame[y1:y2, x1:x2]
#                 if face.size == 0:
#                     continue

#                 if frame_id % CLASSIFY_EVERY_N_FRAMES == 0 or cached_cls[i] is None:
#                     ai, ac, ri, rc = age_race_classifier(face)
#                     cached_cls[i] = {
#                         "age": AGE_CLASSES[ai],
#                         "age_conf": ac,
#                         "race": RACE_CLASSES[ri],
#                         "race_conf": rc
#                     }

#                 overlays.append({
#                     "bbox": (x1,y1,x2,y2),
#                     "cls": cached_cls[i],
#                     "ts": ts
#                 })

#         with overlay_lock:
#             latest_overlays = overlays

#         frame_id += 1

# def display_thread():
#     while not stop_flag:
#         with frame_lock:
#             if latest_frame is None:
#                 continue
#             frame,_ = latest_frame
#             frame = frame.copy()

#         with overlay_lock:
#             overlays = latest_overlays

#         for ov in overlays:
            
#             x1,y1,x2,y2 = ov["bbox"]
#             cls = ov["cls"]
#             latency = time.time() - ov["ts"]

#             latency = time.time() - ov["ts"]
#             fps_e2e = 1.0 / latency if latency > 0 else 0

#             fps_hist.append(fps_e2e)
#             if len(fps_hist) > FPS_HIST_LEN:
#                 fps_hist.pop(0)

#             label = f"{cls['age']} | {cls['race']} | {latency*1000:.1f}ms"
#             cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
#             cv2.putText(frame,label,(x1,y1-10),
#                         cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
#             if fps_hist:
#                 fps_avg = sum(fps_hist) / len(fps_hist)

#                 cv2.putText(
#                     frame,
#                     f"E2E FPS: {fps_avg:.1f}",
#                     (10, 30),                     # 👈 góc trên
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1.0,
#                     (0, 0, 255),
#                     2
#                 )


#         cv2.imshow("Flexible Backend Demo", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             os._exit(0)

# # ================== MAIN ==================

# threads = [
#     threading.Thread(target=reader_thread, daemon=True),
#     threading.Thread(target=inference_thread, daemon=True),
#     threading.Thread(target=display_thread, daemon=True),
# ]

# for t in threads:
#     t.start()
# for t in threads:
#     t.join()


import sys
import os
import cv2
import threading
import time
import numpy as np

# ========= PATH =========
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ========= (PT) classifier class import =========
# Đảm bảo đường dẫn này đúng theo project của bạn
from classification.multi_head import MultiHeadClassifier

# ================== CONFIG ==================
BACKEND = "pt"  # "pt" | "onnx"

CAMERA_ID = 0
FRAME_W, FRAME_H = 640, 480

MAX_FACES = 5  # số face tối đa để hiển thị (tracking vẫn theo tất cả bbox, nhưng bạn có thể giới hạn hiển thị)
IOU_TRACKER_THRES = 0.4
TRACK_MAX_MISSES = 10          # số frame mất detection trước khi drop track
CLS_REFRESH_SEC = 1.5          # refresh age/race theo thời gian (giảm nhấp nháy, vẫn cập nhật khi thay người)
CONF_FACE_MIN = 0.7          # chỉ dùng bbox có conf >= ngưỡng này (áp dụng tốt cho PT; ONNX bạn có thể đã lọc trong loader)

FPS_HIST_LEN = 30              # trung bình trượt cho E2E FPS

DISPLAY = True
WINDOW_NAME = "Age & Race Tracking (E2E)"

AGE_CLASSES = ['0-10','11-19','20-30','31-40','41-50','51-69','70+']
RACE_CLASSES = ['Asian','White','Black','Indian','Others']

# ================== SHARED STATE ==================
latest_frame = None      # dict: {"frame": np.ndarray, "ts_capture": float}
latest_draw = None       # dict: {"frame": np.ndarray, "ts_capture": float, "ts_display_ready": float, "tracks": list}
stop_flag = False

frame_lock = threading.Lock()
draw_lock = threading.Lock()

fps_hist = []  # end-to-end fps history


# ================== TRACKER ==================
class IoUTracker:
    """IoU tracker nhẹ: gán ID ổn định dựa trên IoU bbox"""
    def __init__(self, iou_thres=0.4, max_misses=10):
        self.iou_thres = iou_thres
        self.max_misses = max_misses
        self.tracks = {}   # id -> {"bbox": [x1,y1,x2,y2], "misses": int}
        self.next_id = 1

    @staticmethod
    def _iou(a, b):
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        return inter / (areaA + areaB - inter + 1e-6)

    def update(self, detections):
        """
        detections: list of bbox [x1,y1,x2,y2]
        return: dict id -> bbox
        """
        # tăng misses cho tất cả track trước
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["misses"] += 1

        used_det = set()

        # match existing tracks
        for tid, t in list(self.tracks.items()):
            best_iou = 0.0
            best_idx = -1
            for i, det in enumerate(detections):
                if i in used_det:
                    continue
                iou = self._iou(t["bbox"], det)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= self.iou_thres and best_idx >= 0:
                self.tracks[tid]["bbox"] = detections[best_idx]
                self.tracks[tid]["misses"] = 0
                used_det.add(best_idx)

        # create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i in used_det:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"bbox": det, "misses": 0}

        # drop stale tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["misses"] > self.max_misses:
                del self.tracks[tid]

        # output stable mapping
        return {tid: self.tracks[tid]["bbox"] for tid in self.tracks}


# ================== BACKEND LOADER ==================
def load_backend(backend: str):
    if backend == "onnx":
        from pipline.load_yolo_onnx import load_yolo_onnx
        from pipline.load_onnx_classifier import load_onnx_classifier

        # NOTE: Nếu loader ONNX của bạn chưa bật thread, nên tối ưu ngay trong file loader.
        face_detector = load_yolo_onnx("checkpoint/yolov11n-face.onnx")
        classifier = load_onnx_classifier("checkpoint/age_race_multihead.onnx")
        return face_detector, classifier

    if backend == "pt":
        import torch
        from ultralytics import YOLO
        import torchvision.transforms as T

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # YOLO
        yolo = YOLO("checkpoint/yolov11n-face.pt").to(device)
        yolo.fuse()

        # Classifier (state_dict)
        clf = MultiHeadClassifier(num_age_classes=7, num_race_classes=5, backbone_name="mobilenet_v2")
        checkpoint = torch.load("checkpoint/model_last.pth", map_location=device)

        # các key có thể khác nhau tùy bạn save; bạn đã dùng 'model_state_dict' nên giữ vậy
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            clf.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            clf.load_state_dict(checkpoint["state_dict"])
        else:
            # nếu file chỉ là state_dict
            clf.load_state_dict(checkpoint)

        clf = clf.to(device)
        clf.eval()

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])

        def face_detector(frame_bgr):
            res = yolo(frame_bgr, verbose=False)[0]
            bboxes, scores = [], []
            if res.boxes:
                for b in res.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].detach().cpu().numpy()
                    conf = float(b.conf[0].detach().cpu())
                    if conf < CONF_FACE_MIN:
                        continue
                    bboxes.append([float(x1), float(y1), float(x2), float(y2)])
                    scores.append(conf)
            return bboxes, scores

        def classifier(face_bgr):
            x = transform(face_bgr).unsqueeze(0).to(device)
            with torch.no_grad():
                age_logits, race_logits = clf(x)
            age_p = torch.softmax(age_logits, dim=1)
            race_p = torch.softmax(race_logits, dim=1)
            ai = int(age_p.argmax(1).item())
            ri = int(race_p.argmax(1).item())
            return ai, float(age_p[0, ai].item()), ri, float(race_p[0, ri].item())

        return face_detector, classifier

    raise ValueError("BACKEND phải là 'pt' hoặc 'onnx'")


# ================== LOAD BACKEND ==================
face_detector, age_race_classifier = load_backend(BACKEND)
print(f"Running with backend: {BACKEND}")

tracker = IoUTracker(iou_thres=IOU_TRACKER_THRES, max_misses=TRACK_MAX_MISSES)
cls_cache = {}  # track_id -> {"age":..., "race":..., "age_conf":..., "race_conf":..., "ts_last": float}


# ================== THREADS ==================
def reader_thread():
    """Capture thread: luôn cập nhật frame mới nhất + timestamp capture"""
    global latest_frame, stop_flag

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("Không mở được camera")
        stop_flag = True
        return

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            continue

        ts_capture = time.time()
        with frame_lock:
            latest_frame = {"frame": frame, "ts_capture": ts_capture}

    cap.release()


def inference_thread():
    """
    Inference thread: đo E2E đúng chuẩn bằng cách tạo 'frame đã draw' rồi gửi sang display.
    E2E latency sẽ tính: ts_display - ts_capture.
    """
    global latest_frame, latest_draw, stop_flag

    while not stop_flag:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame["frame"].copy()
            ts_capture = latest_frame["ts_capture"]

        # ---- Detect ----
        bboxes, scores = face_detector(frame)

        # clamp + lọc bbox hợp lệ
        h, w = frame.shape[:2]
        dets = []
        for bb in bboxes:
            x1, y1, x2, y2 = map(int, bb)
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            if (x2 - x1) >= 5 and (y2 - y1) >= 5:
                dets.append([x1, y1, x2, y2])

        # ---- Tracking ----
        tracks = tracker.update(dets)

        # limit hiển thị MAX_FACES theo diện tích
        track_items = []
        for tid, bb in tracks.items():
            x1, y1, x2, y2 = bb
            area = (x2 - x1) * (y2 - y1)
            track_items.append((tid, bb, area))
        track_items.sort(key=lambda x: x[2], reverse=True)
        track_items = track_items[:MAX_FACES]

        # ---- Classify with cache per track ID ----
        now = time.time()
        results_for_draw = []

        for tid, bb, _area in track_items:
            x1, y1, x2, y2 = bb
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            need_refresh = (tid not in cls_cache) or ((now - cls_cache[tid]["ts_last"]) > CLS_REFRESH_SEC)

            if need_refresh:
                try:
                    ai, ac, ri, rc = age_race_classifier(face)
                    cls_cache[tid] = {
                        "age": AGE_CLASSES[int(ai)],
                        "age_conf": float(ac),
                        "race": RACE_CLASSES[int(ri)],
                        "race_conf": float(rc),
                        "ts_last": now,
                    }
                except Exception as e:
                    # nếu lỗi, giữ cache cũ nếu có
                    if tid not in cls_cache:
                        cls_cache[tid] = {"age": "NA", "age_conf": 0.0, "race": "NA", "race_conf": 0.0, "ts_last": now}
                    print(f"Classifier error (tid={tid}): {e}")

            results_for_draw.append({
                "id": tid,
                "bbox": (x1, y1, x2, y2),
                "cls": cls_cache[tid],
            })

        # ---- Draw (bắt buộc để đo E2E đúng định nghĩa bạn đưa) ----
        # E2E: capture -> preprocess -> detect -> classify -> draw -> display
        for r in results_for_draw:
            x1, y1, x2, y2 = r["bbox"]
            cls = r["cls"]
            tid = r["id"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"ID:{tid} | {cls['age']}({cls['age_conf']*100:.0f}%) | {cls['race']}({cls['race_conf']*100:.0f}%)"
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        ts_display_ready = time.time()

        with draw_lock:
            latest_draw = {
                "frame": frame,                # đã draw xong
                "ts_capture": ts_capture,
                "ts_display_ready": ts_display_ready,
                "tracks": results_for_draw
            }


def display_thread():
    """Display thread: hiển thị frame đã draw, tính E2E latency/fps ở góc trên"""
    global latest_draw, stop_flag, fps_hist

    while not stop_flag:
        with draw_lock:
            if latest_draw is None:
                continue
            frame = latest_draw["frame"].copy()
            ts_capture = latest_draw["ts_capture"]
            ts_display_ready = latest_draw["ts_display_ready"]

        # End-to-End latency: capture -> ... -> draw -> (ngay trước display)
        # (Đây là đúng điều kiện đo bạn nêu.)
        e2e_latency = ts_display_ready - ts_capture
        e2e_fps = (1.0 / e2e_latency) if e2e_latency > 0 else 0.0

        fps_hist.append(e2e_fps)
        if len(fps_hist) > FPS_HIST_LEN:
            fps_hist.pop(0)

        fps_avg = sum(fps_hist) / len(fps_hist) if fps_hist else 0.0
        lat_ms = e2e_latency * 1000.0

        # Overlay FPS/Latency ở góc trên
        cv2.putText(frame, f"E2E FPS(avg): {fps_avg:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, f"E2E Latency: {lat_ms:.1f} ms", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Nêu rõ điều kiện đo (khi có >=1 face)
        face_count = 0
        with draw_lock:
            if latest_draw is not None:
                face_count = len(latest_draw.get("tracks", []))

        cv2.putText(frame, f"Faces: {face_count}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 255), 2)

        if DISPLAY:
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                os._exit(0)


# ================== MAIN ==================
def main():
    threads = [
        threading.Thread(target=reader_thread, daemon=True),
        threading.Thread(target=inference_thread, daemon=True),
        threading.Thread(target=display_thread, daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
