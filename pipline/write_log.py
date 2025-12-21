# write_log.py
import time
import threading
import os

class FPSStressLogger:
    """
    Sliding-window FPS logger:
    - update() nhận latency + face_count
    - giữ tối đa window_sec gần nhất
    - save() ghi Average FPS trong window
    - nếu file tồn tại: load raw records và tiếp tục update
    """

    def __init__(self, window_sec=300, log_file="stress_test_fps_log.txt"):
        self.window_sec = window_sec
        self.log_file = log_file
        self.records = []  # (ts, fps)
        self.lock = threading.Lock()
        self._load_existing()

    def _load_existing(self):
        if not os.path.exists(self.log_file):
            return
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("FPS:"):
                        _, ts, fps = line.strip().split()
                        self.records.append((float(ts), float(fps)))
        except Exception:
            self.records = []

    def update(self, e2e_latency_sec: float, face_count: int):
        if face_count < 1 or e2e_latency_sec <= 0:
            return
        fps = 1.0 / e2e_latency_sec
        now = time.time()
        with self.lock:
            self.records.append((now, fps))
            self._trim(now)

    def _trim(self, now=None):
        if now is None:
            now = time.time()
        cutoff = now - self.window_sec
        self.records = [(t, f) for (t, f) in self.records if t >= cutoff]

    def save(self, write_raw=True):
        with self.lock:
            if not self.records:
                # vẫn tạo file “có cấu trúc” để nộp, nhưng báo 0 sample
                with open(self.log_file, "w", encoding="utf-8") as f:
                    f.write("FPS Stress Test Result\n")
                    f.write(f"Window (sec): {self.window_sec}\n")
                    f.write("Samples: 0\n")
                    f.write("Average FPS (End-to-End): 0.00\n")
                print(f"[FPS-LOGGER] No samples. Log saved -> {self.log_file}")
                return

            now = time.time()
            self._trim(now)
            fps_values = [v for (_, v) in self.records]

            avg_fps = sum(fps_values) / len(fps_values)
            min_fps = min(fps_values)
            max_fps = max(fps_values)

            with open(self.log_file, "w", encoding="utf-8") as f:
                # Phần “đúng yêu cầu đề”
                f.write("FPS Stress Test Result\n")
                f.write(f"Window (sec): {self.window_sec}\n")
                f.write(f"Samples: {len(fps_values)}\n")
                f.write(f"Average FPS (End-to-End): {avg_fps:.2f}\n")
                # thêm thống kê phụ (không bắt buộc)
                f.write(f"Min FPS: {min_fps:.2f}\n")
                f.write(f"Max FPS: {max_fps:.2f}\n")

                if write_raw:
                    f.write("\n---- RAW FPS RECORDS ----\n")
                    for ts, fps in self.records:
                        f.write(f"FPS: {ts:.3f} {fps:.3f}\n")

            print(f"[FPS-LOGGER] Log saved ({len(fps_values)} samples) -> {self.log_file}")
